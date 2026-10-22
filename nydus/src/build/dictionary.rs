//! Persistent, append-only chunk dictionaries, one MDBX environment per organization.
//! Import blobs only after publishing them; builds never modify the dictionary. (by GPT-6)

use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use libmdbx::{
    Database, DatabaseOptions, Mode, NoWriteMap, ReadWriteOptions, TableFlags, WriteFlags,
};
use nydus_backend::{BlobBackend, Local};
use nydus_core::build::dictionary::{ChunkDictionary, ChunkKey, DictionaryChunk};
use nydus_core::BlobId;
use nydus_error::{Context, Error, Result};
use nydus_format::blob::{BlobFooter, BlobMetadata};
use nydus_format::erofs::EROFS_BLOCK_SIZE;
use nydus_format::utils::sha256_reader;
use nydus_storage::cache::decode_chunk_group_from_window;

const FORMAT: &[u8] = b"nydus-chunk-dictionary-v1-sha256";
const CHUNKS: &str = "chunks";
const BLOBS: &str = "blobs";

pub struct Dictionary {
    database: Database<NoWriteMap>,
}

impl std::fmt::Debug for Dictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dictionary").finish_non_exhaustive()
    }
}

#[derive(Debug, serde::Serialize)]
pub struct DictionaryBlob {
    pub id: String,
    pub blocks: u32,
    pub size: u64,
}

fn mdbx(err: libmdbx::Error) -> Error {
    Error::Io(std::io::Error::other(err)).context("chunk dictionary MDBX operation failed")
}

impl Dictionary {
    /// Opens a dictionary for writing, creating its directory if needed.
    pub fn create(path: &Path) -> Result<Self> {
        fs::create_dir_all(path)?;
        Self::open_mode(path, false)
    }

    /// Opens a dictionary for writing. The directory must already exist, so a
    /// mistyped path fails instead of becoming a new dictionary; an empty
    /// directory is initialized. (by Claude Opus 5.5)
    pub fn open(path: &Path) -> Result<Self> {
        if !path.is_dir() {
            return Err(Error::InvalidParameter(format!(
                "chunk dictionary directory does not exist: {}",
                path.display()
            )));
        }
        Self::open_mode(path, false)
    }

    pub fn open_read_only(path: &Path) -> Result<Self> {
        Self::open_mode(path, true)
    }

    fn open_mode(path: &Path, read_only: bool) -> Result<Self> {
        let database = Database::open_with_options(
            path,
            DatabaseOptions {
                max_tables: Some(2),
                mode: if read_only {
                    Mode::ReadOnly
                } else {
                    Mode::ReadWrite(ReadWriteOptions {
                        max_size: Some(1 << 40),
                        growth_step: Some(64 << 20),
                        ..Default::default()
                    })
                },
                ..Default::default()
            },
        )
        .map_err(mdbx)
        .with_context(|| format!("failed to open dictionary {}", path.display()))?;
        let dictionary = Self { database };
        if !read_only {
            let txn = dictionary.database.begin_rw_txn().map_err(mdbx)?;
            let main = txn.open_table(None).map_err(mdbx)?;
            match txn.get::<Vec<u8>>(&main, b"format").map_err(mdbx)? {
                Some(format) if format == FORMAT => {}
                Some(_) => {
                    return Err(Error::Unsupported("unknown chunk dictionary format".into()))
                }
                None => {
                    if txn.table_stat(&main).map_err(mdbx)?.entries() != 0 {
                        return Err(Error::InvalidImage("not a nydus chunk dictionary".into()));
                    }
                    txn.put(&main, b"format", FORMAT, WriteFlags::empty())
                        .map_err(mdbx)?;
                    txn.create_table(Some(CHUNKS), TableFlags::empty())
                        .map_err(mdbx)?;
                    txn.create_table(Some(BLOBS), TableFlags::empty())
                        .map_err(mdbx)?;
                }
            }
            txn.commit().map_err(mdbx)?;
        }
        let txn = dictionary.database.begin_ro_txn().map_err(mdbx)?;
        let main = txn.open_table(None).map_err(mdbx)?;
        if txn
            .get::<Vec<u8>>(&main, b"format")
            .map_err(mdbx)?
            .as_deref()
            != Some(FORMAT)
        {
            return Err(Error::Unsupported("unknown chunk dictionary format".into()));
        }
        txn.open_table(Some(CHUNKS)).map_err(mdbx)?;
        txn.open_table(Some(BLOBS)).map_err(mdbx)?;
        drop(txn);
        Ok(dictionary)
    }

    /// Atomically index the physical chunks of an immutable, published full blob.
    /// Rehashes decoded chunks because v3 metadata contains group digests only. (by GPT-6)
    pub fn add_blob(&self, path: &Path) -> Result<BlobId> {
        let mut file = File::open(path)?;
        let id = BlobId::new(sha256_reader(&mut file)?);
        // Readers resolve dependencies only as `<blob dir>/<hex digest>`, so a
        // blob indexed under any other name would make later images unreadable.
        if path.file_name() != Some(OsStr::new(&id.to_hex())) {
            return Err(Error::InvalidParameter(format!(
                "dictionary blob {} must be named by its SHA-256 digest {}",
                path.display(),
                id.to_hex()
            )));
        }
        let size = file.metadata()?.len();
        let footer = BlobFooter::from_blob_path(path)?;
        if footer.is_raw_device() {
            return Err(Error::Unsupported(
                "native EROFS blobs cannot seed a chunk dictionary".into(),
            ));
        }
        file.seek(SeekFrom::Start(footer.blob_metadata_offset()))?;
        let mut bytes = vec![
            0;
            usize::try_from(footer.blob_metadata_size()).map_err(|_| {
                Error::Overflow("blob metadata exceeds address space".into())
            })?
        ];
        file.read_exact(&mut bytes)?;
        let metadata = BlobMetadata::from_bytes(bytes)?;
        if metadata.is_incremental() || metadata.is_redirect() {
            return Err(Error::Unsupported(
                "incremental and redirect blobs cannot seed a chunk dictionary".into(),
            ));
        }
        if metadata.compressed_size() != footer.compressed_data_size() {
            return Err(Error::InvalidImage(
                "dictionary blob metadata/data size mismatch".into(),
            ));
        }
        let blocks = u32::try_from(metadata.logical_block_count())
            .map_err(|_| Error::Overflow("dictionary blob exceeds device block limit".into()))?;
        let mut blob_value = [0; 12];
        blob_value[..4].copy_from_slice(&blocks.to_le_bytes());
        blob_value[4..].copy_from_slice(&size.to_le_bytes());

        let txn = self.database.begin_rw_txn().map_err(mdbx)?;
        let blobs = txn.open_table(Some(BLOBS)).map_err(mdbx)?;
        if let Some(existing) = txn.get::<Vec<u8>>(&blobs, id.as_bytes()).map_err(mdbx)? {
            if existing != blob_value {
                return Err(Error::InvalidImage(
                    "inconsistent dictionary blob record".into(),
                ));
            }
            return Ok(id);
        }
        let chunks = txn.open_table(Some(CHUNKS)).map_err(mdbx)?;
        let backend: Arc<dyn BlobBackend> = Arc::new(Local::new(
            path.parent().unwrap_or(Path::new(".")).to_owned(),
        ));
        let mut encoded = Vec::new();
        let mut decoded = Vec::new();
        for group in metadata.chunk_groups() {
            file.seek(SeekFrom::Start(
                footer.compressed_data_offset() + group.compressed_offset(),
            ))?;
            encoded.resize(group.compressed_size() as usize, 0);
            file.read_exact(&mut encoded)?;
            let payload = decode_chunk_group_from_window(
                &metadata,
                &backend,
                &group,
                group.compressed_offset(),
                &encoded,
                &mut decoded,
            )?;
            for (offset, data) in group.chunks(&metadata, payload)? {
                let key = ChunkKey::from_data(data)?;
                let chunk = DictionaryChunk::new(
                    id,
                    u64::from(blocks),
                    offset / u64::from(EROFS_BLOCK_SIZE),
                    &key,
                )?;
                let mut value = [0; 36];
                value[..32].copy_from_slice(chunk.blob_id().as_bytes());
                value[32..].copy_from_slice(&chunk.block_address().to_le_bytes());
                match txn.put(&chunks, key.as_bytes(), value, WriteFlags::NO_OVERWRITE) {
                    Ok(()) | Err(libmdbx::Error::KeyExist) => {}
                    Err(err) => return Err(mdbx(err)),
                }
            }
        }
        txn.put(&blobs, id.as_bytes(), blob_value, WriteFlags::NO_OVERWRITE)
            .map_err(mdbx)?;
        txn.commit().map_err(mdbx)?;
        Ok(id)
    }

    /// Merge a consistent source snapshot, retaining existing chunk locations.
    /// This also creates a publishable snapshot when the destination is empty. (by GPT-6)
    pub fn merge(&self, source: &Self) -> Result<()> {
        if std::ptr::eq(self, source) {
            return Ok(());
        }
        let read = source.database.begin_ro_txn().map_err(mdbx)?;
        let write = self.database.begin_rw_txn().map_err(mdbx)?;
        let source_blobs = read.open_table(Some(BLOBS)).map_err(mdbx)?;
        let blobs = write.open_table(Some(BLOBS)).map_err(mdbx)?;
        let mut cursor = read.cursor(&source_blobs).map_err(mdbx)?;
        for entry in cursor.iter_start::<Vec<u8>, Vec<u8>>() {
            let (key, value) = entry.map_err(mdbx)?;
            if key.len() != 32 || value.len() != 12 {
                return Err(Error::InvalidImage("invalid dictionary blob record".into()));
            }
            match write.get::<Vec<u8>>(&blobs, &key).map_err(mdbx)? {
                Some(existing) if existing != value => {
                    return Err(Error::InvalidImage(
                        "inconsistent dictionary blob record".into(),
                    ))
                }
                Some(_) => {}
                None => write
                    .put(&blobs, key, value, WriteFlags::NO_OVERWRITE)
                    .map_err(mdbx)?,
            }
        }
        let source_chunks = read.open_table(Some(CHUNKS)).map_err(mdbx)?;
        let chunks = write.open_table(Some(CHUNKS)).map_err(mdbx)?;
        let mut cursor = read.cursor(&source_chunks).map_err(mdbx)?;
        for entry in cursor.iter_start::<Vec<u8>, Vec<u8>>() {
            let (key, value) = entry.map_err(mdbx)?;
            let key = ChunkKey::from_bytes(&key)?;
            if value.len() != 36 {
                return Err(Error::InvalidImage(
                    "invalid dictionary chunk record".into(),
                ));
            }
            let id = BlobId::new(value[..32].try_into().unwrap());
            let blob = read
                .get::<Vec<u8>>(&source_blobs, id.as_bytes())
                .map_err(mdbx)?
                .filter(|blob| blob.len() == 12)
                .ok_or_else(|| {
                    Error::InvalidImage(
                        "dictionary chunk references a missing or invalid blob".into(),
                    )
                })?;
            DictionaryChunk::new(
                id,
                u64::from(u32::from_le_bytes(blob[..4].try_into().unwrap())),
                u64::from(u32::from_le_bytes(value[32..].try_into().unwrap())),
                &key,
            )?;
            match write.put(&chunks, key.as_bytes(), value, WriteFlags::NO_OVERWRITE) {
                Ok(()) | Err(libmdbx::Error::KeyExist) => {}
                Err(err) => return Err(mdbx(err)),
            }
        }
        write.commit().map_err(mdbx)?;
        Ok(())
    }

    pub fn blobs(&self) -> Result<Vec<DictionaryBlob>> {
        let txn = self.database.begin_ro_txn().map_err(mdbx)?;
        let table = txn.open_table(Some(BLOBS)).map_err(mdbx)?;
        let mut cursor = txn.cursor(&table).map_err(mdbx)?;
        cursor
            .iter_start::<Vec<u8>, Vec<u8>>()
            .map(|entry| {
                let (key, value) = entry.map_err(mdbx)?;
                if key.len() != 32 || value.len() != 12 {
                    return Err(Error::InvalidImage("invalid dictionary blob record".into()));
                }
                Ok(DictionaryBlob {
                    id: hex::encode(key),
                    blocks: u32::from_le_bytes(value[..4].try_into().unwrap()),
                    size: u64::from_le_bytes(value[4..].try_into().unwrap()),
                })
            })
            .collect()
    }
}

impl ChunkDictionary for Dictionary {
    fn lookup(&self, key: &ChunkKey) -> Result<Option<DictionaryChunk>> {
        let txn = self.database.begin_ro_txn().map_err(mdbx)?;
        let chunks = txn.open_table(Some(CHUNKS)).map_err(mdbx)?;
        let Some(value) = txn.get::<Vec<u8>>(&chunks, key.as_bytes()).map_err(mdbx)? else {
            return Ok(None);
        };
        if value.len() != 36 {
            return Err(Error::InvalidImage(
                "invalid dictionary chunk record".into(),
            ));
        }
        let id = BlobId::new(value[..32].try_into().unwrap());
        let blobs = txn.open_table(Some(BLOBS)).map_err(mdbx)?;
        let blob = txn
            .get::<Vec<u8>>(&blobs, id.as_bytes())
            .map_err(mdbx)?
            .filter(|blob| blob.len() == 12)
            .ok_or_else(|| {
                Error::InvalidImage("dictionary chunk references a missing or invalid blob".into())
            })?;
        DictionaryChunk::new(
            id,
            u64::from(u32::from_le_bytes(blob[..4].try_into().unwrap())),
            u64::from(u32::from_le_bytes(value[32..].try_into().unwrap())),
            key,
        )
        .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_schema_and_dangling_or_out_of_bounds_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let dictionary = Dictionary::open(dir.path()).unwrap();
        let key = ChunkKey::from_data(&[42; 4096]).unwrap();
        let mut value = [0; 36];
        value[..32].fill(17);
        let write = dictionary.database.begin_rw_txn().unwrap();
        let chunks = write.open_table(Some(CHUNKS)).unwrap();
        write
            .put(&chunks, key.as_bytes(), value, WriteFlags::empty())
            .unwrap();
        write.commit().unwrap();
        assert!(dictionary.lookup(&key).is_err());
        let write = dictionary.database.begin_rw_txn().unwrap();
        let blobs = write.open_table(Some(BLOBS)).unwrap();
        write
            .put(&blobs, &value[..32], [0; 12], WriteFlags::empty())
            .unwrap();
        write.commit().unwrap();
        assert!(dictionary.lookup(&key).is_err());
        let write = dictionary.database.begin_rw_txn().unwrap();
        let main = write.open_table(None).unwrap();
        write
            .put(&main, b"format", b"future-version", WriteFlags::empty())
            .unwrap();
        write.commit().unwrap();
        drop(dictionary);
        assert!(Dictionary::open(dir.path()).is_err());
        assert!(Dictionary::open_read_only(dir.path()).is_err());
    }
}

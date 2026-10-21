//! Image prefetch (`prefetch.scope: all`) warms the chunk groups the mounted
//! image's reachable files use, not whole backing blobs: built images are
//! prefetched over a backend that records every byte range it serves.
//! (by Claude)

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tempfile::tempdir;

use nydus::build::merge::{merge_sources_to_bootstrap_bytes, WhiteoutSpec};
use nydus::build::{build_image, BuildImageOptions, NativeLayout};
use nydus_backend::{BlobBackend, Local, RawDeviceFile, ReadContext, ReadKind};
use nydus_config::PrefetchScope;
use nydus_core::ErofsReader;
use nydus_format::blob::{BlobMetadata, BlobMetadataCompressor};
use nydus_format::erofs::{ZErofsCompressor, EROFS_BLOCK_SIZE};
use nydus_storage::prefetch::{BlobPrefetcher, ImageExtents};

const BLOCK: u64 = EROFS_BLOCK_SIZE as u64;

struct Read {
    blob_id: [u8; 32],
    range: Range<u64>,
    kind: ReadKind,
}

/// A local backend that records every data read and can refuse them all.
struct RecordingBackend {
    local: Local,
    reads: Mutex<Vec<Read>>,
    refuse: AtomicBool,
}

impl RecordingBackend {
    fn new(dir: &Path) -> Arc<Self> {
        Arc::new(Self {
            local: Local::new(dir.to_path_buf()),
            reads: Mutex::new(Vec::new()),
            refuse: AtomicBool::new(false),
        })
    }

    /// The data-region ranges read from `blob_id`, as `(range, kind)`.
    fn reads_of(&self, blob_id: &[u8; 32]) -> Vec<(Range<u64>, ReadKind)> {
        self.reads
            .lock()
            .unwrap()
            .iter()
            .filter(|read| &read.blob_id == blob_id)
            .map(|read| (read.range.clone(), read.kind))
            .collect()
    }

    fn read_count(&self) -> usize {
        self.reads.lock().unwrap().len()
    }
}

impl BlobBackend for RecordingBackend {
    fn blob_metadata(&self, blob_id: &[u8; 32]) -> io::Result<BlobMetadata> {
        self.local.blob_metadata(blob_id)
    }

    fn is_raw_device(&self, blob_id: &[u8; 32]) -> io::Result<bool> {
        self.local.is_raw_device(blob_id)
    }

    fn raw_device_file(&self, blob_id: &[u8; 32]) -> io::Result<Option<RawDeviceFile>> {
        self.local.raw_device_file(blob_id)
    }

    fn read_range_into(
        &self,
        blob_id: &[u8; 32],
        offset: u64,
        dst: &mut [u8],
        context: ReadContext,
    ) -> io::Result<()> {
        self.reads.lock().unwrap().push(Read {
            blob_id: *blob_id,
            range: offset..offset + dst.len() as u64,
            kind: context.kind,
        });
        if self.refuse.load(Ordering::SeqCst) {
            return Err(io::Error::other("backend read refused by the test"));
        }
        self.local.read_range_into(blob_id, offset, dst, context)
    }
}

/// Deterministic incompressible-ish bytes, never an all-zero block.
fn pattern(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as u8) | (index % 4096 == 0) as u8
        })
        .collect()
}

/// A built layer: its full blob in the store, standalone bootstrap and meta.
struct Layer {
    blob_id: [u8; 32],
    blob_path: PathBuf,
    bootstrap: PathBuf,
    meta: Option<BlobMetadata>,
}

fn build_layer(source: &Path, store: &Path, native: Option<NativeLayout>) -> Layer {
    // One chunk per block and one group per whole chunk, so each file's
    // data lands in groups of its own (short tails share packed groups).
    let options = BuildImageOptions::new(
        source.to_path_buf(),
        EROFS_BLOCK_SIZE,
        BlobMetadataCompressor::Zstd,
        HashSet::new(),
        true,
    )
    .unwrap()
    .with_chunk_group_min_size(EROFS_BLOCK_SIZE)
    .unwrap();
    let options = match native {
        Some(layout) => options.with_native(layout, 0).unwrap(),
        None => options,
    };
    let mut blob = Vec::new();
    let image = build_image(&options, &mut blob).unwrap();
    let blob_path = store.join(hex::encode(image.full_blob_digest));
    fs::write(&blob_path, blob).unwrap();
    let bootstrap = source.with_extension("bootstrap");
    fs::write(&bootstrap, image.standalone_bootstrap.unwrap()).unwrap();
    Layer {
        blob_id: image.full_blob_digest,
        blob_path,
        bootstrap,
        meta: image.blob_metadata,
    }
}

fn lookup(reader: &ErofsReader, path: &str) -> u64 {
    let mut nid = reader.superblock().root_nid();
    for name in path.split('/') {
        let dir = reader.inode(nid).unwrap();
        nid = reader
            .lookup_dir_entry(nid, &dir, name.as_bytes())
            .unwrap()
            .unwrap_or_else(|| panic!("{path} not found"));
    }
    nid
}

fn read_file(reader: &ErofsReader, path: &str) -> Vec<u8> {
    let nid = lookup(reader, path);
    let inode = reader.inode(nid).unwrap();
    let mut data = Vec::new();
    reader
        .write_file_data_to(nid, &inode, 0, inode.size() as u32, &mut data)
        .unwrap();
    data
}

/// The `(blob index, chunk group)` pairs file `path` occupies, resolved from
/// its raw chunk index entries and the blobs' metadata — independently of
/// the walk under test.
fn file_groups(
    reader: &ErofsReader,
    metas: &HashMap<u16, &BlobMetadata>,
    path: &str,
) -> BTreeSet<(u16, usize)> {
    let nid = lookup(reader, path);
    let inode = reader.inode(nid).unwrap();
    let chunk_size = reader.chunk_size(&inode);
    let blobs = reader.blob_infos().unwrap();
    let mut groups = BTreeSet::new();
    for (index, addr) in reader
        .read_chunk_index_entries(nid, &inode)
        .unwrap()
        .into_iter()
        .enumerate()
    {
        if addr.is_hole() {
            continue;
        }
        let (blob_index, offset) = if addr.device_id > 0 {
            (addr.device_id, addr.blkaddr * BLOCK)
        } else {
            let blob = blobs
                .iter()
                .find(|blob| {
                    (blob.mapped_blkaddr..blob.mapped_blkaddr + blob.blocks).contains(&addr.blkaddr)
                })
                .unwrap();
            (
                blob.blob_index,
                (addr.blkaddr - blob.mapped_blkaddr) * BLOCK,
            )
        };
        let len = chunk_size.min(inode.size() - index as u64 * chunk_size);
        let meta = metas[&blob_index];
        let first = meta.chunk_group_index(offset).unwrap();
        let last = meta.chunk_group_index(offset + len - 1).unwrap();
        groups.extend((first..=last).map(|group| (blob_index, group)));
    }
    groups
}

fn prefetch(reader: &Arc<ErofsReader>, scope: PrefetchScope) {
    BlobPrefetcher::new(
        reader.blob_caches(),
        reader.prefetch_plan(),
        2,
        scope,
        Duration::ZERO,
        Duration::from_millis(1),
        Duration::from_millis(1),
    )
    .run();
}

/// A two-layer merged image whose upper layer deletes a lower file that sits
/// between kept files in the lower blob: `all` prefetch fetches and caches
/// exactly the groups of reachable files (hardlinked and nested ones once,
/// holes not at all), never the deleted file's groups nor a read bridging
/// them, and needs no backend read afterwards. The lower layer mounted as
/// its own image over the same cache reuses those groups and fetches the
/// deleted file on demand.
#[test]
fn all_prefetch_warms_only_the_reachable_groups_of_a_merged_image() {
    let dir = tempdir().unwrap();
    let store = dir.path().join("store");
    fs::create_dir(&store).unwrap();
    let cache_dir = dir.path().join("cache");

    let lower_dir = dir.path().join("lower");
    fs::create_dir_all(lower_dir.join("sub")).unwrap();
    let mut files: HashMap<&str, Vec<u8>> = HashMap::new();
    files.insert("a_keep", pattern(1, 3 * 4096 + 100));
    files.insert("m_gone", pattern(2, 2 * 4096));
    let mut holes = pattern(3, 3 * 4096);
    holes[4096..8192].fill(0);
    files.insert("n_holes", holes);
    files.insert("sub/deep", pattern(4, 4096 + 1));
    files.insert("z_tail", pattern(5, 2 * 4096));
    files.insert("empty", Vec::new());
    for (path, data) in &files {
        fs::write(lower_dir.join(path), data).unwrap();
    }
    fs::hard_link(lower_dir.join("a_keep"), lower_dir.join("a_link")).unwrap();
    std::os::unix::fs::symlink("a_keep", lower_dir.join("sym")).unwrap();
    files.insert("a_link", files["a_keep"].clone());

    let upper_dir = dir.path().join("upper");
    fs::create_dir(&upper_dir).unwrap();
    fs::write(upper_dir.join(".wh.m_gone"), b"").unwrap();
    let new = pattern(6, 2 * 4096 + 5);
    fs::write(upper_dir.join("new"), &new).unwrap();

    let lower = build_layer(&lower_dir, &store, None);
    let upper = build_layer(&upper_dir, &store, None);
    let lower_meta = lower.meta.as_ref().unwrap();
    let upper_meta = upper.meta.as_ref().unwrap();
    let merged = dir.path().join("merged.bootstrap");
    fs::write(
        &merged,
        merge_sources_to_bootstrap_bytes(
            &[lower.blob_path.clone(), upper.blob_path.clone()],
            WhiteoutSpec::Oci,
        )
        .unwrap(),
    )
    .unwrap();

    let backend = RecordingBackend::new(&store);
    let reader = Arc::new(
        ErofsReader::open_bootstrap(&merged, backend.clone(), Some(&cache_dir), None).unwrap(),
    );
    let metas = HashMap::from([(1u16, lower_meta), (2u16, upper_meta)]);
    let kept = ["a_keep", "a_link", "n_holes", "sub/deep", "z_tail", "empty"];
    let mut needed = BTreeSet::new();
    for path in kept.iter().chain(&["new"]) {
        needed.extend(file_groups(&reader, &metas, path));
    }
    let lower_reader =
        ErofsReader::open_bootstrap(&lower.bootstrap, backend.clone(), None, None).unwrap();
    let gone: BTreeSet<usize> = file_groups(&lower_reader, &metas, "m_gone")
        .into_iter()
        .map(|(blob, group)| {
            assert_eq!(blob, 1);
            group
        })
        .collect();
    drop(lower_reader);
    let needed_lower: BTreeSet<usize> = needed
        .iter()
        .filter(|(blob, _)| *blob == 1)
        .map(|&(_, group)| group)
        .collect();
    // The fixture must put the deleted file's groups strictly between kept
    // groups of the same blob, or "no bridging" would prove nothing.
    assert!(gone.is_disjoint(&needed_lower));
    assert!(needed_lower.first() < gone.first() && gone.last() < needed_lower.last());
    assert_eq!(
        needed_lower.len() + gone.len(),
        lower_meta.chunk_group_count(),
        "every lower group is either kept or deleted"
    );

    // The walk itself: the hole chunk and the deleted file are left out.
    let ImageExtents::Reachable(extents) = reader.reachable_extents().unwrap() else {
        panic!("a chunk-based image resolves to ranges");
    };
    assert_eq!(
        extents.keys().copied().collect::<BTreeSet<_>>(),
        [1, 2].into()
    );

    prefetch(&reader, PrefetchScope::All);

    for (blob_index, blob_id, meta) in [
        (1u16, &lower.blob_id, lower_meta),
        (2u16, &upper.blob_id, upper_meta),
    ] {
        let cache = reader.blob_caches().cache(blob_index).unwrap();
        let reads = backend.reads_of(blob_id);
        assert!(reads.iter().all(|(_, kind)| *kind == ReadKind::Prefetch));
        let mut fetched = 0;
        for group in 0..meta.chunk_group_count() {
            let wanted = needed.contains(&(blob_index, group));
            assert_eq!(
                cache.is_chunk_group_ready(group),
                wanted,
                "blob {blob_index} group {group}"
            );
            let range = meta.chunk_group(group).unwrap().compressed_range();
            let overlapping = reads
                .iter()
                .filter(|(read, _)| read.start < range.end && range.start < read.end)
                .count();
            if wanted {
                assert!(
                    reads
                        .iter()
                        .any(|(read, _)| read.start <= range.start && range.end <= read.end),
                    "blob {blob_index} group {group} was not fetched whole"
                );
                fetched += range.end - range.start;
            } else {
                assert_eq!(
                    overlapping, 0,
                    "blob {blob_index} unused group {group} was read"
                );
            }
        }
        // Each needed group crossed the wire exactly once.
        let read_bytes: u64 = reads.iter().map(|(read, _)| read.end - read.start).sum();
        assert_eq!(read_bytes, fetched, "blob {blob_index}");
    }
    assert!(!reader.blob_caches().cache(1).unwrap().is_all_ready());
    assert!(reader.blob_caches().cache(2).unwrap().is_all_ready());

    // Everything the image can read is now served from the cache.
    backend.refuse.store(true, Ordering::SeqCst);
    let reads_after_prefetch = backend.read_count();
    for path in kept {
        assert_eq!(read_file(&reader, path), files[path], "{path}");
    }
    assert_eq!(read_file(&reader, "new"), new);
    assert_eq!(backend.read_count(), reads_after_prefetch);
    backend.refuse.store(false, Ordering::SeqCst);
    drop(reader);

    // The lower layer as an image of its own, over the same cache: the kept
    // files cost nothing, the deleted file is fetched on demand, alone.
    let other_backend = RecordingBackend::new(&store);
    let lower_image = Arc::new(
        ErofsReader::open_bootstrap(
            &lower.bootstrap,
            other_backend.clone(),
            Some(&cache_dir),
            None,
        )
        .unwrap(),
    );
    for path in ["a_keep", "n_holes", "z_tail"] {
        assert_eq!(read_file(&lower_image, path), files[path], "{path}");
    }
    assert_eq!(other_backend.read_count(), 0);
    assert_eq!(read_file(&lower_image, "m_gone"), files["m_gone"]);
    let gone_span = lower_meta
        .chunk_group(*gone.first().unwrap())
        .unwrap()
        .compressed_offset()
        ..lower_meta
            .chunk_group(*gone.last().unwrap())
            .unwrap()
            .compressed_range()
            .end;
    let reads = other_backend.reads_of(&lower.blob_id);
    assert!(!reads.is_empty());
    for (read, kind) in &reads {
        assert_eq!(*kind, ReadKind::OnDemand);
        assert!(
            gone_span.start <= read.start && read.end <= gone_span.end,
            "{read:?} beyond the deleted file's groups {gone_span:?}"
        );
    }
    let cache = lower_image.blob_caches().cache(1).unwrap();
    assert!(cache.is_all_ready());

    // Its own image prefetch then has nothing left to fetch.
    let before = other_backend.read_count();
    prefetch(&lower_image, PrefetchScope::All);
    assert_eq!(other_backend.read_count(), before);
}

/// z_erofs files are not resolved to ranges: the walk selects whole blobs,
/// which for a native layer means nothing to fetch, since its store file
/// is read in place.
#[test]
fn compressed_images_select_whole_blobs() {
    let dir = tempdir().unwrap();
    let store = dir.path().join("store");
    fs::create_dir(&store).unwrap();
    let source = dir.path().join("native");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("large"), pattern(7, 300_000)).unwrap();
    fs::write(source.join("small"), pattern(8, 700)).unwrap();
    let layer = build_layer(
        &source,
        &store,
        Some(NativeLayout::Compressed(ZErofsCompressor::Lz4)),
    );
    let backend = RecordingBackend::new(&store);
    let reader = Arc::new(
        ErofsReader::open_bootstrap(
            &layer.bootstrap,
            backend.clone(),
            Some(&dir.path().join("cache")),
            None,
        )
        .unwrap(),
    );
    assert!(matches!(
        reader.reachable_extents().unwrap(),
        ImageExtents::WholeBlobs
    ));
    prefetch(&reader, PrefetchScope::All);
    assert_eq!(backend.read_count(), 0);
    assert_eq!(read_file(&reader, "large"), pattern(7, 300_000));
}

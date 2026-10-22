//! Content identities and references used by build-time dictionaries. (by GPT-6)

use std::fmt::Debug;

use nydus_error::{Error, Result};
use nydus_format::erofs::EROFS_BLOCK_SIZE;
use sha2::{Digest, Sha256};

use crate::BlobId;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ChunkKey([u8; 36]);

impl ChunkKey {
    pub fn from_data(data: &[u8]) -> Result<Self> {
        let len = u32::try_from(data.len())
            .map_err(|_| Error::Overflow("dictionary chunk length exceeds u32".into()))?;
        if len == 0 {
            return Err(Error::InvalidParameter("empty dictionary chunk".into()));
        }
        let mut key = [0; 36];
        key[..32].copy_from_slice(&Sha256::digest(data));
        key[32..].copy_from_slice(&len.to_le_bytes());
        Ok(Self(key))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let key = Self(
            bytes
                .try_into()
                .map_err(|_| Error::InvalidImage("invalid dictionary chunk key".into()))?,
        );
        if key.length() == 0 {
            return Err(Error::InvalidImage("empty dictionary chunk key".into()));
        }
        Ok(key)
    }

    pub fn as_bytes(&self) -> &[u8; 36] {
        &self.0
    }

    pub fn length(&self) -> u32 {
        u32::from_le_bytes(self.0[32..].try_into().unwrap())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DictionaryChunk {
    blob_id: BlobId,
    blob_blocks: u32,
    block_address: u32,
}

impl DictionaryChunk {
    pub fn new(
        blob_id: BlobId,
        blob_blocks: u64,
        block_address: u64,
        key: &ChunkKey,
    ) -> Result<Self> {
        let end = block_address
            .checked_add(u64::from(key.length()).div_ceil(u64::from(EROFS_BLOCK_SIZE)));
        if blob_blocks > u64::from(u32::MAX) || end.map_or(true, |end| end > blob_blocks) {
            return Err(Error::InvalidImage(
                "dictionary chunk exceeds its blob address space".into(),
            ));
        }
        Ok(Self {
            blob_id,
            blob_blocks: blob_blocks as u32,
            block_address: block_address as u32,
        })
    }

    pub fn blob_id(&self) -> BlobId {
        self.blob_id
    }
    pub fn blob_blocks(&self) -> u32 {
        self.blob_blocks
    }
    pub fn block_address(&self) -> u32 {
        self.block_address
    }
}

pub trait ChunkDictionary: Debug + Send + Sync {
    fn lookup(&self, key: &ChunkKey) -> Result<Option<DictionaryChunk>>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DedupStats {
    pub reused_chunks: u64,
    pub reused_bytes: u64,
}

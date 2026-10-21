//! The blob data a mounted image can read: what `prefetch.scope: all` warms.
//! (by Claude)

use std::collections::{HashMap, HashSet};
use std::io;

use nydus_format::erofs::{
    EROFS_FT_DIR, EROFS_FT_REG_FILE, EROFS_INODE_CHUNK_BASED, EROFS_INODE_COMPRESSED_FULL,
    EROFS_INODE_FLAT_INLINE, EROFS_INODE_FLAT_PLAIN,
};
use nydus_storage::prefetch::ImageExtents;
use nydus_storage::ranges::BlobRangesBuilder;

use super::ErofsReader;

impl ErofsReader {
    /// Walk the tree from the root and collect, per blob, the decoded byte
    /// ranges that reachable regular files reference. Data no path reaches
    /// — files a layer merge whited out or replaced, or bytes of a shared
    /// blob only other images use — is left out; a hardlinked file counts
    /// once, and chunks several files share merge into one range.
    ///
    /// Only chunk-based files reference blob data the read path resolves
    /// chunk by chunk; flat (plain or inline) data, directories and symlinks
    /// live in the bootstrap. An image with z_erofs (compressed) files
    /// yields [`ImageExtents::WholeBlobs`] instead of resolving pclusters and
    /// packed fragments: such files only exist on native layers, which are
    /// complete local devices that prefetch never fetches. An inode the read
    /// path would reject fails the walk.
    pub fn reachable_extents(&self) -> io::Result<ImageExtents> {
        let root = self.superblock().root_nid();
        let mut ranges: HashMap<u16, BlobRangesBuilder> = HashMap::new();
        // Every other inode is reached through exactly one entry, so only
        // directories (a guard against a corrupt image linking one twice)
        // and files with several links need remembering.
        let mut seen: HashSet<u64> = HashSet::from([root]);
        let mut dirs = vec![root];
        let mut compressed = false;
        while let Some(dir_nid) = dirs.pop() {
            let dir = self.inode(dir_nid)?;
            self.for_each_dir_entry(dir_nid, &dir, |nid, file_type, name| {
                if name == b"." || name == b".." {
                    return Ok(true);
                }
                match file_type {
                    EROFS_FT_DIR => {
                        if seen.insert(nid) {
                            dirs.push(nid);
                        }
                    }
                    EROFS_FT_REG_FILE => {
                        let inode = self.inode(nid)?;
                        if inode.nlink() > 1 && !seen.insert(nid) {
                            return Ok(true);
                        }
                        match inode.data_layout() {
                            EROFS_INODE_CHUNK_BASED => {
                                self.collect_chunk_blob_ranges(nid, &inode, &mut ranges)?
                            }
                            EROFS_INODE_FLAT_PLAIN | EROFS_INODE_FLAT_INLINE => {}
                            EROFS_INODE_COMPRESSED_FULL => {
                                compressed = true;
                                return Ok(false);
                            }
                            layout => {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!("inode {nid} has unsupported data layout {layout}"),
                                ))
                            }
                        }
                    }
                    // Symlinks, devices, fifos and sockets carry no blob data.
                    _ => {}
                }
                Ok(true)
            })?;
            if compressed {
                return Ok(ImageExtents::WholeBlobs);
            }
        }
        Ok(ImageExtents::Reachable(
            ranges
                .into_iter()
                .map(|(blob_index, builder)| (blob_index, builder.build()))
                .collect(),
        ))
    }
}

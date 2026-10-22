//! Cross-image dictionary tests exercising the actual image readers and merge path. (by GPT-6)

use std::collections::HashSet;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nydus::build::dictionary::Dictionary;
use nydus::build::merge::{merge_sources_to_bootstrap_bytes, WhiteoutSpec};
use nydus::build::{build_image, build_image_from_tar_layer, BuildImageOptions, Image};
use nydus::check::{check_image, ImageKind};
use nydus_backend::Local;
use nydus_core::build::dictionary::{ChunkDictionary, ChunkKey};
use nydus_core::ErofsReader;
use nydus_format::blob::BlobMetadataCompressor;
use sha2::{Digest, Sha256};
use tempfile::tempdir;

fn options(source: &Path, compressor: BlobMetadataCompressor) -> BuildImageOptions {
    BuildImageOptions::new(source.to_owned(), 4096, compressor, HashSet::new(), true)
        .unwrap()
        .with_chunk_group_min_size(32768)
        .unwrap()
}

fn build(store: &Path, options: &BuildImageOptions, tar: Option<&Path>) -> (Image, PathBuf) {
    let mut bytes = Vec::new();
    let image = match tar {
        Some(tar) => build_image_from_tar_layer(options, tar, &mut bytes),
        None => build_image(options, &mut bytes),
    }
    .unwrap();
    let path = store.join(hex::encode(image.full_blob_digest));
    fs::write(&path, bytes).unwrap();
    (image, path)
}

fn assert_file(reader: &ErofsReader, name: &[u8], expected: &[u8]) {
    let root_nid = reader.superblock().root_nid();
    let root = reader.inode(root_nid).unwrap();
    let nid = reader
        .lookup_dir_entry(root_nid, &root, name)
        .unwrap()
        .unwrap();
    let inode = reader.inode(nid).unwrap();
    assert_eq!(inode.size(), expected.len() as u64);
    assert_eq!(
        reader
            .read_file_data(nid, &inode, 0, expected.len() as u32)
            .unwrap(),
        expected
    );
    if expected.len() > 100 {
        assert_eq!(
            reader.read_file_data(nid, &inode, 37, 63).unwrap(),
            &expected[37..100]
        );
    }
}

#[test]
fn dictionary_reuses_individual_packed_chunks_across_images_and_layers() {
    for compressor in [
        BlobMetadataCompressor::None,
        BlobMetadataCompressor::Zstd,
        BlobMetadataCompressor::Lz4Block,
    ] {
        for from_tar in [false, true] {
            let dir = tempdir().unwrap();
            let store = dir.path().join("blobs");
            let source = dir.path().join("source");
            fs::create_dir(&store).unwrap();
            fs::create_dir(&source).unwrap();
            let shared = vec![5; 4096];
            let tail = vec![6; 129];
            let new = vec![7; 4096];
            fs::write(source.join("a-unused"), vec![1; 4011]).unwrap();
            fs::write(
                source.join("b-old"),
                [shared.as_slice(), &vec![2; 4096], &tail].concat(),
            )
            .unwrap();
            let (seed, seed_path) = build(&store, &options(&source, compressor), None);
            assert_eq!(seed.blob_metadata.as_ref().unwrap().chunk_group_count(), 1);
            let dict_path = dir.path().join("org-a");
            let dictionary = Dictionary::create(&dict_path).unwrap();
            dictionary.add_blob(&seed_path).unwrap();
            dictionary.add_blob(&seed_path).unwrap();
            assert_eq!(dictionary.blobs().unwrap().len(), 1);
            drop(dictionary);
            let dictionary = Arc::new(Dictionary::open_read_only(&dict_path).unwrap());
            let location = dictionary
                .lookup(&ChunkKey::from_data(&shared).unwrap())
                .unwrap()
                .unwrap();
            assert!(location.block_address() > 0);
            assert!(dictionary
                .lookup(&ChunkKey::from_data(&tail[..128]).unwrap())
                .unwrap()
                .is_none());

            fs::remove_dir_all(&source).unwrap();
            fs::create_dir(&source).unwrap();
            let changed = [shared.as_slice(), &new, &tail].concat();
            fs::write(source.join("changed"), &changed).unwrap();
            fs::write(source.join("duplicate-new"), &new).unwrap();
            fs::hard_link(source.join("changed"), source.join("hardlink")).unwrap();
            fs::write(source.join("hole"), vec![0; 8192]).unwrap();
            let tar = dir.path().join("source.tar");
            if from_tar {
                let mut archive = tar::Builder::new(File::create(&tar).unwrap());
                for name in ["changed", "duplicate-new", "hole"] {
                    archive
                        .append_path_with_name(source.join(name), name)
                        .unwrap();
                }
                let mut header = tar::Header::new_gnu();
                header.set_path("hardlink").unwrap();
                header.set_link_name("changed").unwrap();
                header.set_entry_type(tar::EntryType::Link);
                header.set_size(0);
                header.set_mode(0o644);
                header.set_cksum();
                archive.append(&header, std::io::empty()).unwrap();
                archive.finish().unwrap();
            }
            // Reuse remains valid when the new image uses a different codec and no group digests. (by GPT-6)
            let build_options = options(&source, BlobMetadataCompressor::Zstd)
                .with_digester(None)
                .with_dictionary(dictionary.clone())
                .unwrap();
            let offline_seed = dir.path().join("offline-seed");
            fs::rename(&seed_path, &offline_seed).unwrap();
            let (image, path) = build(&store, &build_options, from_tar.then_some(tar.as_path()));
            fs::rename(&offline_seed, &seed_path).unwrap();
            assert_eq!(image.blob_metadata.as_ref().unwrap().chunk_count(), 1);
            assert_eq!(image.dedup_stats.reused_chunks, 3);
            assert_eq!(image.dedup_stats.reused_bytes, 4096 + 129 + 4096);
            assert_eq!(image.referenced_blobs.len(), 2);
            assert_eq!(image.referenced_blobs[1].as_bytes(), &seed.full_blob_digest);
            assert_eq!(
                dictionary.blobs().unwrap().len(),
                1,
                "build must not publish its output"
            );
            let bootstrap_path = dir.path().join("image.boot");
            fs::write(
                &bootstrap_path,
                image.standalone_bootstrap.as_ref().unwrap(),
            )
            .unwrap();
            let readers = [
                ErofsReader::open_blob(&path, None).unwrap(),
                ErofsReader::open_bootstrap(
                    &bootstrap_path,
                    Arc::new(Local::new(store.clone())),
                    None,
                    None,
                )
                .unwrap(),
            ];
            for reader in &readers {
                assert_file(reader, b"changed", &changed);
                assert_file(reader, b"hardlink", &changed);
                assert_file(reader, b"duplicate-new", &new);
                assert_file(reader, b"hole", &vec![0; 8192]);
            }
            let merged =
                merge_sources_to_bootstrap_bytes(&[seed_path, path.clone()], WhiteoutSpec::Oci)
                    .unwrap();
            fs::write(&bootstrap_path, merged).unwrap();
            let reader = ErofsReader::open_bootstrap(
                &bootstrap_path,
                Arc::new(Local::new(store.clone())),
                None,
                None,
            )
            .unwrap();
            assert_eq!(
                reader.blob_infos().unwrap().len(),
                2,
                "shared blobs must be registered once"
            );
            assert_file(&reader, b"changed", &changed);
            assert_file(&reader, b"a-unused", &vec![1; 4011]);
            drop(build_options);
            drop(dictionary);
            let dictionary = Arc::new(Dictionary::open(&dict_path).unwrap());
            dictionary.add_blob(&path).unwrap();
            let all_reused = options(&source, compressor)
                .with_dictionary(dictionary)
                .unwrap();
            let (image, path) = build(&store, &all_reused, None);
            assert_eq!(image.blob_footer.compressed_data_size(), 0);
            assert_eq!(image.blob_metadata.as_ref().unwrap().chunk_count(), 0);
            assert_eq!(image.referenced_blobs.len(), 3);
            assert_file(
                &ErofsReader::open_blob(&path, None).unwrap(),
                b"changed",
                &changed,
            );
        }
    }
}

#[test]
fn dictionary_isolation_merge_and_concurrent_additions() {
    let dir = tempdir().unwrap();
    let source = dir.path().join("source");
    fs::create_dir(&source).unwrap();
    let first = vec![17; 5000];
    fs::write(source.join("file"), &first).unwrap();
    let (_, first_path) = build(
        dir.path(),
        &options(&source, BlobMetadataCompressor::None),
        None,
    );
    fs::write(source.join("file"), vec![18; 5000]).unwrap();
    let (_, second_path) = build(
        dir.path(),
        &options(&source, BlobMetadataCompressor::Zstd),
        None,
    );
    let dictionary = Arc::new(Dictionary::create(&dir.path().join("org-a")).unwrap());
    std::thread::scope(|scope| {
        for path in [&first_path, &second_path] {
            let dictionary = dictionary.clone();
            scope.spawn(move || {
                dictionary.add_blob(path).unwrap();
            });
        }
    });
    assert_eq!(dictionary.blobs().unwrap().len(), 2);
    let other = Arc::new(Dictionary::create(&dir.path().join("org-b")).unwrap());
    assert!(other
        .lookup(&ChunkKey::from_data(&first[..4096]).unwrap())
        .unwrap()
        .is_none());
    let (isolated, _) = build(
        dir.path(),
        &options(&source, BlobMetadataCompressor::None)
            .with_dictionary(other.clone())
            .unwrap(),
        None,
    );
    assert_eq!(isolated.dedup_stats.reused_chunks, 0);
    assert_eq!(isolated.referenced_blobs.len(), 1);
    let snapshot_path = dir.path().join("org-a-snapshot");
    let snapshot = Dictionary::create(&snapshot_path).unwrap();
    snapshot.merge(&dictionary).unwrap();
    snapshot.merge(&dictionary).unwrap();
    drop(snapshot);
    let snapshot = Dictionary::open_read_only(&snapshot_path).unwrap();
    assert_eq!(snapshot.blobs().unwrap().len(), 2);
    let key = ChunkKey::from_data(&first[..4096]).unwrap();
    assert_eq!(
        snapshot.lookup(&key).unwrap(),
        dictionary.lookup(&key).unwrap()
    );
    assert!(other.lookup(&key).unwrap().is_none());
}

#[test]
fn dictionary_failed_import_is_atomic() {
    let dir = tempdir().unwrap();
    let source = dir.path().join("source");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("file"), [vec![3; 4096], vec![4; 4096]].concat()).unwrap();
    let (image, path) = build(
        dir.path(),
        &options(&source, BlobMetadataCompressor::None)
            .with_chunk_group_min_size(4096)
            .unwrap(),
        None,
    );
    let metadata = image.blob_metadata.unwrap();
    let mut bytes = fs::read(&path).unwrap();
    bytes[metadata.chunk_group(1).unwrap().compressed_offset() as usize] ^= 0xff;
    // Name the corrupt blob by its new digest so the import gets past the
    // file-name check and fails on the chunk-group CRC instead.
    fs::remove_file(&path).unwrap();
    let path = dir.path().join(hex::encode(Sha256::digest(&bytes)));
    fs::write(&path, bytes).unwrap();
    let dictionary = Dictionary::create(&dir.path().join("org")).unwrap();
    let err = dictionary.add_blob(&path).unwrap_err().to_string();
    assert!(!err.contains("must be named"), "{err}");
    assert!(dictionary.blobs().unwrap().is_empty());
    assert!(dictionary
        .lookup(&ChunkKey::from_data(&vec![3; 4096]).unwrap())
        .unwrap()
        .is_none());
}

#[test]
fn dictionary_registers_only_used_blobs_and_spans_device_table_blocks() {
    let dir = tempdir().unwrap();
    let source = dir.path().join("source");
    fs::create_dir(&source).unwrap();
    let dictionary = Arc::new(Dictionary::create(&dir.path().join("org")).unwrap());
    for byte in 1..=26u8 {
        fs::write(source.join("file"), vec![byte; 100]).unwrap();
        let (_, path) = build(
            dir.path(),
            &options(&source, BlobMetadataCompressor::None),
            None,
        );
        dictionary.add_blob(&path).unwrap();
    }
    fs::remove_file(source.join("file")).unwrap();
    for byte in 1..=24u8 {
        fs::write(source.join(byte.to_string()), vec![byte; 100]).unwrap();
    }
    let (image, path) = build(
        dir.path(),
        &options(&source, BlobMetadataCompressor::None)
            .with_dictionary(dictionary)
            .unwrap(),
        None,
    );
    assert_eq!(image.referenced_blobs.len(), 25);
    assert_eq!(image.blob_footer.compressed_data_size(), 0);
    let reader = ErofsReader::open_blob(&path, None).unwrap();
    assert!(reader.superblock().meta_blkaddr() > 1);
    for byte in 1..=24u8 {
        assert_file(&reader, byte.to_string().as_bytes(), &[byte; 100]);
    }
}

#[cfg(feature = "cli")]
#[test]
fn dictionary_cli_build_manifest_and_snapshot() {
    use std::process::Command;
    let dir = tempdir().unwrap();
    let run = |args: &[&str]| {
        let output = Command::new(env!("CARGO_BIN_EXE_nydus"))
            .current_dir(dir.path())
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    };
    fs::create_dir(dir.path().join("source")).unwrap();
    fs::write(dir.path().join("source/data"), vec![11; 9000]).unwrap();
    run(&["dictionary", "init", "org"]);
    run(&[
        "build",
        "source",
        "--blob-dir",
        "blobs",
        "--chunk-dict",
        "org",
        "--output-json",
        "a.json",
    ]);
    // The manifest uses the v2 `nydus-image create --output-json` schema. (by Kimi K3)
    let first: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.path().join("a.json")).unwrap()).unwrap();
    assert!(first["version"].is_string());
    let first_blobs = first["blobs"].as_array().unwrap();
    assert_eq!(first_blobs.len(), 1);
    let id = first_blobs[0].as_str().unwrap();
    let first_events = &first["trace"]["registered_events"];
    assert_eq!(first_events["blob_uncompressed_size"], 9000);
    assert!(first_events.get("dedup_chunks").is_none());
    assert!(first_events.get("dedup_uncompressed_size").is_none());
    run(&[
        "dictionary",
        "add",
        "--dictionary",
        "org",
        &format!("blobs/{id}"),
    ]);
    run(&[
        "build",
        "source",
        "--blob-dir",
        "blobs",
        "--chunk-dict",
        "org",
        "--output-json",
        "b.json",
        "--bootstrap",
        "b.boot",
    ]);
    let second: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.path().join("b.json")).unwrap()).unwrap();
    let second_events = &second["trace"]["registered_events"];
    assert_eq!(second_events["dedup_chunks"], 1);
    assert_eq!(second_events["dedup_uncompressed_size"], 9000);
    // A fully reused build writes no new data, so, as in v2, the counter is
    // absent rather than zero. (by Kimi K3)
    assert!(second_events.get("blob_uncompressed_size").is_none());
    assert!(second["blobs"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value == id));
    run(&["dictionary", "merge", "--dictionary", "snapshot", "org"]);
    fs::remove_file(dir.path().join("snapshot/mdbx.lck")).unwrap();
    let info: serde_json::Value =
        serde_json::from_slice(&run(&["dictionary", "info", "snapshot"])).unwrap();
    assert_eq!(info[0]["id"], id);
    for args in [
        vec!["build", "source", "--blob", "bad", "--chunk-dict", "absent"],
        vec![
            "build",
            "source",
            "--blob",
            "bad",
            "--chunk-dict",
            "org",
            "--compressor",
            "erofs-none",
        ],
        vec![
            "build",
            "source",
            "--blob",
            "org/mdbx.dat",
            "--chunk-dict",
            "org",
        ],
    ] {
        assert!(!Command::new(env!("CARGO_BIN_EXE_nydus"))
            .current_dir(dir.path())
            .args(args)
            .output()
            .unwrap()
            .status
            .success());
    }
    assert!(!dir.path().join("bad").exists());
    let blob_arg = format!("blobs/{id}");
    assert!(!Command::new(env!("CARGO_BIN_EXE_nydus"))
        .current_dir(dir.path())
        .args(["dictionary", "add", "--dictionary", "absent", &blob_arg])
        .output()
        .unwrap()
        .status
        .success());
    assert!(!dir.path().join("absent").exists());
    run(&["dictionary", "info", "org"]);
}

#[test]
fn dictionary_add_requires_existing_directory_and_digest_named_blob() {
    let dir = tempdir().unwrap();
    let source = dir.path().join("source");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("file"), vec![9; 5000]).unwrap();
    let (_, path) = build(
        dir.path(),
        &options(&source, BlobMetadataCompressor::None),
        None,
    );

    let missing = dir.path().join("org-typo");
    assert!(Dictionary::open(&missing).is_err());
    assert!(!missing.exists(), "open must not create the directory");

    let empty = dir.path().join("contribution");
    fs::create_dir(&empty).unwrap();
    let dictionary = Dictionary::open(&empty).unwrap();
    let renamed = dir.path().join("layer.blob");
    fs::copy(&path, &renamed).unwrap();
    let err = dictionary.add_blob(&renamed).unwrap_err().to_string();
    assert!(err.contains("must be named"), "{err}");
    assert!(dictionary.blobs().unwrap().is_empty());
    dictionary.add_blob(&path).unwrap();
    assert_eq!(dictionary.blobs().unwrap().len(), 1);
}

#[test]
fn check_verifies_dictionary_built_blob_data_region() {
    let dir = tempdir().unwrap();
    let store = dir.path().join("blobs");
    let source = dir.path().join("source");
    fs::create_dir(&store).unwrap();
    fs::create_dir(&source).unwrap();
    fs::write(source.join("shared"), vec![21; 8192]).unwrap();
    let (_, seed_path) = build(
        &store,
        &options(&source, BlobMetadataCompressor::None),
        None,
    );
    let dictionary = Arc::new(Dictionary::create(&dir.path().join("org")).unwrap());
    dictionary.add_blob(&seed_path).unwrap();
    fs::write(source.join("new"), vec![22; 8192]).unwrap();
    let (image, path) = build(
        &store,
        &options(&source, BlobMetadataCompressor::None)
            .with_dictionary(dictionary)
            .unwrap(),
        None,
    );
    assert_eq!(image.referenced_blobs.len(), 2);

    let report = check_image(ImageKind::Blob, &path, None).unwrap();
    assert_eq!(report.blobs.len(), 2);
    let mut blobs = report.blobs.values();
    let own = blobs.next().unwrap();
    assert!(own.verified, "the blob's own data region must verify");
    assert_eq!(own.resolved_path.as_deref(), Some(path.as_path()));
    assert!(
        !blobs.next().unwrap().verified,
        "dependency needs --blob-dir"
    );

    let report = check_image(ImageKind::Blob, &path, Some(&store)).unwrap();
    assert!(report.blobs.values().all(|blob| blob.verified));
}

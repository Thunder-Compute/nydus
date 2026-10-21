//! Post-mount blob prefetch: warms the local caches with the ondemand blob
//! declared in the image and, under the `all` scope, with the data the
//! mounted image can reach, so on-demand reads hit the cache instead of the
//! backend.

use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tracing::{info, warn};

use nydus_backend::is_backend_throttled;
use nydus_config::PrefetchScope;

use crate::cache::{BlobCaches, GroupSelection};
use crate::ranges::BlobRanges;

/// How often the reschedule wait re-checks the stop flag while sleeping
/// towards the next deadline.
const RESCHEDULE_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Drives blob-level prefetch after a nydus filesystem is mounted.
///
/// Workflow:
/// 1. Resolve [`PrefetchScope::Auto`] from the priority blobs' metadata:
///    [`PrefetchScope::Ondemand`] when one of them is an "ondemand" (REDIRECT)
///    blob, otherwise [`PrefetchScope::All`].
/// 2. Stream the ondemand blobs declared in the root
///    `trusted.nydus.prefetch.blobs` xattr sequentially (single thread) into
///    their source blobs' caches. Under `ondemand` nothing else is pulled.
/// 3. Under `all`, resolve the image's [`ImageExtents`] — the data its
///    reachable files reference — then warm the chunk groups holding it: the
///    other priority blobs sequentially in declared order, then the remaining
///    blobs concurrently with a worker pool. Groups no reachable file needs
///    (data that layer merges or deletions made unreachable, or that only
///    other images sharing the blob use) are not fetched. Under the other
///    scopes, only open every blob's cache (blob meta, sparse files) in the
///    background, alongside step 2, so the backend bandwidth stays focused on
///    the access-ordered hot set while no later read pays the metadata round
///    trips.
/// 4. Blobs whose prefetch the backend throttled (Dragonfly `429`, detected
///    via [`is_backend_throttled`]) are rescheduled after a random delay in
///    the configured window and re-attempted until they stop being throttled
///    or the [stop flag](Self::stop_flag) is raised. Other failures are
///    logged and skipped.
pub struct BlobPrefetcher {
    caches: Arc<BlobCaches>,
    priority: Vec<u16>,
    rest: Vec<u16>,
    /// Taken (and run, under `all`) once by [`run`](Self::run).
    extents: Option<ImageExtentsResolver>,
    threads: usize,
    scope: PrefetchScope,
    /// Per-blob prefetch timeout; `0s` disables the bound.
    timeout: Duration,
    /// The `[min, max]` window for the random delay before a throttled blob
    /// prefetch is re-attempted.
    retry_delay_min: Duration,
    retry_delay_max: Duration,
    /// Cooperative stop flag: once raised, no new prefetch work is started
    /// and the reschedule loop exits.
    stop: Arc<AtomicBool>,
}

/// The blob data a mounted image can read, which `all` prefetch warms.
#[derive(Debug)]
pub enum ImageExtents {
    /// Per blob index, the decoded byte ranges that the image's reachable
    /// regular files reference, each byte once however many hardlinks or
    /// files share it. A blob without an entry holds no reachable data.
    Reachable(HashMap<u16, BlobRanges>),
    /// Every byte of every blob: the conservative superset, used when the
    /// image's data is not resolved to ranges (z_erofs inodes, whose
    /// pclusters live on native layers that prefetch never fetches anyway,
    /// or a metadata walk that failed).
    WholeBlobs,
}

impl ImageExtents {
    /// The chunk groups of the blob identified by `blob_index` that hold
    /// image data: under [`Reachable`](Self::Reachable) only those its
    /// reachable files reference (none for a blob without an entry), under
    /// [`WholeBlobs`](Self::WholeBlobs) every group.
    pub fn selection(&self, blob_index: u16) -> GroupSelection<'_> {
        match self {
            ImageExtents::WholeBlobs => GroupSelection::All,
            ImageExtents::Reachable(blobs) => match blobs.get(&blob_index) {
                Some(ranges) if !ranges.is_empty() => GroupSelection::Ranges(ranges),
                _ => GroupSelection::None,
            },
        }
    }
}

/// Computes the [`ImageExtents`] of the mounted image. Run at most once, on
/// the prefetch thread, so the metadata walk behind it never delays the
/// mount or an ondemand-only prefetch.
pub type ImageExtentsResolver = Box<dyn FnOnce() -> io::Result<ImageExtents> + Send>;

/// The blob prefetch order: `priority` blobs stream first, sequentially and
/// in declared order; `rest` follows through the worker pool when the scope
/// is [`PrefetchScope::All`], which warms only what `extents` selects.
pub struct PrefetchPlan {
    /// Blob indexes to warm first, in declared order.
    pub priority: Vec<u16>,
    /// The remaining blob indexes, in ascending order.
    pub rest: Vec<u16>,
    /// The data of the plain (non-ondemand) blobs that `all` warms.
    pub extents: ImageExtentsResolver,
}

impl BlobPrefetcher {
    /// `plan` is typically the result of `ErofsReader::prefetch_plan`, and
    /// `blobs` the matching cache set (`ErofsReader::blob_caches`).
    /// `retry_delay_min ..= retry_delay_max` is the window for the random
    /// delay before a throttled blob prefetch is re-attempted.
    pub fn new(
        caches: Arc<BlobCaches>,
        plan: PrefetchPlan,
        threads: usize,
        scope: PrefetchScope,
        timeout: Duration,
        retry_delay_min: Duration,
        retry_delay_max: Duration,
    ) -> Self {
        Self {
            caches,
            priority: plan.priority,
            rest: plan.rest,
            extents: Some(plan.extents),
            threads: threads.max(1),
            scope,
            timeout,
            retry_delay_min,
            retry_delay_max: retry_delay_max.max(retry_delay_min),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    /// A handle to this prefetcher's stop flag. Storing `true` makes the
    /// prefetcher stop starting new work and exit its reschedule loop, so a
    /// detached prefetch thread can be wound down on unmount.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Run the plan's extents resolver. A failed walk falls back to whole
    /// blobs: warming too much is better than silently skipping data the
    /// image needs.
    fn resolve_extents(&mut self) -> ImageExtents {
        let resolve = self
            .extents
            .take()
            .expect("image extents are resolved once");
        let start = Instant::now();
        match resolve() {
            Ok(ImageExtents::Reachable(blobs)) => {
                info!(
                    "image prefetch selects {} reachable bytes in {} blobs (resolved in {:.3?})",
                    blobs.values().map(BlobRanges::len_bytes).sum::<u64>(),
                    blobs.len(),
                    start.elapsed(),
                );
                ImageExtents::Reachable(blobs)
            }
            Ok(ImageExtents::WholeBlobs) => {
                info!("image data is not resolved to ranges, prefetching whole blobs");
                ImageExtents::WholeBlobs
            }
            Err(err) => {
                warn!(
                    "failed to resolve the image's reachable data, prefetching whole blobs: {err}"
                );
                ImageExtents::WholeBlobs
            }
        }
    }

    /// Open every blob's cache on a small worker pool, priority blobs first,
    /// so the blob meta round trips overlap each other and the priority
    /// prefetch instead of being paid one blob at a time — by the phase 1
    /// `is_redirect` checks (an optimized image lists every blob in its
    /// prefetch xattr) or by the first reads. Returns the workers' handles.
    fn spawn_blob_openers(&self) -> Vec<JoinHandle<()>> {
        // Popped from the back, so reverse to hand out the priority blobs first.
        let mut queue: Vec<u16> = self.priority.iter().chain(&self.rest).copied().collect();
        queue.reverse();
        let worker_count = self.threads.min(queue.len());
        let queue = Arc::new(Mutex::new(queue));
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let blobs = self.caches.clone();
            let queue = queue.clone();
            let stop = self.stop.clone();
            let handle = thread::Builder::new()
                .name("nydus_blob_open".to_string())
                .spawn(move || loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let blob_index = queue.lock().unwrap().pop();
                    match blob_index {
                        Some(blob_index) => {
                            if let Err(err) = blobs.cache(blob_index) {
                                warn!("failed to open blob {} cache: {}", blob_index, err);
                            }
                        }
                        None => break,
                    }
                });
            match handle {
                Ok(handle) => handles.push(handle),
                Err(err) => warn!("failed to spawn blob open worker: {}", err),
            }
        }
        handles
    }

    /// Spawn a background thread that drives the whole prefetch workflow. The
    /// returned handle may be detached by the caller.
    pub fn spawn(self) -> io::Result<JoinHandle<()>> {
        thread::Builder::new()
            .name("nydus_prefetch".to_string())
            .spawn(move || self.run())
    }

    /// Prefetch one priority blob through `prefetch`, sequentially with the
    /// full worker count, queueing it for a delayed retry when the backend
    /// throttles it.
    fn prefetch_priority_blob(
        &self,
        blob_index: u16,
        throttled: &mut Vec<u16>,
        prefetch: impl FnOnce(&BlobCaches) -> io::Result<()>,
    ) {
        match prefetch(&self.caches) {
            Ok(()) => info!("prefetched priority blob {}", blob_index),
            Err(err) if is_backend_throttled(&err) => {
                self.caches.metrics().inc_prefetch_reschedule();
                warn!(
                    "backend throttled prefetch of priority blob {}, rescheduling: {}",
                    blob_index, err
                );
                throttled.push(blob_index);
            }
            Err(err) => warn!("failed to prefetch priority blob {}: {}", blob_index, err),
        }
    }

    /// Drive the whole prefetch workflow synchronously on the calling thread:
    /// ondemand priority blobs first, then (only when the scope resolves to
    /// [`PrefetchScope::All`]) the image's reachable data in the other
    /// priority blobs in declared order and in the remaining blobs through a
    /// worker pool, then delayed retries of throttled blobs. Per-blob
    /// failures other than backend throttling are logged and skipped.
    pub fn run(mut self) {
        if self.scope == PrefetchScope::None {
            return;
        }

        // Blobs the backend throttled, awaiting a delayed retry.
        let mut throttled: Vec<u16> = Vec::new();

        // Under the "ondemand" scope the non-ondemand blobs are not pulled, but
        // every cache is opened in the background (blob meta fetched and
        // validated, sparse files created) so nothing pays those round trips
        // serially later: neither the ondemand checks below nor block-device
        // frontends, which probe many blobs right after the device appears.
        // "auto" needs the same checks to pick its scope, so it starts the
        // openers too. The openers run alongside the priority prefetch and
        // are joined once it is done.
        let openers = if self.scope != PrefetchScope::All {
            self.spawn_blob_openers()
        } else {
            Vec::new()
        };

        // Which priority blobs are ondemand (REDIRECT) blobs: needed to resolve
        // `auto`, to pick the blobs `ondemand` pulls and to order phase 1 under
        // `all`. A blob whose meta cannot be read is treated as a plain blob
        // and logged.
        let ondemand: Vec<bool> = self
            .priority
            .iter()
            .map(|&blob_index| match self.caches.is_redirect(blob_index) {
                Ok(redirect) => redirect,
                Err(err) => {
                    warn!("failed to inspect priority blob {}: {}", blob_index, err);
                    false
                }
            })
            .collect();
        let scope = match self.scope {
            PrefetchScope::Auto => {
                let resolved = if ondemand.contains(&true) {
                    PrefetchScope::Ondemand
                } else {
                    PrefetchScope::All
                };
                info!("prefetch scope auto resolved to {:?}", resolved);
                resolved
            }
            scope => scope,
        };
        if self.stopped() {
            return;
        }

        // Phase 1a: ondemand priority blobs, sequential. They go first (they
        // stream the recorded working set into the source blobs' caches, so
        // the image prefetch that follows skips the groups already filled),
        // and under "ondemand" they are the only blobs pulled.
        let (redirect, plain): (Vec<_>, Vec<_>) = self
            .priority
            .iter()
            .copied()
            .zip(ondemand.iter().copied())
            .partition(|&(_, redirect)| redirect);
        for (blob_index, _) in redirect {
            if self.stopped() {
                return;
            }
            self.prefetch_priority_blob(blob_index, &mut throttled, |caches| {
                caches.prefetch_blob(blob_index, self.threads, self.timeout)
            });
        }

        // Only "all" walks the image metadata; under "ondemand" no plain blob
        // data is reachable for prefetch, so retries of throttled ondemand
        // blobs (which ignore the extents) are all that remains.
        let extents = if scope == PrefetchScope::All {
            self.resolve_extents()
        } else {
            drop(self.extents.take());
            ImageExtents::Reachable(HashMap::new())
        };
        let extents = Arc::new(extents);

        // Phase 1b: under "all", the image data in the other priority blobs,
        // sequential and in declared order.
        if scope == PrefetchScope::All {
            for (blob_index, _) in plain {
                if self.stopped() {
                    return;
                }
                self.prefetch_priority_blob(blob_index, &mut throttled, |caches| {
                    caches.prefetch_blob_groups(
                        blob_index,
                        extents.selection(blob_index),
                        self.threads,
                        self.timeout,
                    )
                });
            }
        }

        for handle in openers {
            let _ = handle.join();
        }

        // Phase 2: the image data in the remaining blobs, concurrent worker
        // pool. Skipped unless the scope resolved to "all".
        if scope == PrefetchScope::All && !self.rest.is_empty() {
            let worker_count = self.threads.min(self.rest.len());
            let queue = Arc::new(Mutex::new(std::mem::take(&mut self.rest)));
            let throttled_shared = Arc::new(Mutex::new(Vec::new()));
            let timeout = self.timeout;
            let mut handles = Vec::with_capacity(worker_count);
            for _ in 0..worker_count {
                let blobs = self.caches.clone();
                let queue = queue.clone();
                let throttled_shared = throttled_shared.clone();
                let stop = self.stop.clone();
                let extents = extents.clone();
                let handle = thread::Builder::new()
                    .name("nydus_prefetch_worker".to_string())
                    .spawn(move || loop {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let blob_index = {
                            let mut guard = queue.lock().unwrap();
                            guard.pop()
                        };
                        match blob_index {
                            Some(blob_index) => {
                                match blobs.prefetch_blob_groups(
                                    blob_index,
                                    extents.selection(blob_index),
                                    1,
                                    timeout,
                                ) {
                                    Ok(()) => info!("prefetched blob {}", blob_index),
                                    Err(err) if is_backend_throttled(&err) => {
                                        blobs.metrics().inc_prefetch_reschedule();
                                        warn!(
                                            "backend throttled prefetch of blob {}, rescheduling: {}",
                                            blob_index, err
                                        );
                                        throttled_shared.lock().unwrap().push(blob_index);
                                    }
                                    Err(err) => {
                                        warn!("failed to prefetch blob {}: {}", blob_index, err)
                                    }
                                }
                            }
                            None => break,
                        }
                    });
                match handle {
                    Ok(handle) => handles.push(handle),
                    Err(err) => warn!("failed to spawn prefetch worker: {}", err),
                }
            }
            for handle in handles {
                let _ = handle.join();
            }
            throttled.append(&mut throttled_shared.lock().unwrap());
        }

        // Phase 3: delayed retries of throttled blobs. Each blob gets a fresh
        // random deadline inside the retry window; retries that get throttled
        // again are rescheduled, other failures are dropped. The cross-process
        // prefetch flock and group-map skip logic inside the blob prefetch
        // make a rescheduled prefetch behind another node's progress nearly
        // free.
        let mut queue: Vec<(Instant, u16)> = throttled
            .into_iter()
            .map(|blob_index| (Instant::now() + self.retry_delay(), blob_index))
            .collect();
        while !queue.is_empty() {
            queue.sort_by_key(|(deadline, _)| *deadline);
            let (deadline, blob_index) = queue.remove(0);
            while Instant::now() < deadline {
                if self.stopped() {
                    return;
                }
                let remaining = deadline.saturating_duration_since(Instant::now());
                thread::sleep(remaining.min(RESCHEDULE_POLL_INTERVAL));
            }
            if self.stopped() {
                return;
            }
            self.caches.metrics().inc_prefetch_reschedule_run();
            match self.caches.prefetch_blob_groups(
                blob_index,
                extents.selection(blob_index),
                self.threads,
                self.timeout,
            ) {
                Ok(()) => info!("prefetched rescheduled blob {}", blob_index),
                Err(err) if is_backend_throttled(&err) => {
                    self.caches.metrics().inc_prefetch_reschedule();
                    warn!(
                        "backend throttled rescheduled prefetch of blob {}, rescheduling again: {}",
                        blob_index, err
                    );
                    queue.push((Instant::now() + self.retry_delay(), blob_index));
                }
                Err(err) => warn!(
                    "failed to prefetch rescheduled blob {}, giving up: {}",
                    blob_index, err
                ),
            }
        }
    }

    /// A random delay inside the configured retry window, seeded from OS
    /// entropy via `RandomState` so no `rand` dependency is needed.
    fn retry_delay(&self) -> Duration {
        let span = self.retry_delay_max.saturating_sub(self.retry_delay_min);
        if span.is_zero() {
            return self.retry_delay_min;
        }
        use std::hash::{BuildHasher, Hasher};
        let seed = std::collections::hash_map::RandomState::new()
            .build_hasher()
            .finish();
        // The default window (6h span) is far below `u64::MAX` nanoseconds
        // (~584 years); clamp anyway so pathological configs cannot overflow.
        let span_nanos = u64::try_from(span.as_nanos()).unwrap_or(u64::MAX);
        self.retry_delay_min + Duration::from_nanos(seed % span_nanos.saturating_add(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::atomic::AtomicUsize;

    use nydus_backend::{throttled_error, BlobBackend, Local, ReadContext};
    use nydus_format::blob::BlobMetadata;
    use nydus_format::utils::SHA256_DIGEST_SIZE;
    use nydus_test_utils::blob::FullBlob;
    use nydus_test_utils::metadata::{single_chunk_group, single_chunk_group_redirect};
    use tempfile::tempdir;

    /// Wraps a local backend and fails the first `failures` data reads with
    /// the given error builder, then delegates, counting every read attempt.
    struct FlakyBackend {
        inner: Local,
        failures: usize,
        error: fn() -> io::Error,
        attempts: AtomicUsize,
    }

    impl FlakyBackend {
        fn new(dir: &Path, failures: usize, error: fn() -> io::Error) -> Arc<Self> {
            Arc::new(Self {
                inner: Local::new(dir.to_path_buf()),
                failures,
                error,
                attempts: AtomicUsize::new(0),
            })
        }

        fn attempts(&self) -> usize {
            self.attempts.load(Ordering::SeqCst)
        }
    }

    impl BlobBackend for FlakyBackend {
        fn blob_metadata(&self, blob_id: &[u8; SHA256_DIGEST_SIZE]) -> io::Result<BlobMetadata> {
            self.inner.blob_metadata(blob_id)
        }

        fn read_range_into(
            &self,
            blob_id: &[u8; SHA256_DIGEST_SIZE],
            offset: u64,
            dst: &mut [u8],
            ctx: ReadContext,
        ) -> io::Result<()> {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            if attempt < self.failures {
                return Err((self.error)());
            }
            self.inner.read_range_into(blob_id, offset, dst, ctx)
        }
    }

    /// A single-blob prefetcher over a `FlakyBackend`, with a tight retry
    /// window so tests run in milliseconds.
    fn prefetcher_over(
        backend: Arc<FlakyBackend>,
        backend_dir: &Path,
        cache_dir: &Path,
        payload: &[u8],
        meta: &BlobMetadata,
    ) -> BlobPrefetcher {
        let full_blob_id = FullBlob::new(payload)
            .with_blob_metadata(meta)
            .with_blob_metadata_file()
            .save_in(backend_dir);
        let caches = Arc::new(
            BlobCaches::new([(0u16, full_blob_id)], backend, Some(cache_dir), None).unwrap(),
        );
        BlobPrefetcher::new(
            caches,
            whole_blobs(vec![0], Vec::new()),
            1,
            PrefetchScope::All,
            Duration::ZERO,
            Duration::from_millis(30),
            Duration::from_millis(60),
        )
    }

    fn plan(priority: Vec<u16>, rest: Vec<u16>, extents: ImageExtents) -> PrefetchPlan {
        PrefetchPlan {
            priority,
            rest,
            extents: Box::new(move || Ok(extents)),
        }
    }

    fn whole_blobs(priority: Vec<u16>, rest: Vec<u16>) -> PrefetchPlan {
        plan(priority, rest, ImageExtents::WholeBlobs)
    }

    fn test_payload() -> (Vec<u8>, BlobMetadata) {
        let payload = vec![0xabu8; 4096];
        let meta = single_chunk_group(&payload);
        (payload, meta)
    }

    #[test]
    fn throttled_prefetch_is_rescheduled_and_retried() {
        let backend_dir = tempdir().unwrap();
        let cache_dir = tempdir().unwrap();
        let (payload, meta) = test_payload();
        // The first data read is throttled; the delayed retry succeeds.
        let backend = FlakyBackend::new(backend_dir.path(), 1, || {
            throttled_error("proxy answered 429")
        });
        let prefetcher = prefetcher_over(
            backend.clone(),
            backend_dir.path(),
            cache_dir.path(),
            &payload,
            &meta,
        );

        let metrics = prefetcher.caches.metrics().clone();
        let reschedules_before = metrics.prefetch_reschedule_total();
        let runs_before = metrics.prefetch_reschedule_run_total();
        let start = Instant::now();
        prefetcher.run();

        // One throttled attempt plus the successful delayed retry.
        assert!(backend.attempts() >= 2, "attempts={}", backend.attempts());
        // The retry waited out (at least) the minimum delay.
        assert!(start.elapsed() >= Duration::from_millis(30));
        assert!(metrics.prefetch_reschedule_total() > reschedules_before);
        assert!(metrics.prefetch_reschedule_run_total() > runs_before);
    }

    #[test]
    fn non_throttled_failure_is_not_rescheduled() {
        let backend_dir = tempdir().unwrap();
        let cache_dir = tempdir().unwrap();
        let (payload, meta) = test_payload();
        // Always fail with an ordinary error: log-and-skip, no reschedule.
        let backend = FlakyBackend::new(backend_dir.path(), usize::MAX, || {
            io::Error::other("ordinary failure")
        });
        let prefetcher = prefetcher_over(
            backend.clone(),
            backend_dir.path(),
            cache_dir.path(),
            &payload,
            &meta,
        );

        let start = Instant::now();
        prefetcher.run();

        // No delayed retry: exactly the initial attempt, and no retry-window
        // sleep.
        assert_eq!(backend.attempts(), 1);
        assert!(start.elapsed() < Duration::from_millis(30));
    }

    #[test]
    fn raised_stop_flag_prevents_rescheduled_retries() {
        let backend_dir = tempdir().unwrap();
        let cache_dir = tempdir().unwrap();
        let (payload, meta) = test_payload();
        let backend = FlakyBackend::new(backend_dir.path(), usize::MAX, || {
            throttled_error("proxy answered 429")
        });
        let prefetcher = prefetcher_over(
            backend.clone(),
            backend_dir.path(),
            cache_dir.path(),
            &payload,
            &meta,
        );

        // Raise the stop flag before running: the initial pass is skipped
        // entirely and run() returns without sleeping towards a retry.
        prefetcher.stop_flag().store(true, Ordering::SeqCst);
        let start = Instant::now();
        prefetcher.run();

        assert_eq!(backend.attempts(), 0);
        assert!(start.elapsed() < Duration::from_millis(30));
    }

    /// Records which blob every data read targets, in call order.
    struct RecordingBackend {
        inner: Local,
        reads: Mutex<Vec<[u8; SHA256_DIGEST_SIZE]>>,
    }

    impl BlobBackend for RecordingBackend {
        fn blob_metadata(&self, blob_id: &[u8; SHA256_DIGEST_SIZE]) -> io::Result<BlobMetadata> {
            self.inner.blob_metadata(blob_id)
        }

        fn read_range_into(
            &self,
            blob_id: &[u8; SHA256_DIGEST_SIZE],
            offset: u64,
            dst: &mut [u8],
            ctx: ReadContext,
        ) -> io::Result<()> {
            self.reads.lock().unwrap().push(*blob_id);
            self.inner.read_range_into(blob_id, offset, dst, ctx)
        }
    }

    /// Two blobs in the backend dir: plain blob 1 and blob 2, an "ondemand"
    /// (REDIRECT) blob whose single group redirects to blob 1's group 0.
    /// Returns the backend, the cache set and the blob ids of (1, 2).
    fn plain_and_redirect_blobs(
        backend_dir: &Path,
        cache_dir: &Path,
    ) -> (
        Arc<RecordingBackend>,
        Arc<BlobCaches>,
        [[u8; SHA256_DIGEST_SIZE]; 2],
    ) {
        let (payload, plain_meta) = test_payload();
        let plain_id = FullBlob::new(&payload)
            .with_blob_metadata(&plain_meta)
            .with_blob_metadata_file()
            .save_in(backend_dir);
        let redirect_meta = single_chunk_group_redirect(&payload, 1);
        assert!(redirect_meta.is_redirect());
        let redirect_id = FullBlob::new(&payload)
            .with_blob_metadata(&redirect_meta)
            .with_blob_metadata_file()
            .save_in(backend_dir);
        let backend = Arc::new(RecordingBackend {
            inner: Local::new(backend_dir.to_path_buf()),
            reads: Mutex::new(Vec::new()),
        });
        let caches = Arc::new(
            BlobCaches::new(
                [(1u16, plain_id), (2u16, redirect_id)],
                backend.clone(),
                Some(cache_dir),
                None,
            )
            .unwrap(),
        );
        (backend, caches, [plain_id, redirect_id])
    }

    /// Run a prefetcher with `scope` over `plan` and return the distinct blobs
    /// read from the backend, in first-read order, as blob indexes.
    fn blobs_read(scope: PrefetchScope, plan: PrefetchPlan) -> Vec<u16> {
        let backend_dir = tempdir().unwrap();
        let cache_dir = tempdir().unwrap();
        let (backend, caches, ids) = plain_and_redirect_blobs(backend_dir.path(), cache_dir.path());
        BlobPrefetcher::new(
            caches,
            plan,
            2,
            scope,
            Duration::ZERO,
            Duration::from_millis(1),
            Duration::from_millis(1),
        )
        .run();
        let mut order = Vec::new();
        for read in backend.reads.lock().unwrap().iter() {
            let index = if *read == ids[0] { 1 } else { 2 };
            if !order.contains(&index) {
                order.push(index);
            }
        }
        order
    }

    #[test]
    fn scope_none_pulls_nothing_even_with_an_ondemand_blob() {
        let plan = whole_blobs(vec![2, 1], Vec::new());
        assert!(blobs_read(PrefetchScope::None, plan).is_empty());
    }

    #[test]
    fn scope_ondemand_streams_only_the_ondemand_blob() {
        let plan = whole_blobs(vec![2, 1], Vec::new());
        // The redirect stream fills blob 1's cache from blob 2's data; blob 1
        // itself is never read from the backend.
        assert_eq!(blobs_read(PrefetchScope::Ondemand, plan), vec![2]);
        let plan = whole_blobs(vec![1], vec![2]);
        // No ondemand blob among the priority blobs: nothing is pulled.
        assert!(blobs_read(PrefetchScope::Ondemand, plan).is_empty());
    }

    #[test]
    fn scope_all_streams_the_ondemand_blob_before_the_others() {
        // Declared order puts the plain blob first; the ondemand blob is
        // still streamed first, then the plain blob's remaining groups.
        let plan = whole_blobs(vec![1, 2], Vec::new());
        let read = blobs_read(PrefetchScope::All, plan);
        assert_eq!(read.first(), Some(&2), "reads: {read:?}");
    }

    #[test]
    fn scope_auto_picks_ondemand_with_an_ondemand_blob_and_all_without() {
        let plan = whole_blobs(vec![2, 1], Vec::new());
        assert_eq!(blobs_read(PrefetchScope::Auto, plan), vec![2]);
        let plan = whole_blobs(vec![1], Vec::new());
        assert_eq!(blobs_read(PrefetchScope::Auto, plan), vec![1]);
    }

    #[test]
    fn scope_all_warms_only_blobs_with_reachable_data() {
        // Blob 1 holds no data the image reaches: nothing is pulled.
        let unreachable = plan(vec![1], Vec::new(), ImageExtents::Reachable(HashMap::new()));
        assert!(blobs_read(PrefetchScope::All, unreachable).is_empty());
        let reachable = plan(
            Vec::new(),
            vec![1],
            ImageExtents::Reachable(HashMap::from([(1, std::iter::once(0..100).collect())])),
        );
        assert_eq!(blobs_read(PrefetchScope::All, reachable), vec![1]);
    }

    #[test]
    fn failed_extents_walk_falls_back_to_whole_blobs() {
        let plan = PrefetchPlan {
            priority: vec![1],
            rest: Vec::new(),
            extents: Box::new(|| Err(io::Error::other("corrupt inode"))),
        };
        assert_eq!(blobs_read(PrefetchScope::All, plan), vec![1]);
    }

    #[test]
    fn only_scope_all_walks_the_image() {
        for scope in [
            PrefetchScope::None,
            PrefetchScope::Ondemand,
            PrefetchScope::Auto,
        ] {
            let walked = Arc::new(AtomicBool::new(false));
            let flag = walked.clone();
            let plan = PrefetchPlan {
                priority: vec![2, 1],
                rest: Vec::new(),
                extents: Box::new(move || {
                    flag.store(true, Ordering::SeqCst);
                    Ok(ImageExtents::WholeBlobs)
                }),
            };
            blobs_read(scope, plan);
            assert!(!walked.load(Ordering::SeqCst), "{scope:?} walked the image");
        }
    }
}

//! Byte ranges of a blob's decoded address space, the unit image prefetch
//! selects data in. (by Claude)

use std::ops::Range;

/// Sorted, disjoint, non-empty, non-adjacent byte ranges of one blob's
/// decoded (uncompressed, padded) address space: the offsets EROFS chunk
/// indexes and blob cache reads use. Built with [`BlobRangesBuilder`], so
/// every value upholds the ordering invariant.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BlobRanges(Vec<Range<u64>>);

impl BlobRanges {
    /// The ranges in ascending order.
    pub fn as_slice(&self) -> &[Range<u64>] {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Total bytes covered.
    pub fn len_bytes(&self) -> u64 {
        self.0.iter().map(|range| range.end - range.start).sum()
    }
}

/// Collects ranges in any order and normalizes them into [`BlobRanges`].
/// A range touching the last pushed one is merged on the spot, so a file
/// whose chunks sit back to back costs one entry rather than one per chunk.
#[derive(Debug, Default)]
pub struct BlobRangesBuilder(Vec<Range<u64>>);

impl BlobRangesBuilder {
    /// Add `range`; empty ranges are ignored. Panics on a reversed range,
    /// which can only come from a caller swapping its endpoints.
    pub fn push(&mut self, range: Range<u64>) {
        assert!(range.start <= range.end, "reversed blob range {range:?}");
        if range.start == range.end {
            return;
        }
        match self.0.last_mut() {
            Some(last) if range.start <= last.end && range.end >= last.start => {
                last.start = last.start.min(range.start);
                last.end = last.end.max(range.end);
            }
            _ => self.0.push(range),
        }
    }

    pub fn build(mut self) -> BlobRanges {
        self.0.sort_unstable_by_key(|range| range.start);
        let mut merged: Vec<Range<u64>> = Vec::with_capacity(self.0.len());
        for range in self.0 {
            match merged.last_mut() {
                Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
                _ => merged.push(range),
            }
        }
        debug_assert!(merged
            .windows(2)
            .all(|pair| pair[0].start < pair[0].end && pair[0].end < pair[1].start));
        BlobRanges(merged)
    }
}

impl FromIterator<Range<u64>> for BlobRanges {
    fn from_iter<I: IntoIterator<Item = Range<u64>>>(iter: I) -> Self {
        let mut builder = BlobRangesBuilder::default();
        for range in iter {
            builder.push(range);
        }
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_are_sorted_merged_and_non_empty() {
        let ranges: BlobRanges = [10..20, 0..4, 4..6, 30..30, 15..25, 40..50, 8..9]
            .into_iter()
            .collect();
        assert_eq!(ranges.as_slice(), &[0..6, 8..9, 10..25, 40..50]);
        assert_eq!(ranges.len_bytes(), 6 + 1 + 15 + 10);
        assert!(BlobRanges::from_iter(std::iter::once(5..5)).is_empty());
    }

    #[test]
    #[should_panic(expected = "reversed blob range")]
    fn reversed_range_panics() {
        #[allow(clippy::reversed_empty_ranges)]
        BlobRangesBuilder::default().push(9..5);
    }
}

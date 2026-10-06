//! Shared precomputation for the baby-step giant-step solvers: the hashed index of baby steps, the
//! affine walk helpers the giant walks are built from, the cached `BabyStepsTable`, and the process-wide
//! table cache.

use alloc::{sync::Arc, vec::Vec};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{batch_inversion, BigInteger, Field, PrimeField, Zero};
use core::hash::{BuildHasher, Hash};
use hashbrown::{hash_map::Entry, HashMap};

#[cfg(feature = "ahash")]
use ahash::RandomState;
#[cfg(feature = "std")]
use ark_serialize::CanonicalSerialize;
#[cfg(feature = "std")]
use core::any::{Any, TypeId};
#[cfg(feature = "std")]
use std::sync::{OnceLock, RwLock};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "std")]
pub const MAX_NUM_BABY_STEPS: u64 = 1 << 18;
#[cfg(not(feature = "std"))]
pub const MAX_NUM_BABY_STEPS: u64 = 1 << 16;

/// Cap on the baby steps table a batch may size itself to. About 60 MB at `1 << 21` entries.
#[cfg(feature = "std")]
pub const MAX_NUM_BABY_STEPS_BATCH: u64 = 1 << 21;
#[cfg(not(feature = "std"))]
pub const MAX_NUM_BABY_STEPS_BATCH: u64 = MAX_NUM_BABY_STEPS;

/// Multiples of a walk step kept in affine, and the largest walk block.
pub(super) const NORMALIZE_BLOCK: usize = 1 << 10;

/// Smallest block of an early-exit giant walk. Blocks double up to `NORMALIZE_BLOCK`.
pub(super) const BLOCK_MIN: usize = 64;

/// Giant steps per parallel chunk in the bsgs walks.
#[cfg(feature = "parallel")]
pub(super) const PAR_CHUNK_CENTERS: u64 = 1 << 9;

// Hasher for the precomputed tables and cache. `ahash` when enabled, else hashbrown's default.
#[cfg(feature = "ahash")]
pub(super) type BsgsHasher = RandomState;
#[cfg(not(feature = "ahash"))]
pub(super) type BsgsHasher = hashbrown::DefaultHashBuilder;

#[cfg(feature = "ahash")]
pub(super) fn bsgs_hasher() -> BsgsHasher {
    RandomState::new()
}
#[cfg(not(feature = "ahash"))]
pub(super) fn bsgs_hasher() -> BsgsHasher {
    BsgsHasher::default()
}

/// Affine coordinates of a non-identity point.
pub(super) type Xy<F> = (F, F);

/// Sign bit of a y-coordinate: parity of its first nonzero base-prime-field coordinate. `y` and `-y` differ
/// in it since the characteristic is odd. Undefined only for `y = 0` (2-torsion).
pub(super) fn y_sign<F: Field>(y: &F) -> bool {
    y.to_base_prime_field_elements()
        .find(|c| !c.is_zero())
        .map_or(false, |c| c.into_bigint().is_odd())
}

/// x-coordinate and y-sign of an affine point, `None` for the identity.
pub(super) fn xy_sign<G: CurveGroup>(p: &G::Affine) -> Option<(G::BaseField, bool)> {
    p.xy().map(|(x, y)| (x, y_sign(&y)))
}

/// Maps a field element to an index in the table, kept as 128-bit hash of the field element.
/// Keyed by a 64-bit hash of the key with a second 64-bit hash stored per entry. Keys whose
/// first hash collides at build time go to `spill` and are matched on both hashes.
pub struct FfToIndexMap {
    map: HashMap<u64, (u64, u32), BsgsHasher>,
    spill: Vec<(u64, u64, u32)>,
}

impl FfToIndexMap {
    /// `entries` are `(first hash, second hash, value)`.
    pub fn from_entries(hasher: BsgsHasher, entries: Vec<(u64, u64, u32)>) -> Self {
        let mut map = HashMap::with_capacity_and_hasher(entries.len(), hasher);
        let mut spill = Vec::new();
        for (k_1, k_2, v) in entries {
            match map.entry(k_1) {
                Entry::Vacant(e) => {
                    e.insert((k_2, v));
                }
                Entry::Occupied(_) => spill.push((k_1, k_2, v)),
            }
        }
        Self { map, spill }
    }

    pub fn get<T: Hash>(&self, key: &T) -> Option<u32> {
        let hasher = self.map.hasher();
        let (k_1, expected_k_2) = Self::hash(hasher, key);
        let &(k_2, v) = self.map.get(&k_1)?;
        if k_2 == expected_k_2 {
            return Some(v);
        }
        self.spill
            .iter()
            .find(|e| e.0 == k_1 && e.1 == expected_k_2)
            .map(|e| e.2)
    }

    /// This gives a 128-bit hash of `key`
    pub fn hash<T: Hash>(hasher: &BsgsHasher, key: &T) -> (u64, u64) {
        (
            BuildHasher::hash_one(hasher, key),
            // Use a DST to create new hash function since `hash_one` only gives 64-bit output
            BuildHasher::hash_one(hasher, &(1u8, key)),
        )
    }

    /// Entries held in the map and in the spill list.
    #[cfg(test)]
    pub fn lengths(&self) -> (usize, usize) {
        (self.map.len(), self.spill.len())
    }
}

// A precomputed table, kept behind an `Arc`. Keyed by the type of the table and the serialized
// `base` so the same table is reused across calls for the same `base`.
#[cfg(feature = "std")]
type CachedTable = Arc<dyn Any + Send + Sync>;
#[cfg(feature = "std")]
type BsgsCache = HashMap<(TypeId, Vec<u8>), CachedTable, BsgsHasher>;

#[cfg(feature = "std")]
static CACHE: OnceLock<RwLock<BsgsCache>> = OnceLock::new();

#[cfg(feature = "std")]
fn cache() -> &'static RwLock<BsgsCache> {
    CACHE.get_or_init(|| RwLock::new(HashMap::with_hasher(bsgs_hasher())))
}

#[cfg(feature = "std")]
fn cache_read<R>(f: impl FnOnce(&BsgsCache) -> R) -> R {
    f(&cache().read().unwrap())
}

#[cfg(feature = "std")]
fn cache_write<R>(f: impl FnOnce(&mut BsgsCache) -> R) -> R {
    f(&mut cache().write().unwrap())
}

/// Drop every table in the process-wide cache. A table still held by a caller is freed once released.
/// The next solve for a `base` rebuilds its table.
#[cfg(feature = "std")]
pub fn clear_cached_tables() {
    cache_write(|c| {
        c.clear();
        c.shrink_to_fit();
    });
}

// Nothing is cached.
#[cfg(not(feature = "std"))]
pub fn clear_cached_tables() {}

// Look up a cached table by `map_key` and return it if `is_enough`, else build one, given the entry it
// supersedes, and store it, re-checking under the write lock so a concurrent larger build is not
// clobbered.
#[cfg(feature = "std")]
pub(super) fn get_or_build_cached<T: Any + Send + Sync>(
    map_key: (TypeId, Vec<u8>),
    is_enough: impl Fn(&T) -> bool,
    build: impl FnOnce(Option<&T>) -> T,
) -> Arc<T> {
    let cached = cache_read(|c| c.get(&map_key).and_then(|t| t.clone().downcast::<T>().ok()));
    if let Some(table) = &cached {
        if is_enough(table) {
            return table.clone();
        }
    }
    let table = Arc::new(build(cached.as_deref()));
    cache_write(
        |c| match c.get(&map_key).and_then(|t| t.clone().downcast::<T>().ok()) {
            Some(existing) if is_enough(&existing) => existing,
            _ => {
                c.insert(map_key, table.clone());
                table
            }
        },
    )
}

// Table value packs the baby step `i` (low 31 bits) with the y-sign of `base * i` as MSB.
pub(super) const SIGN_BIT: u32 = 1 << 31;

/// Pack the sign bit as the MSB, the other 31 bits are for `index`
fn pack(index: u32, sign: bool) -> u32 {
    debug_assert!(index < SIGN_BIT, "baby step index must fit in 31 bits");
    index | ((sign as u32) << 31)
}

/// Unpack a `pack`ed value returning the index and sign bit
fn unpack(v: u32) -> (u32, bool) {
    (v & !SIGN_BIT, v & SIGN_BIT != 0)
}

/// Baby steps `base * i -> i` for `i` in `[1, num_baby_steps]`, keyed by the x-coordinate so `base * i`
/// and `base * -i` share an entry. The value packs `i` with the y-sign of `base * i`.
pub struct BabyStepsTable<G: CurveGroup> {
    /// Number of baby steps
    pub num_steps: u64,
    /// `base * (2 * num_baby_steps + 1)`, subtracted per giant step in the walk.
    pub giant_step: G::Affine,
    /// `(1..=B) * -giant_step` for the walk block `B`.
    pub neg_giant_multiples: Vec<Xy<G::BaseField>>,
    pub map: FfToIndexMap,
}

impl<G: CurveGroup + Send + Sync> BabyStepsTable<G> {
    pub fn new(base: G, num_steps: u64) -> Self {
        let hasher = bsgs_hasher();
        let entries = Self::baby_steps(base, num_steps, &hasher);
        let index = FfToIndexMap::from_entries(hasher, entries);
        let giant = base.mul_bigint([Self::giant_step_size_given_num_baby_steps(num_steps)]);
        // Walk blocks are capped at the table size so a small table stays cheap to build.
        let mult_count = NORMALIZE_BLOCK.min(num_steps.max(BLOCK_MIN as u64) as usize);
        Self {
            num_steps,
            giant_step: giant.into_affine(),
            neg_giant_multiples: affine_multiples(-giant, mult_count),
            map: index,
        }
    }

    pub fn get_unpacked(&self, x: &G::BaseField) -> Option<(u32, bool)> {
        self.map.get(x).map(unpack)
    }

    /// Giant step `2 * num_baby_steps + 1`: the width of one giant's window.
    fn giant_step_size_given_num_baby_steps(num_baby_steps: u64) -> u64 {
        2 * num_baby_steps + 1
    }

    pub(super) fn giant_step_size(&self) -> u64 {
        Self::giant_step_size_given_num_baby_steps(self.num_steps)
    }

    // Builds the table from the affine base, so serializing it for the cache key costs no inversion. A cached
    // table is reused only if it holds at least `num_steps` baby steps, else it is rebuilt larger and replaces
    // the smaller one, so the first caller's `table_size` does not cap later calls.
    #[cfg(feature = "std")]
    pub fn get_or_build(base: G::Affine, num_steps: u64) -> Option<Arc<Self>> {
        let mut key = Vec::with_capacity(base.compressed_size());
        base.serialize_compressed(&mut key).ok()?;
        let map_key = (TypeId::of::<G>(), key);
        Some(get_or_build_cached(
            map_key,
            |t: &Self| t.num_steps >= num_steps,
            |_| BabyStepsTable::new(base.into_group(), num_steps),
        ))
    }

    // The table is rebuilt in each call.
    #[cfg(not(feature = "std"))]
    pub fn get_or_build(base: G::Affine, num_steps: u64) -> Option<Arc<Self>> {
        Some(Arc::new(BabyStepsTable::new(base.into_group(), num_steps)))
    }

    // `(first hash, second hash, packed i and sign)` of `base * i` for `i` in `[1, num_steps]`.
    fn baby_steps(base: G, num_steps: u64, hasher: &BsgsHasher) -> Vec<(u64, u64, u32)> {
        const CHUNK_SIZE: u64 = 1 << 12;
        if num_steps == 0 {
            return Vec::new();
        }
        let multiples = affine_multiples(base, NORMALIZE_BLOCK.min(num_steps as usize));
        let num_chunks = num_steps.div_ceil(CHUNK_SIZE);
        let chunk_step = base.mul_bigint([CHUNK_SIZE]);
        // Compute anchors per chunk, each anchor processed in parallel.
        let mut anchors = Vec::with_capacity(num_chunks as usize);
        let mut anchor = G::zero();
        for _ in 0..num_chunks {
            anchors.push(anchor);
            anchor += chunk_step;
        }
        let anchors_xy: Vec<_> = G::normalize_batch(&anchors)
            .iter()
            .map(|p| p.xy())
            .collect();

        let points_for_chunk = |(k, &start): (usize, &G)| {
            let first = k as u64 * CHUNK_SIZE;
            let last = (first + CHUNK_SIZE).min(num_steps);
            let mut out = Vec::with_capacity((last - first) as usize);
            walk_blocks(
                start,
                anchors_xy[k],
                base,
                &multiples,
                first,
                last,
                NORMALIZE_BLOCK,
                false,
                |i, p| {
                    let (x, y) = p.expect("baby step is never the identity");
                    let (f1, f2) = FfToIndexMap::hash(hasher, x);
                    out.push((f1, f2, pack(i as u32, y_sign(y))));
                    None
                },
            );
            out
        };

        // A single chunk stays on the calling thread: rayon's job injection costs more than the chunk.
        #[cfg(feature = "parallel")]
        if anchors.len() > 1 {
            return anchors
                .par_iter()
                .enumerate()
                .flat_map(points_for_chunk)
                .collect();
        }
        anchors
            .iter()
            .enumerate()
            .flat_map(points_for_chunk)
            .collect()
    }
}

/// Walk `start + k * step` in affine coordinates. Each `advance` adds the anchor to the precomputed
/// multiples `(1..=n) * step` sharing one batch inversion. A zero denominator (the
/// anchor is +/- that multiple) falls back to projective arithmetic for that point.
/// TODO: Note that this on for Short Weierstrass curves
pub(super) struct AffineWalk<G: CurveGroup> {
    start: G,
    step: G,
    /// Steps taken from `start`.
    offset: u64,
    /// `start + offset * step`, `None` for the identity.
    anchor: Option<Xy<G::BaseField>>,
    dens: Vec<G::BaseField>,
    out: Vec<Option<Xy<G::BaseField>>>,
}

impl<G: CurveGroup> AffineWalk<G> {
    /// `anchor` is `start + offset * step`. `cap` is the largest `n` passed to `advance`.
    pub(super) fn new(
        start: G,
        offset: u64,
        anchor: Option<Xy<G::BaseField>>,
        step: G,
        cap: usize,
    ) -> Self {
        Self {
            start,
            step,
            offset,
            anchor,
            dens: Vec::with_capacity(cap),
            out: Vec::with_capacity(cap),
        }
    }

    /// Points at offsets `offset + 1 ..= offset + n` from `start`, `None` for the identity.
    /// `multiples` is `(1..=n) * step`.
    pub(super) fn advance(
        &mut self,
        multiples: &[Xy<G::BaseField>],
        n: usize,
    ) -> &[Option<Xy<G::BaseField>>] {
        debug_assert!(n >= 1 && n <= multiples.len());
        let mults = &multiples[..n];
        self.out.clear();
        match self.anchor {
            None => self.out.extend(mults.iter().map(|&m| Some(m))),
            Some((x_a, y_a)) => {
                self.dens.clear();
                self.dens.extend(mults.iter().map(|&(x_m, _)| x_m - x_a));
                batch_inversion(&mut self.dens);
                for (i, &(x_m, y_m)) in mults.iter().enumerate() {
                    let inv = self.dens[i];
                    let p = if inv.is_zero() {
                        // Handle exception case when x-coords same
                        let m = self.offset + i as u64 + 1;
                        (self.start + self.step.mul_bigint([m])).into_affine().xy()
                    } else {
                        let l = (y_m - y_a) * inv;
                        let x = l.square() - x_a - x_m;
                        let y = l * (x_a - x) - y_a;
                        Some((x, y))
                    };
                    self.out.push(p);
                }
            }
        }
        self.offset += n as u64;
        self.anchor = self.out[n - 1];
        &self.out
    }
}

/// `(1..=count) * step` in affine coordinates, built in doubling rounds so each round costs one
/// inversion. A round reads the first `n` multiples and appends the `n` that follow it.
pub(super) fn affine_multiples<G: CurveGroup>(step: G, count: usize) -> Vec<Xy<G::BaseField>> {
    let mut multiples = Vec::with_capacity(count);
    if count == 0 {
        return multiples;
    }
    multiples.push(step.into_affine().xy().expect("step is never the identity"));
    let mut walk = AffineWalk::new(G::zero(), 1, Some(multiples[0]), step, count.div_ceil(2));
    while multiples.len() < count {
        let offset = multiples.len();
        let n = offset.min(count - offset);
        let next = walk.advance(&multiples[..n], n);
        multiples.extend(
            next.iter()
                .map(|p| p.expect("multiple is never the identity")),
        );
    }
    multiples
}

/// Walk `start + i * step` for `i` in `first..=last`, calling `f(k, point)` on each point and
/// returning the first `Some` it yields. `start_xy` is `start` in affine, `multiples` is
/// `(1..=B) * step`. Blocks start at `block` points and double up to `B`, one batch inversion each.
/// With `check_start`, `f` is also called on `start` itself as `i = first`.
pub(super) fn walk_blocks<G: CurveGroup>(
    start: G,
    start_xy: Option<Xy<G::BaseField>>,
    step: G,
    multiples: &[Xy<G::BaseField>],
    first: u64,
    last: u64,
    block: usize,
    check_start: bool,
    mut f: impl FnMut(u64, Option<&Xy<G::BaseField>>) -> Option<u64>,
) -> Option<u64> {
    if check_start {
        if let Some(r) = f(first, start_xy.as_ref()) {
            return Some(r);
        }
    }
    debug_assert!(!multiples.is_empty());
    let cap = last.saturating_sub(first).min(multiples.len() as u64) as usize;
    let mut walk = AffineWalk::new(start, 0, start_xy, step, cap);
    let mut block = block.clamp(1, multiples.len());
    let mut i = first;
    while i < last {
        let n = (last - i).min(block as u64) as usize;
        for (j, p) in walk.advance(multiples, n).iter().enumerate() {
            if let Some(r) = f(i + 1 + j as u64, p.as_ref()) {
                return Some(r);
            }
        }
        i += n as u64;
        block = (block * 2).min(multiples.len());
    }
    None
}

//! Baby-step giant-step solvers for small discrete logs.
//! Based on the paper [Computing Elliptic Curve Discrete Logarithms with Improved Baby-step Giant-step Algorithm](https://eprint.iacr.org/2015/605)

use alloc::string::ToString;
use ark_ff::{AdditiveGroup, Zero};
use hashbrown::HashMap;
use alloc::{sync::Arc, vec::Vec};
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    AffineRepr, CurveGroup, PrimeGroup,
};
#[cfg(feature = "std")]
use ark_serialize::CanonicalSerialize;
#[cfg(feature = "std")]
use core::any::{Any, TypeId};
#[cfg(feature = "std")]
use std::sync::{OnceLock, RwLock};

#[cfg(feature = "ahash")]
use ahash::RandomState;
use ark_std::cfg_iter;
#[cfg(not(feature = "std"))]
use integer_sqrt::IntegerSquareRoot;

#[cfg(feature = "std")]
use integer_sqrt::IntegerSquareRoot;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "std")]
pub const MAX_NUM_BABY_STEPS: u64 = 1 << 18;
#[cfg(not(feature = "std"))]
pub const MAX_NUM_BABY_STEPS: u64 = 1 << 16;

/// Solve discrete log using brute force.
/// `max` is the maximum value of the discrete log and this returns `x` such that `1 <= x <= max` and `base * x = target`
/// if such `x` exists, else return None.
pub fn solve_discrete_log_brute_force<G: AdditiveGroup>(
    max: u64,
    base: G,
    target: G,
) -> Option<u64> {
    if target == base {
        return Some(1);
    }
    let mut cur = base;
    for j in 2..=max {
        cur += base;
        if cur == target {
            return Some(j);
        }
    }
    None
}

/// Solve discrete log using Baby Step Giant Step as described in section 2 of <https://eprint.iacr.org/2015/605>
/// `max` is the maximum value of the discrete log and this returns `x` such that `1 <= x <= max` and `base * x = target`
/// if such `x` exists, else return None.
/// `max` is of type u64 but only accurate till a 52 bit value since 12 bit precision is lost while taking square root.
pub fn solve_discrete_log_bsgs<G: AdditiveGroup>(max: u64, base: G, target: G) -> Option<u64> {
    // Will lose 12 bits of precision
    #[cfg(feature = "std")]
    let m = (max as f64).sqrt().ceil() as u64;
    #[cfg(not(feature = "std"))]
    let m = max.integer_sqrt();
    solve_discrete_log_bsgs_inner(m, m, base, target)
}

/// Solve discrete log using Baby Step Giant Step with worse worst-case performance but better average case performance as described in section 2 of <https://eprint.iacr.org/2015/605>.
/// `max` is the maximum value of the discrete log and this returns `x` such that `1 <= x <= max` and `base * x = target`
/// if such `x` exists, else return None.
/// `max` is of type u64 but only accurate till a 52 bit value since 12 bit precision is lost while taking square root.
pub fn solve_discrete_log_bsgs_alt<G: AdditiveGroup>(max: u64, base: G, target: G) -> Option<u64> {
    // Will lose 12 bits of precision
    #[cfg(feature = "std")]
    let m = (max as f64 / 2.0).sqrt().ceil() as u64;
    #[cfg(not(feature = "std"))]
    let m = (max / 2).integer_sqrt();

    let baby_steps = core::cmp::min(m, MAX_NUM_BABY_STEPS);
    let giant_steps = (max + baby_steps - 1) / baby_steps;
    solve_discrete_log_bsgs_inner(baby_steps, giant_steps, base, target)
}

fn solve_discrete_log_bsgs_inner<G: AdditiveGroup>(
    num_baby_steps: u64,
    num_giant_steps: u64,
    base: G,
    target: G,
) -> Option<u64> {
    if base == target {
        return Some(1);
    }
    if target.is_zero() {
        return Some(0);
    }
    // Create a map of `base * i -> i` for `i` in `[1, num_baby_steps]`
    #[cfg(feature = "ahash")]
    let mut baby_steps = HashMap::with_hasher(RandomState::new());
    #[cfg(not(feature = "ahash"))]
    let mut baby_steps = HashMap::new();
    baby_steps.insert(base.to_string(), 1);
    let mut cur = base;
    for i in 2..=num_baby_steps {
        cur = cur + base;
        if cur == target {
            return Some(i);
        }
        baby_steps.insert(cur.to_string(), i);
    }
    let base_m = cur;
    let mut cur = target;
    for i in 0..num_giant_steps {
        if let Some(b) = baby_steps.get(&cur.to_string()) {
            return Some(i * num_baby_steps + b);
        }
        cur = cur - base_m;
    }
    None
}

// Hasher for the precomputed tables and cache. `ahash` when enabled, else hashbrown's default.
#[cfg(feature = "ahash")]
type BsgsHasher = RandomState;
#[cfg(not(feature = "ahash"))]
type BsgsHasher = hashbrown::DefaultHashBuilder;

#[cfg(feature = "ahash")]
fn bsgs_hasher() -> BsgsHasher {
    RandomState::new()
}
#[cfg(not(feature = "ahash"))]
fn bsgs_hasher() -> BsgsHasher {
    BsgsHasher::default()
}

// A precomputed baby steps table, kept behind an `Arc`. Keyed by the type of the group and the
// serialized `base` so the same table is reused across calls for the same `base`.
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

/// Baby steps `base * i -> i` for `i` in `[1, num_baby_steps]`.
pub struct BabyStepsTable<G: CurveGroup> {
    /// Number of baby steps
    num_steps: u64,
    // Subtracted per giant step in the walk. `base * (2 * num_baby_steps + 1)`.
    giant_step: G::Affine,
    // Keyed by the x-coordinate field element directly with `base * i` and `base * -i` sharing a key.
    // Value is the 31 bit dlog and the MSB as the sign of the point (y-coordinate) dlog corresponds to.
    table: HashMap<G::BaseField, u32, BsgsHasher>,
}

impl<G: CurveGroup + Send + Sync> BabyStepsTable<G> {
    pub fn new(base: G, num_steps: u64) -> Self {
        let steps = Self::baby_steps(base, num_steps);
        let mut table = HashMap::with_capacity_and_hasher(steps.len(), bsgs_hasher());
        table.extend(steps);
        let giant = base.mul_bigint([Self::giant_step_size_given_num_baby_steps(num_steps)]);
        Self {
            num_steps,
            giant_step: giant.into_affine(),
            table,
        }
    }

    pub fn get_unpacked(&self, x: &G::BaseField) -> Option<(u32, bool)> {
        self.table.get(x).map(|&packed| unpack(packed))
    }

    /// Giant step `2 * num_baby_steps + 1`: the width of one giant's window.
    fn giant_step_size_given_num_baby_steps(num_baby_steps: u64) -> u64 {
        2 * num_baby_steps + 1
    }

    fn giant_step_size(&self) -> u64 {
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
        if let Some(table) = cache_read(|c| {
            c.get(&map_key)
                .and_then(|t| t.clone().downcast::<Self>().ok())
        }) {
            if table.num_steps >= num_steps {
                return Some(table);
            }
        }
        let table = Arc::new(BabyStepsTable::new(base.into_group(), num_steps));
        let stored = cache_write(|c| {
            match c.get(&map_key).and_then(|t| t.clone().downcast::<Self>().ok()) {
                Some(existing) if existing.num_steps >= num_steps => existing,
                _ => {
                    c.insert(map_key, table.clone());
                    table.clone()
                }
            }
        });
        Some(stored)
    }

    // The table is rebuilt in each call.
    #[cfg(not(feature = "std"))]
    pub fn get_or_build(base: G::Affine, num_steps: u64) -> Option<Arc<Self>> {
        Some(Arc::new(BabyStepsTable::new(base.into_group(), num_steps)))
    }

    // `base * i -> i` for `i` in `[1, num_steps]`
    fn baby_steps(base: G, num_steps: u64) -> Vec<(G::BaseField, u32)> {
        // Computed chunk-wise so each chunk costs one inversion but each chunk can be
        // processed in parallel
        const CHUNK_SIZE: u64 = 1 << 16;
        // num_chunks = ceil(num_steps/CHUNK_SIZE)
        let num_chunks = (num_steps + CHUNK_SIZE - 1) / CHUNK_SIZE;
        let mut chunks = Vec::with_capacity(num_chunks as usize);
        for i in 0..num_chunks {
            let start = 1 + i * CHUNK_SIZE;
            let chunk_len = CHUNK_SIZE.min(num_steps - (start - 1));
            chunks.push((start, base.mul_bigint([start]), chunk_len));
        }

        let base = base.into_affine();

        let points_for_chunk = |&(start, starting_point, len): &(u64, G, u64)| {
            let mut points = Vec::with_capacity(len as usize);
            let mut cur = starting_point;
            for _ in 0..len {
                points.push(cur);
                cur += base;
            }
            let points = G::normalize_batch(&points);
            points
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let x = p.x().expect("baby step is never the identity");
                    (x, pack((start + i as u64) as u32, y_sign::<G>(p)))
                })
                .collect::<Vec<_>>()
        };

        cfg_iter!(chunks).flat_map(points_for_chunk).collect()
    }
}

/// `width` is the maximum difference between dlog of `target` and `min`
pub fn solve_discrete_log_given_table<G: CurveGroup + Send + Sync + 'static>(
    table: &BabyStepsTable<G>,
    base: G,
    min: u64,
    width: u64,
    target: G,
) -> Option<u64> {
    // Shift by the lower bound so the search is over [0, width]
    let target = if min == 0 {
        target
    } else {
        target - base.mul_bigint([min])
    };
    if target.is_zero() {
        return Some(min);
    }
    solve_given_shifted_target(table, min, width, target.into_affine())
}

/// `width` is the maximum difference between dlog of `target` and `min`
fn solve_given_shifted_target<G: CurveGroup + Send + Sync + 'static>(
    table: &BabyStepsTable<G>,
    min: u64,
    width: u64,
    target: G::Affine,
) -> Option<u64> {
    let num_baby_steps = table.num_steps;
    let giant_step = table.giant_step_size();

    // Fast path for a dlog in `[0, m]`: an x hit means the shifted target is `base * i` or `base * -i`,
    // so its dlog is `i`
    let target_x = target.x().expect("shifted target is non-identity here");
    if let Some((i, stored_sign)) = table.get_unpacked(&target_x) {
        let i = i as u64;
        return if i <= width && y_sign::<G>(&target) == stored_sign {
            Some(min + i)
        } else {
            None
        };
    }

    let last_center = (width + num_baby_steps) / giant_step;
    let target = target.into_group();

    // Split the center range into chunks each spanning about 2^32 of the search space. Chunk 0 is scanned
    // first directly from `target`, so a dlog below 2^32 is found without building the `width`/2^32 chunk
    // anchors; only when it is absent are the later chunks anchored and their giant walks run in parallel.
    #[cfg(feature = "parallel")]
    {
        let centers_per_chunk = ((1u64 << 32) / giant_step).max(1);
        // If more than 1 chunk to process
        if last_center + 1 > centers_per_chunk {
            // For smaller values (which are frequent in the use-case), scan first chunk
            if let Some(v) = scan_centers(table, min, width, target, 0, centers_per_chunk - 1) {
                return Some(v);
            }
            let chunk_giant = table.giant_step.mul_bigint([centers_per_chunk]);
            let chunk_count = last_center / centers_per_chunk + 1;
            let mut chunks = Vec::with_capacity((chunk_count - 1) as usize);
            let mut start = target - chunk_giant;
            for k in 1..chunk_count {
                let first = k * centers_per_chunk;
                let last = ((k + 1) * centers_per_chunk - 1).min(last_center);
                chunks.push((start, first, last));
                start = start - chunk_giant;
            }
            return chunks
                .into_par_iter()
                .find_map_any(|(start, first, last)| {
                    scan_centers(table, min, width, start, first, last)
                });
        }
    }

    scan_centers(table, min, width, target, 0, last_center)
}

/// Solve discrete log using Baby Step Giant Step with a table of baby steps that is precomputed once per `base`,
/// cached, and reused across calls, and the negation map from section 3 of <https://eprint.iacr.org/2015/605>
/// which halves the giant steps. Returns `x` such that `min <= x <= max` and `base * x = target` if such `x`
/// exists, else `None`. The table holds `MAX_NUM_BABY_STEPS` baby steps (capped at `max - min`).
pub fn solve_discrete_log_bsgs_precomputed<G: CurveGroup + Send + Sync + 'static>(
    max: u64,
    min: u64,
    base: G,
    target: G,
) -> Option<u64> {
    solve_discrete_log_bsgs_precomputed_with_table_size(max, min, MAX_NUM_BABY_STEPS, base, target)
}

/// Same as `solve_discrete_log_bsgs_precomputed` but with `table_size` baby steps. The table for a `base` is
/// built at the largest `table_size` requested so far for that `base`, growing on demand and reused after.
pub fn solve_discrete_log_bsgs_precomputed_with_table_size<G: CurveGroup + Send + Sync + 'static>(
    max: u64,
    min: u64,
    table_size: u64,
    base: G,
    target: G,
) -> Option<u64> {
    if max < min {
        return None;
    }
    let width = max - min;
    // Clamp so the baby step index fits below the sign bit in the packed table value.
    let m = table_size.min(width).max(1).min(SIGN_BIT as u64 - 1);
    // Shift by the lower bound so the search is over `[0, width]`.
    let target = if min == 0 {
        target
    } else {
        target - base.mul_bigint([min])
    };
    if target.is_zero() {
        return Some(min);
    }
    let base_and_target = G::normalize_batch(&[base, target]);
    let table = BabyStepsTable::<G>::get_or_build(base_and_target[0], m)?;
    solve_given_shifted_target(&table, min, width, base_and_target[1])
}

/// Scan the giant steps centered at `first_center..=last_center` starting from `start = target' - first_center * giant`.
/// Centered at `c`, one giant step covers the `2m + 1` values `[c * giant_step - m, c * giant_step + m]`.
/// Consecutive centers cover `[0, width]` exactly since `giant_step = 2m + 1`, so a range of
/// centers is a partition of the search space.
fn scan_centers<G: CurveGroup + Send + Sync + 'static>(
    table: &BabyStepsTable<G>,
    min: u64,
    width: u64,
    start: G,
    first_center: u64,
    last_center: u64,
) -> Option<u64> {
    let giant_step = table.giant_step_size();
    const GIANT_BLOCK_SIZE: usize = 1 << 10;
    let mut cur = start;
    let mut center = first_center;
    // Reused across blocks to avoid per-block allocation churn, sized to the scan when it is short.
    let mut block = Vec::with_capacity(GIANT_BLOCK_SIZE.min((last_center - first_center + 1) as usize));
    while center <= last_center {
        let end = (center + GIANT_BLOCK_SIZE as u64 - 1).min(last_center);
        block.clear();
        let first = center;
        let mut c = center;
        loop {
            block.push(cur);
            if c == end {
                break;
            }
            cur -= table.giant_step;
            c += 1;
        }
        if end < last_center {
            cur -= table.giant_step;
        }
        let affine = G::normalize_batch(&block);
        for (k, point) in affine.iter().enumerate() {
            let base_center = (first + k as u64) * giant_step;
            if block[k].is_zero() {
                if base_center <= width {
                    return Some(min + base_center);
                }
                continue;
            }
            let x = point.x().expect("giant step is non-identity here");
            if let Some((i, stored_sign)) = table.get_unpacked(&x) {
                let i = i as u64;
                // `block[k] = base * (base_center + i)` or `base * (base_center - i)`; the y-sign of
                // `point` matches the stored sign of `base * i` in the former case.
                let found = if y_sign::<G>(point) == stored_sign {
                    Some(base_center + i)
                } else if base_center >= i {
                    Some(base_center - i)
                } else {
                    None
                };
                if let Some(x) = found {
                    if x <= width {
                        return Some(min + x);
                    }
                }
            }
        }
        center = end + 1;
    }
    None
}

// Deterministic sign of a point's y-coordinate: whether `y` sorts after `-y` in the base field.
// `base * i` and `base * -i` share an x-coordinate but carry `y` and `-y`, so this bit tells them
// apart without recomputing `base * i` at query time.
// Undefined only for `y = 0` (2-torsion), which a prime-order group never has.
fn y_sign<G: CurveGroup>(point: &G::Affine) -> bool {
    let y = point.y().expect("baby step is never the identity");
    y > -y
}

// Table value packs the baby step `i` (low 31 bits) with the y-sign of `base * i` (top bit).
const SIGN_BIT: u32 = 1 << 31;

/// Pack the sign bit as the MSB, the other 31 bits are for `index`
fn pack(index: u32, sign: bool) -> u32 {
    debug_assert!(index < SIGN_BIT, "baby step index must fit in 31 bits");
    index | ((sign as u32) << 31)
}

/// Unpack a `pack`ed value returning the index and sign bit
fn unpack(v: u32) -> (u32, bool) {
    (v & !SIGN_BIT, v & SIGN_BIT != 0)
}


/// Baby steps `base * i -> i` for `i` in `[1, num_baby_steps]` in the target group, keyed by the target-field
/// element directly. `giant` is `base * num_baby_steps`. The negation map is not used since there is no x-coordinate.
struct PairingBabyStepsTable<E: Pairing> {
    num_steps: u64,
    giant: PairingOutput<E>,
    table: HashMap<E::TargetField, u32, BsgsHasher>,
}

impl<E: Pairing> PairingBabyStepsTable<E> {
    fn new(base: PairingOutput<E>, num_baby_steps: u64) -> Self {
        let mut table = HashMap::with_capacity_and_hasher(num_baby_steps as usize, bsgs_hasher());
        let mut cur = base;
        for i in 1..=num_baby_steps {
            table.insert(cur.0, i as u32);
            if i < num_baby_steps {
                cur = cur + base;
            }
        }
        Self {
            num_steps: num_baby_steps,
            giant: base.mul_bigint([num_baby_steps]),
            table,
        }
    }

    pub fn get(&self, x: &E::TargetField) -> Option<&u32> {
        self.table.get(x)
    }

    #[cfg(feature = "std")]
    fn get_or_build(base: PairingOutput<E>, num_baby_steps: u64) -> Option<Arc<Self>> {
        let mut key = Vec::with_capacity(base.compressed_size());
        base.serialize_compressed(&mut key).ok()?;
        let map_key = (TypeId::of::<Self>(), key);
        if let Some(table) = cache_read(|c| {
            c.get(&map_key)
                .and_then(|t| t.clone().downcast::<Self>().ok())
        }) {
            if table.num_steps >= num_baby_steps {
                return Some(table);
            }
        }
        let table = Arc::new(PairingBabyStepsTable::new(base, num_baby_steps));
        let stored = cache_write(|c| {
            match c.get(&map_key).and_then(|t| t.clone().downcast::<Self>().ok()) {
                Some(existing) if existing.num_steps >= num_baby_steps => existing,
                _ => {
                    c.insert(map_key, table.clone());
                    table.clone()
                }
            }
        });
        Some(stored)
    }

    // The table is rebuilt in each call.
    #[cfg(not(feature = "std"))]
    fn get_or_build(base: PairingOutput<E>, num_baby_steps: u64) -> Option<Arc<Self>> {
        Some(Arc::new(PairingBabyStepsTable::new(base, num_baby_steps)))
    }
}

fn solve_pairing_given_table<E: Pairing>(
    table: &PairingBabyStepsTable<E>,
    base: PairingOutput<E>,
    min: u64,
    width: u64,
    target: PairingOutput<E>,
) -> Option<u64> {
    let m = table.num_steps;
    let target = if min == 0 {
        target
    } else {
        target - base.mul_bigint([min])
    };
    let last_giant = width / m;
    let mut cur = target;
    for i in 0..=last_giant {
        let base_center = i * m;
        if cur.is_zero() {
            if base_center <= width {
                return Some(min + base_center);
            }
        } else if let Some(&b) = table.get(&cur.0) {
            let x = base_center + b as u64;
            if x <= width {
                return Some(min + x);
            }
        }
        cur = cur - table.giant;
    }
    None
}

/// Solve discrete log in the target group using Baby Step Giant Step with a table of baby steps precomputed once
/// per `base`, cached, and reused across calls. Returns `x` such that `min <= x <= max` and `base * x = target`
/// if such `x` exists, else `None`. The table holds about `sqrt(max - min)` baby steps.
pub fn solve_discrete_log_bsgs_precomputed_pairing<E: Pairing>(
    max: u64,
    min: u64,
    base: PairingOutput<E>,
    target: PairingOutput<E>,
) -> Option<u64> {
    if max < min {
        return None;
    }
    let m = (max - min).integer_sqrt().max(1);
    solve_discrete_log_bsgs_precomputed_pairing_with_table_size(max, min, m, base, target)
}

/// Same as `solve_discrete_log_bsgs_precomputed_pairing` but with `table_size` baby steps. The table for a `base`
/// is built at the largest `table_size` requested so far for that `base`, growing on demand and reused after.
pub fn solve_discrete_log_bsgs_precomputed_pairing_with_table_size<E: Pairing>(
    max: u64,
    min: u64,
    table_size: u64,
    base: PairingOutput<E>,
    target: PairingOutput<E>,
) -> Option<u64> {
    if max < min {
        return None;
    }
    let width = max - min;
    // Clamp so the baby step index fits in the u32 table value.
    let m = table_size.min(width).max(1).min(u32::MAX as u64);
    let table = PairingBabyStepsTable::get_or_build(base, m)?;
    solve_pairing_given_table(&table, base, min, width, target)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::{
        ops::Mul,
        time::{Duration, Instant},
    };

    use ark_bls12_381::{Bls12_381, Fr, G1Projective, G2Projective};
    use ark_ec::pairing::{Pairing, PairingOutput};
    use ark_ff::AdditiveGroup;
    use ark_std::{
        rand::{prelude::StdRng, SeedableRng},
        UniformRand,
    };

    #[test]
    fn solving_discrete_log() {
        let mut rng = StdRng::seed_from_u64(0u64);

        fn check<G: AdditiveGroup + Mul<Fr, Output = G>>(
            rng: &mut StdRng,
            base: G,
            check_large_value: bool,
        ) {
            let checks_per_max = 10;
            let mut total_checks = 0;
            let mut duration_naive = Duration::default();
            let mut duration_bsgs = Duration::default();
            let mut duration_bsgs_alt = Duration::default();

            for max in [1, 2, 3, 4, 5, 6, 8, 15, 16, 31, 32, 255, 256, 65535] {
                for _ in 0..checks_per_max {
                    let dl = (u16::rand(rng) as u64 % max) + 1;
                    let target = base * Fr::from(dl);

                    // println!("For max={} and discrete log={}", max, dl);
                    let start = Instant::now();
                    let dl_naive = solve_discrete_log_brute_force(max, base, target);
                    let time = start.elapsed();
                    assert_eq!(dl, dl_naive.unwrap());
                    // println!("Time for naive approach: {:?}", time);
                    duration_naive += time;

                    let start = Instant::now();
                    let dl_bsgs = solve_discrete_log_bsgs(max, base, target);
                    let time = start.elapsed();
                    assert_eq!(dl, dl_bsgs.unwrap());
                    // println!("Time for BSGS approach: {:?}", time);
                    duration_bsgs += time;

                    let start = Instant::now();
                    let dl_bsgs_alt = solve_discrete_log_bsgs_alt(max, base, target);
                    let time = start.elapsed();
                    assert_eq!(dl, dl_bsgs_alt.unwrap());
                    // println!("Time for alt. BSGS approach: {:?}", time);
                    duration_bsgs_alt += time;

                    total_checks += 1;
                }
            }

            if check_large_value {
                for dl in [
                    u32::MAX as u64,                  // 32-bit value
                    u32::MAX as u64 * u8::MAX as u64, // 40-bit value
                ] {
                    let target = base * Fr::from(dl);
                    println!("For discrete log={}", dl);

                    let start = Instant::now();
                    let dl_bsgs = solve_discrete_log_bsgs(dl, base, target);
                    let time = start.elapsed();
                    assert_eq!(dl, dl_bsgs.unwrap());
                    println!("Time for BSGS approach: {:?}", time);

                    let start = Instant::now();
                    let dl_bsgs_alt = solve_discrete_log_bsgs_alt(dl, base, target);
                    let time = start.elapsed();
                    assert_eq!(dl, dl_bsgs_alt.unwrap());
                    println!("Time for alt. BSGS approach: {:?}", time);
                }
            }

            let target = base * Fr::from(10);
            assert!(solve_discrete_log_brute_force(8, base, target).is_none());
            assert!(solve_discrete_log_bsgs(8, base, target).is_none());

            println!("For total {} checks, brute force took {:?} and baby step giant step took {:?} and alt. baby step giant step took {:?}", total_checks, duration_naive, duration_bsgs, duration_bsgs_alt);
        }

        println!("\n\nTesting for group G1");
        let g1 = G1Projective::rand(&mut rng);
        check::<G1Projective>(&mut rng, g1, true);

        println!("\n\nTesting for group G2");
        let g2 = G2Projective::rand(&mut rng);
        check::<G2Projective>(&mut rng, g2, true);

        println!("\n\nTesting for group GT");
        let gt = <Bls12_381 as Pairing>::pairing(g1, g2);
        check::<PairingOutput<Bls12_381>>(&mut rng, gt, false);
    }

    #[test]
    fn solving_discrete_log_precomputed() {
        let mut rng = StdRng::seed_from_u64(1u64);

        fn check_curve<G: CurveGroup + Send + Sync + 'static + Mul<Fr, Output = G>>(
            rng: &mut StdRng,
            base: G,
        ) {
            for &(min, max) in &[
                (0u64, 255u64),
                (1, 255),
                (0, 65535),
                (1000, 5000),
                (5, 5),
                (0, 0),
            ] {
                let span = max - min + 1;
                for _ in 0..5 {
                    let dl = min + (u16::rand(rng) as u64 % span);
                    let target = base * Fr::from(dl);
                    assert_eq!(
                        Some(dl),
                        solve_discrete_log_bsgs_precomputed(max, min, base, target)
                    );
                    assert_eq!(
                        Some(dl),
                        solve_discrete_log_bsgs_precomputed_with_table_size(
                            max, min, 64, base, target
                        )
                    );
                }
                // A discrete log below `min` is not returned.
                if min > 0 {
                    let target = base * Fr::from(min - 1);
                    assert_eq!(
                        None,
                        solve_discrete_log_bsgs_precomputed(max, min, base, target)
                    );
                }
                // A discrete log above `max` is not returned.
                let target = base * Fr::from(max + 1);
                assert_eq!(
                    None,
                    solve_discrete_log_bsgs_precomputed(max, min, base, target)
                );
            }
        }

        fn check_gt(rng: &mut StdRng, base: PairingOutput<Bls12_381>) {
            for &(min, max) in &[(0u64, 255u64), (1, 255), (0, 65535), (100, 1000), (0, 0)] {
                let span = max - min + 1;
                for _ in 0..3 {
                    let dl = min + (u16::rand(rng) as u64 % span);
                    let target = base * Fr::from(dl);
                    assert_eq!(
                        Some(dl),
                        solve_discrete_log_bsgs_precomputed_pairing(max, min, base, target)
                    );
                }
                if min > 0 {
                    let target = base * Fr::from(min - 1);
                    assert_eq!(
                        None,
                        solve_discrete_log_bsgs_precomputed_pairing(max, min, base, target)
                    );
                }
                let target = base * Fr::from(max + 1);
                assert_eq!(
                    None,
                    solve_discrete_log_bsgs_precomputed_pairing(max, min, base, target)
                );
            }
        }

        let g1 = G1Projective::rand(&mut rng);
        check_curve::<G1Projective>(&mut rng, g1);
        let g2 = G2Projective::rand(&mut rng);
        check_curve::<G2Projective>(&mut rng, g2);
        let gt = <Bls12_381 as Pairing>::pairing(g1, g2);
        check_gt(&mut rng, gt);

        // Amortized comparison against `solve_discrete_log_bsgs_alt` for repeated decryption in the target group.
        let base = <Bls12_381 as Pairing>::pairing(G1Projective::rand(&mut rng), g2);
        let max = 65535u64;
        let iters = 200usize;
        let dls: Vec<u64> = (0..iters).map(|_| u16::rand(&mut rng) as u64).collect();
        let targets: Vec<_> = dls.iter().map(|&d| base * Fr::from(d)).collect();

        let start = Instant::now();
        for (i, t) in targets.iter().enumerate() {
            assert_eq!(
                Some(dls[i]),
                solve_discrete_log_bsgs_precomputed_pairing_with_table_size(max, 0, 256, base, *t)
            );
        }
        let precomputed = start.elapsed();

        let start = Instant::now();
        for (i, t) in targets.iter().enumerate() {
            assert_eq!(Some(dls[i]), solve_discrete_log_bsgs_alt(max, base, *t));
        }
        let existing = start.elapsed();

        println!(
            "GT, {} solves up to {}: precomputed(+cache) {:?} vs bsgs_alt {:?}",
            iters, max, precomputed, existing
        );

        let c_base = G1Projective::rand(&mut rng);
        let c_targets: Vec<_> = dls.iter().map(|&d| c_base * Fr::from(d)).collect();

        let start = Instant::now();
        for (i, t) in c_targets.iter().enumerate() {
            assert_eq!(
                Some(dls[i]),
                solve_discrete_log_bsgs_precomputed(max, 0, c_base, *t)
            );
        }
        let c_precomputed = start.elapsed();

        let start = Instant::now();
        for (i, t) in c_targets.iter().enumerate() {
            assert_eq!(Some(dls[i]), solve_discrete_log_bsgs_alt(max, c_base, *t));
        }
        let c_existing = start.elapsed();

        println!(
            "G1, {} solves up to {}: precomputed(+cache) {:?} vs bsgs_alt {:?}",
            iters, max, c_precomputed, c_existing
        );

        let shot_max = 65535u64;
        let m_bal = shot_max.integer_sqrt();

        let mut t_old = Duration::default();
        let mut t_new = Duration::default();
        for _ in 0..iters {
            let b = <Bls12_381 as Pairing>::pairing(G1Projective::rand(&mut rng), g2);
            let dl = u16::rand(&mut rng) as u64;
            let target = b * Fr::from(dl);
            let s = Instant::now();
            assert_eq!(Some(dl), solve_discrete_log_bsgs(shot_max, b, target));
            t_old += s.elapsed();
            let s = Instant::now();
            assert_eq!(
                Some(dl),
                solve_discrete_log_bsgs_precomputed_pairing_with_table_size(
                    shot_max, 0, m_bal, b, target
                )
            );
            t_new += s.elapsed();
        }
        println!(
            "GT single-shot (no reuse), {} solves up to {} (m={}): bsgs {:?} vs new {:?}",
            iters, shot_max, m_bal, t_old, t_new
        );

        let mut t_old = Duration::default();
        let mut t_new = Duration::default();
        for _ in 0..iters {
            let b = G1Projective::rand(&mut rng);
            let dl = u16::rand(&mut rng) as u64;
            let target = b * Fr::from(dl);
            let s = Instant::now();
            assert_eq!(Some(dl), solve_discrete_log_bsgs(shot_max, b, target));
            t_old += s.elapsed();
            let s = Instant::now();
            assert_eq!(
                Some(dl),
                solve_discrete_log_bsgs_precomputed_with_table_size(shot_max, 0, m_bal, b, target)
            );
            t_new += s.elapsed();
        }
        println!(
            "G1 single-shot (no reuse), {} solves up to {} (m={}): bsgs {:?} vs new {:?}",
            iters, shot_max, m_bal, t_old, t_new
        );
    }

    // Exercises the chunked path: `max` above `u32::MAX` and a table small enough that the center range spans
    // more than one 2^32-wide chunk, so the discrete logs below land in later chunks.
    #[test]
    fn solving_discrete_log_chunked() {
        let mut rng = StdRng::seed_from_u64(2u64);
        let base = G1Projective::rand(&mut rng);
        let max = 1u64 << 33;
        let table_size = 1u64 << 16;
        for dl in [
            0u64,
            12345,
            (1u64 << 32) - 1,
            1u64 << 32,
            (1u64 << 32) + 6789,
            5_000_000_123,
            max,
        ] {
            let target = base * Fr::from(dl);
            assert_eq!(
                Some(dl),
                solve_discrete_log_bsgs_precomputed_with_table_size(max, 0, table_size, base, target)
            );
        }
    }

    // Exercises the direct baby-step fast path, a giant walk inside chunk 0, and the chunk-0 ->
    // chunk-1 boundary, i.e. every branch of the fast-path / chunk-0-first restructure.
    #[test]
    fn solving_discrete_log_fast_path_and_at_boundary() {
        let mut rng = StdRng::seed_from_u64(3u64);
        let base = G1Projective::rand(&mut rng);
        let m: u64 = 1 << 16;
        let max = 1u64 << 33;
        for dl in [
            0u64,             // identity fast path
            1,                // direct baby step
            m - 1,
            m,                // largest baby step (direct hit)
            m + 1,            // first value past the table (giant walk in chunk 0)
            1u64 << 20,       // mid chunk 0
            (1u64 << 32) - 1, // end of chunk 0
            1u64 << 32,       // start of chunk 1 (parallel path)
            (1u64 << 32) + 1,
        ] {
            let target = base * Fr::from(dl);
            assert_eq!(
                Some(dl),
                solve_discrete_log_bsgs_precomputed_with_table_size(max, 0, m, base, target),
                "dl={dl}"
            );
        }
        // A non-zero `min` shifts the same fast path / boundary.
        let min = 1000u64;
        for dl in [min, min + m, min + (1u64 << 32)] {
            let target = base * Fr::from(dl);
            assert_eq!(
                Some(dl),
                solve_discrete_log_bsgs_precomputed_with_table_size(max, min, m, base, target),
                "min={min}, dl={dl}"
            );
        }
    }
}

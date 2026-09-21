//! Baby-step giant-step over a cached `BabyStepsTable`, using the negation map from section 3 of
//! <https://eprint.iacr.org/2015/605> so one giant lookup resolves a window of `2m + 1` values.

use alloc::{sync::Arc, vec, vec::Vec};
use ark_ec::{AffineRepr, CurveGroup};
use integer_sqrt::IntegerSquareRoot;

#[cfg(feature = "std")]
use ark_serialize::CanonicalSerialize;
#[cfg(feature = "std")]
use core::any::TypeId;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use super::setup::PAR_CHUNK_CENTERS;
#[cfg(feature = "std")]
use super::setup::get_or_build_cached;
use super::setup::{
    walk_blocks, y_sign, BabyStepsTable, BLOCK_MIN, MAX_NUM_BABY_STEPS, MAX_NUM_BABY_STEPS_BATCH,
    SIGN_BIT,
};

/// Batch sizes at which a `BaseTable` window table should be used.
const MUL_TABLE_MIN_TARGETS: usize = 8;

/// Multiplies `base` by a `u64`. `table[i][j]` is `base * ((j + 1) << (i * window))`.
pub struct BaseTable<G: CurveGroup> {
    base: G,
    table: Vec<Vec<G::Affine>>,
    window: u32,
}

impl<G: CurveGroup + Send + Sync> BaseTable<G> {
    /// Table for products below `2^bits`.
    pub fn new(base: G, bits: u32, window: u32) -> Self {
        let num_rows = bits.div_ceil(window) as usize;
        let per_row = (1usize << window) - 1;
        let mut points = Vec::with_capacity(num_rows * per_row);
        let mut step = base;
        for _ in 0..num_rows {
            let mut acc = G::zero();
            for _ in 0..per_row {
                acc += step;
                points.push(acc);
            }
            for _ in 0..window {
                step.double_in_place();
            }
        }
        let table = G::normalize_batch(&points)
            .chunks(per_row)
            .map(|r| r.to_vec())
            .collect();
        Self {
            base,
            table,
            window,
        }
    }

    /// Window balancing the table build against `n` products.
    pub fn window(n: usize) -> u32 {
        (usize::BITS - n.leading_zeros())
            .saturating_sub(1)
            .clamp(1, 12)
    }

    pub fn mul(&self, v: u64) -> G {
        if self.table.is_empty() {
            return self.base.mul_bigint([v]);
        }
        let mask = (1u64 << self.window) - 1;
        let mut acc = G::zero();
        for (i, row) in self.table.iter().enumerate() {
            let j = (v >> (i as u32 * self.window)) & mask;
            if j != 0 {
                acc += row[j as usize - 1];
            }
        }
        acc
    }

    pub fn empty(base: G) -> Self {
        Self {
            base,
            table: Vec::new(),
            window: 0,
        }
    }

    /// Product bits the table covers.
    fn bits(&self) -> u32 {
        self.table.len() as u32 * self.window
    }

    // Cached per `base` like `BabyStepsTable`, grown to the largest `bits` and `window` asked for so far.
    #[cfg(feature = "std")]
    pub fn get_or_build(base: G::Affine, bits: u32, window: u32) -> Option<Arc<Self>> {
        let mut key = Vec::with_capacity(base.compressed_size());
        base.serialize_compressed(&mut key).ok()?;
        let map_key = (TypeId::of::<Self>(), key);
        Some(get_or_build_cached(
            map_key,
            |t: &Self| t.bits() >= bits && t.window >= window,
            |cached| {
                let (bits, window) = cached.map_or((bits, window), |c| {
                    (bits.max(c.bits()), window.max(c.window))
                });
                Self::new(base.into_group(), bits, window)
            },
        ))
    }

    // The table is rebuilt in each call.
    #[cfg(not(feature = "std"))]
    pub fn get_or_build(base: G::Affine, bits: u32, window: u32) -> Option<Arc<Self>> {
        Some(Arc::new(Self::new(base.into_group(), bits, window)))
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
    solve_discrete_log_given_table_inner(table, base, min, width, target, false)
}

/// Same as `solve_discrete_log_given_table` but tuned for dlogs above 2^32: chunk anchors are computed
/// lazily inside the parallel workers instead of serially up front.
pub fn solve_discrete_log_given_table_large<G: CurveGroup + Send + Sync + 'static>(
    table: &BabyStepsTable<G>,
    base: G,
    min: u64,
    width: u64,
    target: G,
) -> Option<u64> {
    solve_discrete_log_given_table_inner(table, base, min, width, target, true)
}

fn solve_discrete_log_given_table_inner<G: CurveGroup + Send + Sync + 'static>(
    table: &BabyStepsTable<G>,
    base: G,
    min: u64,
    width: u64,
    target: G,
    large: bool,
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
    solve_given_shifted_target(
        table,
        &BaseTable::empty(base),
        min,
        width,
        target.into_affine(),
        large,
        false,
    )
}

/// `width` is the maximum difference between dlog of `target` and `min`. With `serial` the centers are
/// scanned on the calling thread, leaving parallelism to the caller. An identity `target` has dlog `min`.
#[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
fn solve_given_shifted_target<G: CurveGroup + Send + Sync + 'static>(
    baby_steps_table: &BabyStepsTable<G>,
    base_table: &BaseTable<G>,
    min: u64,
    width: u64,
    target: G::Affine,
    large: bool,
    serial: bool,
) -> Option<u64> {
    let num_baby_steps = baby_steps_table.num_steps;
    let giant_step = baby_steps_table.giant_step_size();
    let target_g = target.into_group();

    // Fast path for a dlog in `[0, m]`: an x hit means the shifted target is `base * i` or `base * -i`,
    // so its dlog is `i` or `-i`. This also covers center 0 of the walk, whose window is `[-m, m]`.
    let Some((target_x, target_y)) = target.xy() else {
        return Some(min);
    };
    if let Some((i, stored_sign)) = baby_steps_table.get_unpacked(&target_x) {
        let i = i as u64;
        if i <= width && y_sign(&target_y) == stored_sign && base_table.mul(i) == target_g {
            return Some(min + i);
        }
    }

    // `last_center = ceil((width - num_baby_steps) / giant_step) = (width - num_baby_steps + giant_step - 1) / giant_step`
    // and `giant_step = 2*num_baby_steps + 1`
    let last_center = (width + num_baby_steps) / giant_step;
    if last_center == 0 {
        return None;
    }

    // Split the centers into chunks of `PAR_CHUNK_CENTERS` and scan them in parallel. With `large`, each
    // chunk anchors itself with one scalar multiplication inside the worker, so no chunk is anchored until
    // scheduled; this suits dlogs above 2^32. Otherwise chunk 0 is scanned first directly from `target` and
    // the rest are anchored by successive additions, cheaper when the dlog is below 2^32.
    #[cfg(feature = "parallel")]
    {
        let cpc = PAR_CHUNK_CENTERS;
        if !serial && last_center >= cpc {
            let chunk_count = last_center / cpc + 1;
            let bounds = |k: u64| (k * cpc, ((k + 1) * cpc - 1).min(last_center));
            if large {
                return (0..chunk_count).into_par_iter().find_map_any(|k| {
                    let (first, last) = bounds(k);
                    if k == 0 {
                        return scan_centers(
                            baby_steps_table, base_table, target_g, min, width, target, 0, last, false,
                        );
                    }
                    let anchor = (target_g - baby_steps_table.giant_step.mul_bigint([first])).into_affine();
                    scan_centers(baby_steps_table, base_table, target_g, min, width, anchor, first, last, true)
                });
            }
            // For smaller values (which are frequent in the use-case), scan first chunk
            if let Some(v) = scan_centers(
                baby_steps_table,
                base_table,
                target_g,
                min,
                width,
                target,
                0,
                bounds(0).1,
                false,
            ) {
                return Some(v);
            }
            let chunk_giant = baby_steps_table.giant_step.mul_bigint([cpc]);
            let mut anchors = Vec::with_capacity((chunk_count - 1) as usize);
            let mut anchor = target_g - chunk_giant;
            for _ in 1..chunk_count {
                anchors.push(anchor);
                anchor -= chunk_giant;
            }
            return G::normalize_batch(&anchors)
                .into_par_iter()
                .enumerate()
                .find_map_any(|(j, anchor)| {
                    let (first, last) = bounds(j as u64 + 1);
                    scan_centers(baby_steps_table, base_table, target_g, min, width, anchor, first, last, true)
                });
        }
    }

    scan_centers(
        baby_steps_table,
        base_table,
        target_g,
        min,
        width,
        target,
        0,
        last_center,
        false,
    )
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

/// Same as `solve_discrete_log_bsgs_precomputed` but tuned for dlogs above 2^32.
pub fn solve_discrete_log_bsgs_precomputed_large<G: CurveGroup + Send + Sync + 'static>(
    max: u64,
    min: u64,
    base: G,
    target: G,
) -> Option<u64> {
    solve_discrete_log_bsgs_precomputed_with_table_size_large(
        max,
        min,
        MAX_NUM_BABY_STEPS,
        base,
        target,
    )
}

/// Same as `solve_discrete_log_bsgs_precomputed` but with `table_size` baby steps. The table for a `base` is
/// built at the largest `table_size` requested so far for that `base`, growing on demand and reused after.
pub fn solve_discrete_log_bsgs_precomputed_with_table_size<
    G: CurveGroup + Send + Sync + 'static,
>(
    max: u64,
    min: u64,
    table_size: u64,
    base: G,
    target: G,
) -> Option<u64> {
    solve_discrete_log_bsgs_precomputed_inner(max, min, table_size, base, target, false)
}

/// Same as `solve_discrete_log_bsgs_precomputed_with_table_size` but tuned for dlogs above 2^32.
pub fn solve_discrete_log_bsgs_precomputed_with_table_size_large<
    G: CurveGroup + Send + Sync + 'static,
>(
    max: u64,
    min: u64,
    table_size: u64,
    base: G,
    target: G,
) -> Option<u64> {
    solve_discrete_log_bsgs_precomputed_inner(max, min, table_size, base, target, true)
}

fn solve_discrete_log_bsgs_precomputed_inner<G: CurveGroup + Send + Sync + 'static>(
    max: u64,
    min: u64,
    table_size: u64,
    base: G,
    target: G,
    large: bool,
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
    solve_given_shifted_target(
        &table,
        &BaseTable::empty(base),
        min,
        width,
        base_and_target[1],
        large,
        false,
    )
}

/// Solve for many `targets` sharing `base`, `min` and `max`. The shift by `min`, the normalization and the
/// baby steps table are shared, and the table is sized from the number of targets so each giant walk is
/// shorter than a single-target solve would make it. Returns one entry per target, in order.
pub fn solve_discrete_log_bsgs_precomputed_batch<G: CurveGroup + Send + Sync + 'static>(
    max: u64,
    min: u64,
    base: G,
    targets: &[G],
) -> Vec<Option<u64>> {
    solve_discrete_log_bsgs_precomputed_batch_inner(
        max,
        min,
        MAX_NUM_BABY_STEPS_BATCH,
        base,
        targets,
        false,
    )
}

/// Same as `solve_discrete_log_bsgs_precomputed_batch` but tuned for dlogs above 2^32.
pub fn solve_discrete_log_bsgs_precomputed_batch_large<G: CurveGroup + Send + Sync + 'static>(
    max: u64,
    min: u64,
    base: G,
    targets: &[G],
) -> Vec<Option<u64>> {
    solve_discrete_log_bsgs_precomputed_batch_inner(
        max,
        min,
        MAX_NUM_BABY_STEPS_BATCH,
        base,
        targets,
        true,
    )
}

/// Same as `solve_discrete_log_bsgs_precomputed_batch` but with `table_size` capping the baby steps.
pub fn solve_discrete_log_bsgs_precomputed_batch_with_table_size<
    G: CurveGroup + Send + Sync + 'static,
>(
    max: u64,
    min: u64,
    table_size: u64,
    base: G,
    targets: &[G],
) -> Vec<Option<u64>> {
    solve_discrete_log_bsgs_precomputed_batch_inner(max, min, table_size, base, targets, false)
}

/// Same as `solve_discrete_log_bsgs_precomputed_batch_with_table_size` but tuned for dlogs above 2^32.
pub fn solve_discrete_log_bsgs_precomputed_batch_with_table_size_large<
    G: CurveGroup + Send + Sync + 'static,
>(
    max: u64,
    min: u64,
    table_size: u64,
    base: G,
    targets: &[G],
) -> Vec<Option<u64>> {
    solve_discrete_log_bsgs_precomputed_batch_inner(max, min, table_size, base, targets, true)
}

/// `width` is the maximum difference between the dlog of any target and `min`.
pub fn solve_discrete_log_given_table_batch<G: CurveGroup + Send + Sync + 'static>(
    table: &BabyStepsTable<G>,
    base: G,
    min: u64,
    width: u64,
    targets: &[G],
) -> Vec<Option<u64>> {
    let affines = shift_and_normalize(base, min, targets);
    solve_given_shifted_targets(table, affines[0], min, width, &affines[1..], false)
}

/// Same as `solve_discrete_log_given_table_batch` but tuned for dlogs above 2^32.
pub fn solve_discrete_log_given_table_batch_large<G: CurveGroup + Send + Sync + 'static>(
    table: &BabyStepsTable<G>,
    base: G,
    min: u64,
    width: u64,
    targets: &[G],
) -> Vec<Option<u64>> {
    let affines = shift_and_normalize(base, min, targets);
    solve_given_shifted_targets(table, affines[0], min, width, &affines[1..], true)
}

fn solve_discrete_log_bsgs_precomputed_batch_inner<G: CurveGroup + Send + Sync + 'static>(
    max: u64,
    min: u64,
    table_size: u64,
    base: G,
    targets: &[G],
    large: bool,
) -> Vec<Option<u64>> {
    if max < min || targets.is_empty() {
        return vec![None; targets.len()];
    }
    let width = max - min;
    let m = batch_num_baby_steps(table_size, width, targets.len());
    let affines = shift_and_normalize(base, min, targets);
    let Some(table) = BabyStepsTable::<G>::get_or_build(affines[0], m) else {
        return vec![None; targets.len()];
    };
    solve_given_shifted_targets(&table, affines[0], min, width, &affines[1..], large)
}

/// `base` followed by `targets[j] - base * min`, in affine and sharing one inversion.
fn shift_and_normalize<G: CurveGroup>(base: G, min: u64, targets: &[G]) -> Vec<G::Affine> {
    let shift = (min != 0).then(|| base.mul_bigint([min]));
    let mut pts = Vec::with_capacity(targets.len() + 1);
    pts.push(base);
    pts.extend(targets.iter().map(|t| shift.map_or(*t, |s| *t - s)));
    G::normalize_batch(&pts)
}

/// Baby steps for `n` targets: `sqrt(n * width / 2)` balances building the table against `n` giant walks.
/// Never below the single-target default, and capped by `table_size`.
fn batch_num_baby_steps(table_size: u64, width: u64, n: usize) -> u64 {
    let balanced = ((n as u128 * width as u128) / 2).integer_sqrt() as u64;
    balanced
        .max(MAX_NUM_BABY_STEPS)
        .min(table_size)
        .min(width)
        .max(1)
        .min(SIGN_BIT as u64 - 1)
}

fn solve_given_shifted_targets<G: CurveGroup + Send + Sync + 'static>(
    table: &BabyStepsTable<G>,
    base: G::Affine,
    min: u64,
    width: u64,
    targets: &[G::Affine],
    large: bool,
) -> Vec<Option<u64>> {
    let base = if targets.len() >= MUL_TABLE_MIN_TARGETS {
        let bits = (u64::BITS - width.leading_zeros()).max(1);
        BaseTable::get_or_build(base, bits, BaseTable::<G>::window(targets.len()))
    } else {
        None
    }
    .unwrap_or_else(|| Arc::new(BaseTable::empty(base.into_group())));
    // One target per task once there are enough targets to fill the cores: each then scans its centers
    // serially, so the per-target chunk fan-out disappears.
    #[cfg(feature = "parallel")]
    {
        if targets.len() >= rayon::current_num_threads() {
            return targets
                .par_iter()
                .map(|t| solve_given_shifted_target(table, &base, min, width, *t, large, true))
                .collect();
        }
    }
    targets
        .iter()
        .map(|t| solve_given_shifted_target(table, &base, min, width, *t, large, false))
        .collect()
}

/// Scan giant steps for centers `first_center..=last_center`, walking from `start`, the point at center
/// `first_center` (`start = t - first_center * giant` for the shifted target `t`, whose dlog is in
/// `[0, width]`). Center `c` sits at value `c * giant_step`; via the negation map one lookup there resolves
/// any dlog in the `2m + 1` window `[c * giant_step - m, c * giant_step + m]`, with `m = num_baby_steps` and
/// `giant_step = 2m + 1`. The windows tile the integers, so `0..=last_center` covers `[0, width]`. `start`
/// itself is checked only with `check_start`. The identity at a center means its dlog equals the center value.
/// A baby-table hit is confirmed by `base * v == shifted_target` before it is accepted, so a hash
/// false positive keeps scanning instead of returning a wrong dlog.
fn scan_centers<G: CurveGroup + Send + Sync + 'static>(
    baby_steps_table: &BabyStepsTable<G>,
    base_table: &BaseTable<G>,
    shifted_target: G,
    min: u64,
    width: u64,
    start: G::Affine,
    first_center: u64,
    last_center: u64,
    check_start: bool,
) -> Option<u64> {
    let giant_step = baby_steps_table.giant_step_size();
    let neg_giant = (-baby_steps_table.giant_step).into_group();
    walk_blocks(
        start.into_group(),
        start.xy(),
        neg_giant,
        &baby_steps_table.neg_giant_multiples,
        first_center,
        last_center,
        BLOCK_MIN,
        check_start,
        |c, p| {
            let base_center = c * giant_step;
            let Some((x, y)) = p else {
                // Identity at a center pins the dlog to `base_center`. Out of range only at the final
                // center, so returning `None` there ends the walk on the next step anyway.
                return (base_center <= width).then_some(min + base_center);
            };
            if let Some((i, stored_sign)) = baby_steps_table.get_unpacked(x) {
                let i = i as u64;
                // The point is `base * (base_center + i)` or `base * (base_center - i)`.
                let found = if y_sign(y) == stored_sign {
                    Some(base_center + i)
                } else {
                    base_center.checked_sub(i)
                };
                if let Some(v) = found.filter(|&v| v <= width) {
                    if base_table.mul(v) == shifted_target {
                        return Some(min + v);
                    }
                }
            }
            None
        },
    )
}

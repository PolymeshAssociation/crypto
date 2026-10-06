//! Baby-step giant-step in a pairing target group. The negation map does not apply there, so the table
//! is keyed by the target-field element itself.

use alloc::{sync::Arc, vec::Vec};
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    PrimeGroup,
};
use ark_ff::Zero;

#[cfg(feature = "std")]
use ark_serialize::CanonicalSerialize;
#[cfg(feature = "std")]
use core::any::TypeId;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "std")]
use super::setup::get_or_build_cached;
#[cfg(feature = "parallel")]
use super::setup::PAR_CHUNK_CENTERS;
use super::setup::{bsgs_hasher, BsgsHasher, FfToIndexMap, MAX_NUM_BABY_STEPS};

/// Baby steps `base * i -> i` for `i` in `[1, num_baby_steps]` in the target group, keyed by the target-field
/// element. The negation map is not used since there is no x-coordinate.
struct PairingBabyStepsTable<E: Pairing> {
    num_steps: u64,
    /// `-(base * num_baby_steps)`, added per giant step in the walk.
    neg_giant: PairingOutput<E>,
    index: FfToIndexMap,
}

impl<E: Pairing> PairingBabyStepsTable<E> {
    fn new(base: PairingOutput<E>, num_baby_steps: u64) -> Self {
        let hasher = bsgs_hasher();
        let entries = Self::baby_steps(base, num_baby_steps, &hasher);
        Self {
            num_steps: num_baby_steps,
            neg_giant: -base.mul_bigint([num_baby_steps]),
            index: FfToIndexMap::from_entries(hasher, entries),
        }
    }

    // `(first hash, second hash, i)` of `base * i` for `i` in `[1, num_steps]`, in order of `i`. Chunks of
    // `CHUNK_SIZE` steps run in parallel from anchors `base * (chunk_start + 1)`.
    fn baby_steps(
        base: PairingOutput<E>,
        num_steps: u64,
        hasher: &BsgsHasher,
    ) -> Vec<(u64, u64, u32)> {
        const CHUNK_SIZE: u64 = 1 << 12;
        let num_chunks = num_steps.div_ceil(CHUNK_SIZE);
        let chunk_step = base.mul_bigint([CHUNK_SIZE]);
        let mut anchors = Vec::with_capacity(num_chunks as usize);
        let mut anchor = base;
        for _ in 0..num_chunks {
            anchors.push(anchor);
            anchor += &chunk_step;
        }
        let points_for_chunk = |(k, &start): (usize, &PairingOutput<E>)| {
            let first = k as u64 * CHUNK_SIZE + 1;
            let last = (first - 1 + CHUNK_SIZE).min(num_steps);
            let mut out = Vec::with_capacity((last - first + 1) as usize);
            let mut cur = start;
            for i in first..=last {
                let (f1, f2) = FfToIndexMap::hash(hasher, &cur.0);
                out.push((f1, f2, i as u32));
                cur += &base;
            }
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

    fn get(&self, x: &E::TargetField) -> Option<u32> {
        self.index.get(x)
    }

    #[cfg(feature = "std")]
    fn get_or_build(base: PairingOutput<E>, num_baby_steps: u64) -> Option<Arc<Self>> {
        let mut key = Vec::with_capacity(base.compressed_size());
        base.serialize_compressed(&mut key).ok()?;
        let map_key = (TypeId::of::<Self>(), key);
        Some(get_or_build_cached(
            map_key,
            |t: &Self| t.num_steps >= num_baby_steps,
            |_| PairingBabyStepsTable::new(base, num_baby_steps),
        ))
    }

    // The table is rebuilt in each call.
    #[cfg(not(feature = "std"))]
    fn get_or_build(base: PairingOutput<E>, num_baby_steps: u64) -> Option<Arc<Self>> {
        Some(Arc::new(PairingBabyStepsTable::new(base, num_baby_steps)))
    }
}

/// Giant walk in the target group over giants `first..=last` from `anchor`, the point at giant `first`.
/// Giant `i` sits at value `i * m` and resolves any dlog in `[i * m, i * m + m]`. A table hit is confirmed
/// by `base * x == shifted_target` before it is accepted, so a hash false positive keeps scanning.
fn pairing_walk<E: Pairing>(
    table: &PairingBabyStepsTable<E>,
    base: PairingOutput<E>,
    shifted_target: PairingOutput<E>,
    min: u64,
    width: u64,
    anchor: PairingOutput<E>,
    first: u64,
    last: u64,
) -> Option<u64> {
    let m = table.num_steps;
    let mut cur = anchor;
    for i in first..=last {
        let base_center = i * m;
        if cur.is_zero() {
            if base_center <= width {
                return Some(min + base_center);
            }
        } else if let Some(b) = table.get(&cur.0) {
            if let Some(x) = base_center
                .checked_add(b as u64)
                .filter(|&x| x <= width && base.mul_bigint([x]) == shifted_target)
            {
                return Some(min + x);
            }
        }
        cur += &table.neg_giant;
    }
    None
}

/// Giants `0..=width / m` are split into chunks of `PAR_CHUNK_CENTERS`; chunk 0 is walked first from
/// `target`, the rest in parallel from anchors computed by successive additions.
fn solve_pairing_given_table<E: Pairing>(
    table: &PairingBabyStepsTable<E>,
    base: PairingOutput<E>,
    min: u64,
    width: u64,
    target: PairingOutput<E>,
) -> Option<u64> {
    let target = if min == 0 {
        target
    } else {
        target - base.mul_bigint([min])
    };
    let last_giant = width / table.num_steps;
    #[cfg(feature = "parallel")]
    {
        let cpc = PAR_CHUNK_CENTERS;
        if last_giant >= cpc {
            let chunk_count = last_giant / cpc + 1;
            if let Some(v) = pairing_walk(table, base, target, min, width, target, 0, cpc - 1) {
                return Some(v);
            }
            let chunk_step = table.neg_giant.mul_bigint([cpc]);
            let mut anchors = Vec::with_capacity((chunk_count - 1) as usize);
            let mut anchor = target + chunk_step;
            for _ in 1..chunk_count {
                anchors.push(anchor);
                anchor += &chunk_step;
            }
            return anchors
                .into_par_iter()
                .enumerate()
                .find_map_any(|(j, anchor)| {
                    let first = (j as u64 + 1) * cpc;
                    let last = (first + cpc - 1).min(last_giant);
                    pairing_walk(table, base, target, min, width, anchor, first, last)
                });
        }
    }
    pairing_walk(table, base, target, min, width, target, 0, last_giant)
}

/// Solve discrete log in the target group using Baby Step Giant Step with a table of baby steps precomputed once
/// per `base`, cached, and reused across calls. Returns `x` such that `min <= x <= max` and `base * x = target`
/// if such `x` exists, else `None`. The table holds `MAX_NUM_BABY_STEPS` baby steps (capped at `max - min`).
pub fn solve_discrete_log_bsgs_precomputed_pairing<E: Pairing>(
    max: u64,
    min: u64,
    base: PairingOutput<E>,
    target: PairingOutput<E>,
) -> Option<u64> {
    solve_discrete_log_bsgs_precomputed_pairing_with_table_size(
        max,
        min,
        MAX_NUM_BABY_STEPS,
        base,
        target,
    )
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

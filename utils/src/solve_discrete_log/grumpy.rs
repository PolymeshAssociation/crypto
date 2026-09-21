//! Grumpy giants with efficient inversion, section 3.1 of <https://eprint.iacr.org/2015/605>: two giant
//! walks, `target + j*M*base` and `2*target - k*(M+1)*base`, matched against each other and against the
//! baby steps on the x-coordinate.

use alloc::vec::Vec;
use ark_ec::{AffineRepr, CurveGroup};
use hashbrown::HashMap;
use integer_sqrt::IntegerSquareRoot;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use super::setup::walk_blocks;
use super::setup::{
    affine_multiples, bsgs_hasher, xy_sign, y_sign, AffineWalk, BabyStepsTable, BsgsHasher, Xy,
    BLOCK_MIN, MAX_NUM_BABY_STEPS, NORMALIZE_BLOCK, SIGN_BIT,
};

/// Walk indices per parallel task in `grumpy_scan_parallel`.
#[cfg(feature = "parallel")]
const GRUMPY_PAR_CHUNK: u64 = 1 << 16;

/// Walk parameters for the grumpy-giants algorithms below. `max` plays the role of the group
/// size `N`: giant steps `M = floor(sqrt(max/2)) + 1` and `M + 1`, walk length covers `2*max/(M+1)`.
fn grumpy_params(max: u64) -> (u64, u64, u64) {
    let m = (max / 2).integer_sqrt() + 1;
    let m1 = m + 1;
    let lmax = ((2u128 * max as u128) / m1 as u128 + 2) as u64;
    (m, m1, lmax.max(m1))
}

// Walk length rationale: for any `n <= max`, `2n = q*(M+1) + r` with `r <= M` and `q <= 2*max/(M+1)`,
// so the baby-g2 path alone reaches every `n` within `lmax` steps. All larger solution sets (g1-g2)
// only find `n` earlier.

/// Verify a grumpy x-match candidate: range-check against `max` and confirm `base * n == target`.
/// Every candidate from every equation family funnels through here, so a wrong-sign or out-of-range
/// candidate can never escape a probe site.
fn grumpy_check<G: CurveGroup>(base: G, target: G, max: u64, n: u128) -> Option<u64> {
    if n <= max as u128 && base.mul_bigint([n as u64]) == target {
        Some(n as u64)
    } else {
        None
    }
}

/// Baby-vs-giant-1 candidate (`G1 = (n + j*M)P` matches baby `±iP`): only `n = i - j*M` with matching
/// y-sign is possible; the `-i` case would need `n + j*M + i = order`.
fn grumpy_baby_g1(i: u64, j: u64, m: u64) -> Option<u128> {
    (i as u128).checked_sub(j as u128 * m as u128)
}

/// Baby-vs-giant-2 candidate (`G2 = (2n - k*(M+1))P` matches baby `±iP`).
fn grumpy_baby_g2(k: u64, m1: u64, i: u64, same_sign: bool) -> Option<u128> {
    let (km, i) = (k as u128 * m1 as u128, i as u128);
    let s = if same_sign {
        km.checked_add(i)?
    } else {
        km.checked_sub(i)?
    };
    (s % 2 == 0).then_some(s / 2)
}

/// Giant-vs-giant candidate (`G1 = (n + j*M)P` against `G2 = (2n - k*(M+1))P`): same y-sign gives
/// `n = j*M + k*(M+1)`, opposite y-sign gives `3n = k*(M+1) - j*M`.
fn grumpy_giant_giant(j: u64, k: u64, m: u64, m1: u64, same_sign: bool) -> Option<u128> {
    let (jm, km) = (j as u128 * m as u128, k as u128 * m1 as u128);
    if same_sign {
        jm.checked_add(km)
    } else {
        let d = km.checked_sub(jm)?;
        (d % 3 == 0).then_some(d / 3)
    }
}

/// Solve discrete log with the grumpy-giants variant using efficient inversion from section 3.1 of <https://eprint.iacr.org/2015/605>.
/// Matches on the x-coordinate with the y-sign disambiguating `±i`. Returns `x` with `0 <= x <= max`
/// and `base * x = target` if found within the walk bound, else `None`. Only practical for small `max`.
/// Reference implementation: fresh baby table per call, serial giant walk. The precomputed variant below
/// implements the same walks with a shared cached table, batched normalization, and a parallel scan.
pub fn solve_discrete_log_grumpy<G: CurveGroup>(max: u64, base: G, target: G) -> Option<u64> {
    if target.is_zero() {
        return Some(0);
    }
    if max == 0 {
        return None;
    }
    if base == target {
        return Some(1);
    }
    let (m, m1, lmax) = grumpy_params(max);
    let p0 = base.mul_bigint([m]);
    let p00 = p0 + base;
    // Baby x-table for `1..=M` via one batched normalization. Index 0 is the identity. Remainders
    // below `M + 1` suffice: every `2n` hits one within the walk, and `n <= M` hits at `l = 0`.
    let mut pts = Vec::with_capacity((m + 1) as usize);
    let mut c = G::zero();
    for _ in 0..=m {
        pts.push(c);
        c = c + base;
    }
    let aff = G::normalize_batch(&pts);
    let mut baby: HashMap<G::BaseField, (u64, bool), BsgsHasher> =
        HashMap::with_hasher(bsgs_hasher());
    for (i, a) in aff.iter().enumerate().skip(1) {
        if let Some((x, s)) = xy_sign::<G>(a) {
            baby.entry(x).or_insert((i as u64, s));
        }
    }
    let mut g1map: HashMap<G::BaseField, (u64, bool), BsgsHasher> =
        HashMap::with_hasher(bsgs_hasher());
    let mut g2map: HashMap<G::BaseField, (u64, bool), BsgsHasher> =
        HashMap::with_hasher(bsgs_hasher());
    let mut c1 = target;
    let mut c2 = target + target;
    for l in 0..=lmax {
        let a1 = c1.into_affine();
        if !a1.is_zero() {
            let (x1, sign1) = xy_sign::<G>(&a1).expect("checked non-identity");
            // G1(l) = (n + l*M)P against baby `±iP`: only `n = i - l*M` is possible, so the
            // y-sign must match.
            if let Some(&(i, s)) = baby.get(&x1) {
                if sign1 == s {
                    if let Some(n) =
                        grumpy_baby_g1(i, l, m).and_then(|n| grumpy_check(base, target, max, n))
                    {
                        return Some(n);
                    }
                }
            }
            // G1(l) against prior g2(k): same y gives `n = l*M + k*(M+1)`, opposite y gives
            // `3n = k*(M+1) - l*M`.
            if let Some(&(k, sk)) = g2map.get(&x1) {
                if let Some(n) = grumpy_giant_giant(l, k, m, m1, sign1 == sk)
                    .and_then(|n| grumpy_check(base, target, max, n))
                {
                    return Some(n);
                }
            }
            g1map.entry(x1).or_insert((l, sign1));
        }
        let a2 = c2.into_affine();
        if a2.is_zero() {
            // `2n - l*(M+1) = 0`.
            let km = l as u128 * m1 as u128;
            if km % 2 == 0 {
                if let Some(n) = grumpy_check(base, target, max, km / 2) {
                    return Some(n);
                }
            }
        } else {
            let (x2, sign2) = xy_sign::<G>(&a2).expect("checked non-identity");
            // G2(l) = (2n - l*(M+1))P against baby `±iP`.
            if let Some(&(i, s)) = baby.get(&x2) {
                if let Some(n) = grumpy_baby_g2(l, m1, i, sign2 == s)
                    .and_then(|n| grumpy_check(base, target, max, n))
                {
                    return Some(n);
                }
            }
            // G2(l) against g1(j), including the current block's g1(l) inserted above.
            if let Some(&(j, sj)) = g1map.get(&x2) {
                if let Some(n) = grumpy_giant_giant(j, l, m, m1, sign2 == sj)
                    .and_then(|n| grumpy_check(base, target, max, n))
                {
                    return Some(n);
                }
            }
            g2map.entry(x2).or_insert((l, sign2));
        }
        c1 = c1 + p0;
        c2 = c2 - p00;
    }
    None
}

/// Giant x-map value: walk index in the high bits, y-sign in the low bit.
fn pack_idx(idx: u64, sign: bool) -> u64 {
    (idx << 1) | sign as u64
}

fn unpack_idx(v: u64) -> (u64, bool) {
    (v >> 1, v & 1 != 0)
}

type GiantMap<F> = HashMap<F, u64, BsgsHasher>;

/// Step points of the two grumpy walks with their affine multiples, sized to the walk length.
struct GrumpySteps<G: CurveGroup> {
    /// `M * base`, first grumpy giant step.
    giant_1: G,
    /// `-(M + 1) * base`, second grumpy giant step.
    giant_2: G,
    giant_1_multiples: Vec<Xy<G::BaseField>>,
    giant_2_multiples: Vec<Xy<G::BaseField>>,
}

impl<G: CurveGroup> GrumpySteps<G> {
    fn new(base: G, m: u64, lmax: u64) -> Self {
        let giant_1 = base.mul_bigint([m]);
        let giant_2 = -(giant_1 + base);
        let count = (lmax / 16)
            .next_power_of_two()
            .max(16)
            .min(NORMALIZE_BLOCK as u64)
            .min(lmax.max(1)) as usize;
        Self {
            giant_1,
            giant_2,
            giant_1_multiples: affine_multiples(giant_1, count),
            giant_2_multiples: affine_multiples(giant_2, count),
        }
    }
}

/// Probe giant-2 point `p` (x-coordinate and y-sign, `None` for the identity) at index `idx` against the
/// baby table and the giant-1 x-map, returning the discrete log if this probe site resolves it. The
/// identity means `2n - idx*(M+1) = 0`.
fn probe_g2<G: CurveGroup>(
    idx: u64,
    p: Option<(&G::BaseField, bool)>,
    table: &BabyStepsTable<G>,
    g1map: &GiantMap<G::BaseField>,
    base: G,
    target: G,
    m: u64,
    m1: u64,
    max: u64,
) -> Option<u64> {
    let Some((x, s)) = p else {
        let km = idx as u128 * m1 as u128;
        return if km % 2 == 0 {
            grumpy_check(base, target, max, km / 2)
        } else {
            None
        };
    };
    if let Some((i, bs)) = table.get_unpacked(x) {
        if let Some(n) = grumpy_baby_g2(idx, m1, i as u64, s == bs)
            .and_then(|n| grumpy_check(base, target, max, n))
        {
            return Some(n);
        }
    }
    if let Some(&v) = g1map.get(x) {
        let (j, sj) = unpack_idx(v);
        if let Some(n) = grumpy_giant_giant(j, idx, m, m1, s == sj)
            .and_then(|n| grumpy_check(base, target, max, n))
        {
            return Some(n);
        }
    }
    None
}

/// Interleaved grumpy giant scan: advances the g1 walk (`target + j*p0`) and the g2 walk
/// (`2*target - k*p00`) in lockstep, one block of each per round, checking every collision as points
/// appear and returning at the first match. Holds both giant x-maps so a giant-giant collision is caught
/// whichever walk reaches the second point first.
fn grumpy_scan_interleaved<G: CurveGroup>(
    table: &BabyStepsTable<G>,
    base: G,
    target: G,
    target_xy: Option<Xy<G::BaseField>>,
    m: u64,
    m1: u64,
    lmax: u64,
    max: u64,
) -> Option<u64> {
    let steps = GrumpySteps::new(base, m, lmax);
    let cap = (lmax / 2 + 1) as usize;
    let mut g1map: GiantMap<G::BaseField> = HashMap::with_capacity_and_hasher(cap, bsgs_hasher());
    let mut g2map: GiantMap<G::BaseField> = HashMap::with_capacity_and_hasher(cap, bsgs_hasher());
    // g1(idx) against earlier g2 points, then recorded. g2(idx) against the baby table and all g1 points
    // recorded so far, then recorded.
    let mut process = |idx: u64, p1: Option<&Xy<G::BaseField>>, p2: Option<&Xy<G::BaseField>>| {
        if let Some((x1, y1)) = p1 {
            let s1 = y_sign(y1);
            if let Some(&v) = g2map.get(x1) {
                let (k, sk) = unpack_idx(v);
                if let Some(n) = grumpy_giant_giant(idx, k, m, m1, s1 == sk)
                    .and_then(|n| grumpy_check(base, target, max, n))
                {
                    return Some(n);
                }
            }
            g1map.entry(*x1).or_insert(pack_idx(idx, s1));
        }
        let p2 = p2.map(|(x, y)| (x, y_sign(y)));
        if let Some(n) = probe_g2(idx, p2, table, &g1map, base, target, m, m1, max) {
            return Some(n);
        }
        if let Some((x2, s2)) = p2 {
            g2map.entry(*x2).or_insert(pack_idx(idx, s2));
        }
        None
    };
    let two_q = target + target;
    let two_q_xy = two_q.into_affine().xy();
    if let Some(n) = process(0, target_xy.as_ref(), two_q_xy.as_ref()) {
        return Some(n);
    }
    let cap = steps.giant_1_multiples.len();
    let mut w1 = AffineWalk::new(target, 0, target_xy, steps.giant_1, cap);
    let mut w2 = AffineWalk::new(two_q, 0, two_q_xy, steps.giant_2, cap);
    let mut block = BLOCK_MIN.min(steps.giant_1_multiples.len());
    let mut b = 1u64;
    while b <= lmax {
        let n = (lmax - b + 1).min(block as u64) as usize;
        let pts1 = w1.advance(&steps.giant_1_multiples, n);
        let pts2 = w2.advance(&steps.giant_2_multiples, n);
        for k in 0..n {
            if let Some(v) = process(b + k as u64, pts1[k].as_ref(), pts2[k].as_ref()) {
                return Some(v);
            }
        }
        b += n as u64;
        block = (block * 2).min(steps.giant_1_multiples.len());
    }
    None
}

/// Parallel grumpy giant scan for large `max`: phase 1 builds the complete g1 x-map from serially
/// anchored chunks walked in parallel, phase 2 walks the g2 chunks in parallel from their anchors,
/// probing against the baby table and the merged g1 map, joined by `find_map_any`.
#[cfg(feature = "parallel")]
fn grumpy_scan_parallel<G: CurveGroup + Send + Sync>(
    table: &BabyStepsTable<G>,
    base: G,
    target: G,
    m: u64,
    m1: u64,
    lmax: u64,
    max: u64,
) -> Option<u64> {
    let steps = GrumpySteps::new(base, m, lmax);
    let starts: Vec<u64> = (0..=lmax).step_by(GRUMPY_PAR_CHUNK as usize).collect();
    let end_of = |t: usize| starts[t].saturating_add(GRUMPY_PAR_CHUNK - 1).min(lmax);
    // Serial per-chunk anchors: `anchors[t]` and `g2anchors[t]` are chunk `t`'s g1 and g2 points at
    // index `starts[t]`.
    let chunk_giant = steps.giant_1.mul_bigint([GRUMPY_PAR_CHUNK]);
    let chunk_giant2 = steps.giant_2.mul_bigint([GRUMPY_PAR_CHUNK]);
    let mut anchors = Vec::with_capacity(starts.len());
    let mut g2anchors = Vec::with_capacity(starts.len());
    let (mut a1, mut a2) = (target, target + target);
    for _ in &starts {
        anchors.push(a1);
        g2anchors.push(a2);
        a1 += chunk_giant;
        a2 += chunk_giant2;
    }
    let anchors_xy: Vec<_> = G::normalize_batch(&anchors)
        .iter()
        .map(|p| p.xy())
        .collect();
    let g2anchors_xy: Vec<_> = G::normalize_batch(&g2anchors)
        .iter()
        .map(|p| p.xy())
        .collect();
    // Phase 1: build each chunk's g1 x-map in parallel, then merge into one pre-sized map.
    let local_maps: Vec<GiantMap<G::BaseField>> = (0..starts.len())
        .into_par_iter()
        .map(|t| {
            let (first, last) = (starts[t], end_of(t));
            let mut map: GiantMap<G::BaseField> =
                HashMap::with_capacity_and_hasher((last - first + 1) as usize, bsgs_hasher());
            let _ = walk_blocks(
                anchors[t],
                anchors_xy[t],
                steps.giant_1,
                &steps.giant_1_multiples,
                first,
                last,
                NORMALIZE_BLOCK,
                true,
                |idx, p| {
                    if let Some((x, y)) = p {
                        map.entry(*x).or_insert(pack_idx(idx, y_sign(y)));
                    }
                    None
                },
            );
            map
        })
        .collect();
    let total: usize = local_maps.iter().map(|lm| lm.len()).sum();
    let mut g1map: GiantMap<G::BaseField> = HashMap::with_capacity_and_hasher(total, bsgs_hasher());
    for map in local_maps {
        g1map.extend(map);
    }
    // Phase 2: each chunk walks g2 from its anchor, probing against the baby table and `g1map`.
    (0..starts.len()).into_par_iter().find_map_any(|t| {
        walk_blocks(
            g2anchors[t],
            g2anchors_xy[t],
            steps.giant_2,
            &steps.giant_2_multiples,
            starts[t],
            end_of(t),
            BLOCK_MIN,
            true,
            |idx, p| {
                let p = p.map(|(x, y)| (x, y_sign(y)));
                probe_g2(idx, p, table, &g1map, base, target, m, m1, max)
            },
        )
    })
}

/// Fast-path check that `target` is itself a baby step, then the grumpy scan. Callers must have
/// handled `target == 0`, `max == 0`, and `base == target`.
fn grumpy_scan_with_table<G: CurveGroup + Send + Sync>(
    table: &BabyStepsTable<G>,
    base: G,
    target: G,
    target_aff: G::Affine,
    max: u64,
) -> Option<u64> {
    let target_xy = target_aff.xy();
    // An x-hit with matching y-sign means `target = base * i`, so its dlog is `i`.
    if let Some((x, y)) = &target_xy {
        if let Some((i, s)) = table.get_unpacked(x) {
            if y_sign(y) == s {
                let n = i as u64;
                if n <= max && base.mul_bigint([n]) == target {
                    return Some(n);
                }
            }
        }
    }
    let (m, m1, lmax) = grumpy_params(max);
    // Large max spanning multiple parallel chunks: phased chunked scan. Otherwise (the common small
    // case, and every no-`parallel` build) the interleaved scan, which exits at the first collision.
    #[cfg(feature = "parallel")]
    {
        if lmax >= GRUMPY_PAR_CHUNK {
            return grumpy_scan_parallel(table, base, target, m, m1, lmax, max);
        }
    }
    grumpy_scan_interleaved(table, base, target, target_xy, m, m1, lmax, max)
}

/// Solve discrete log with grumpy giants and efficient inversion (section 3.1 of <https://eprint.iacr.org/2015/605>)
/// reusing the cached `BabyStepsTable` for the baby side and batched normalization for the giant walks.
/// Returns `x` with `0 <= x <= max` and `base * x = target` if found, else `None`.
/// Memory scales with `sqrt(max)`: the baby table plus the giant x-map(s).
pub fn solve_discrete_log_grumpy_precomputed<G: CurveGroup + Send + Sync + 'static>(
    max: u64,
    base: G,
    target: G,
) -> Option<u64> {
    solve_discrete_log_grumpy_precomputed_with_table_size(max, MAX_NUM_BABY_STEPS, base, target)
}

/// Same as `solve_discrete_log_grumpy_precomputed` but with `table_size` baby steps.
pub fn solve_discrete_log_grumpy_precomputed_with_table_size<
    G: CurveGroup + Send + Sync + 'static,
>(
    max: u64,
    table_size: u64,
    base: G,
    target: G,
) -> Option<u64> {
    if target.is_zero() {
        return Some(0);
    }
    if max == 0 {
        return None;
    }
    if base == target {
        return Some(1);
    }
    let affines = G::normalize_batch(&[base, target]);
    let (base_aff, target_aff) = (affines[0], affines[1]);
    // Clamp like `solve_discrete_log_bsgs_precomputed_inner` so the index fits below the sign bit.
    let tbl = table_size.min(max).max(1).min(SIGN_BIT as u64 - 1);
    let table = BabyStepsTable::<G>::get_or_build(base_aff, tbl)?;
    grumpy_scan_with_table(&table, base, target, target_aff, max)
}

/// Solve discrete log with grumpy giants given an already constructed baby table, mirroring
/// `solve_discrete_log_given_table`. Only the table's x-coordinate map is used; its giant step is
/// ignored. Completeness does not depend on the table size (the giant-1/giant-2 families cover every
/// in-range dlog on their own); a larger baby table only lets more dlogs resolve early via the baby
/// path, trading space for speed.
pub fn solve_discrete_log_grumpy_given_table<G: CurveGroup + Send + Sync + 'static>(
    table: &BabyStepsTable<G>,
    base: G,
    max: u64,
    target: G,
) -> Option<u64> {
    if target.is_zero() {
        return Some(0);
    }
    if max == 0 {
        return None;
    }
    if base == target {
        return Some(1);
    }
    grumpy_scan_with_table(table, base, target, target.into_affine(), max)
}

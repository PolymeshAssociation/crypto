//! The original solvers, keyed by the string encoding of a group element.

use alloc::string::ToString;
use ark_ff::AdditiveGroup;
use hashbrown::HashMap;

#[cfg(feature = "ahash")]
use ahash::RandomState;
#[cfg(not(feature = "std"))]
use integer_sqrt::IntegerSquareRoot;

use super::setup::MAX_NUM_BABY_STEPS;

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

use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};
use ark_std::{cfg_into_iter, cfg_iter, cfg_iter_mut, vec::Vec};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[macro_export]
macro_rules! cfg_iter_sum {
    ($iter: expr, $initial: tt) => {{
        #[cfg(feature = "parallel")]
        let result = $iter.reduce($initial, |a, b| a + b);

        #[cfg(not(feature = "parallel"))]
        let result = $iter.fold($initial(), |a, b| a + b);

        result
    }};
}

/// Inner product of 2 vectors `a` and `b`
pub fn inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    let size = a.len().min(b.len());
    let product = cfg_into_iter!(0..size).map(|i| a[i] * b[i]);
    let zero = F::zero;
    cfg_iter_sum!(product, zero)
}

/// Hadamard product of two vectors of scalars
pub fn hadamard_product<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    cfg_iter!(a)
        .zip(cfg_iter!(b))
        .map(|(a_i, b_i)| *a_i * b_i)
        .collect()
}

/// Add two vectors of scalars
pub fn add_vecs<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    let (a_len, b_len) = (a.len(), b.len());
    let res_len = ark_std::cmp::max(a_len, b_len);
    cfg_into_iter!(0..res_len)
        .map(|i| {
            let a_i = a.get(i).cloned().unwrap_or(F::zero());
            let b_i = b.get(i).cloned().unwrap_or(F::zero());
            a_i + b_i
        })
        .collect()
}

/// Weighted inner product of 2 vectors `a` and `b` with weight `w`. Calculated as `\sum_{i=0}(a_i * b_i * w^{i+1})`
pub fn weighted_inner_product<F: Field>(a: &[F], b: &[F], w: &F) -> F {
    let size = a.len().min(b.len());

    let mut weights = powers(w, size as u32 + 1);
    weights.remove(0);

    let product = cfg_into_iter!(0..size).map(|i| a[i] * b[i] * weights[i]);

    let zero = F::zero;
    cfg_iter_sum!(product, zero)
}

/// Weighted inner product of the vector `n` with itself. Calculated as `\sum_{i=0}(n_i * n_i * w^{i+1})`
pub fn weighted_norm<F: Field>(n: &[F], w: &F) -> F {
    weighted_inner_product(n, n, w)
}

pub fn non_zero_random<F: Field, R: RngCore + CryptoRng>(rng: &mut R) -> F {
    let mut r = F::rand(rng);
    while r.is_zero() {
        r = F::rand(rng);
    }
    r
}

// Two residue classes expose independent field multiplications while sharing
// one squared step. Appending avoids zero-filling the output before it is
// overwritten; short vectors retain the smaller scalar loop.
const TWO_LANE_POWER_VECTOR_MIN_LEN: usize = 16;

/// Powers of a finite field as `[1, s, s^2, .. s^{num-1}]`
pub fn powers<F: Field>(s: &F, num: u32) -> Vec<F> {
    powers_starting_from(F::one(), s, num)
}

/// Powers of a finite field as `[start, start*exp, start * exp^2, .. start * exp^{num-1}]`
pub fn powers_starting_from<F: Field>(start: F, exp: &F, num: u32) -> Vec<F> {
    let len = num as usize;
    if len == 0 {
        return Vec::new();
    }
    if len >= TWO_LANE_POWER_VECTOR_MIN_LEN {
        // Taken from https://github.com/zakura-core/common/blob/main/crates/halo2_proofs/src/poly.rs
        let step = exp.square();
        let mut lane_powers = [start, start * exp];
        let mut powers = Vec::with_capacity(len);
        powers.extend_from_slice(&lane_powers);
        while len - powers.len() >= lane_powers.len() {
            lane_powers[0] *= step;
            lane_powers[1] *= step;
            powers.extend_from_slice(&lane_powers);
        }
        if powers.len() != len {
            lane_powers[0] *= step;
            powers.push(lane_powers[0]);
        }
        return powers;
    }

    let mut powers = Vec::with_capacity(len);
    powers.push(start);
    for i in 1..len {
        powers.push(powers[i - 1] * exp);
    }
    powers
}

/// SUM of a geometric progression
/// SUM a^i = (1 - a^n) / (1 - a) = -(1-a^n)/-(1-a)
/// = (a^n - 1) / (a - 1)
pub fn sum_of_powers<F: Field>(r: &F, num: u32) -> F {
    (r.pow([num as u64]) - &F::one()) * (*r - F::one()).inverse().unwrap()
}

/// Return a scaled vector by multiplying each of its elements by `factor`. Returns `[arr_0 * factor, arr_1 * factor, arr_2 * factor, ...]`
pub fn scale<F: Field>(arr: &[F], factor: &F) -> Vec<F> {
    cfg_iter!(arr).map(|elem| *elem * factor).collect()
}

/// Mutate a vector `arr` by multiplying each of its elements by `factor`.
pub fn scale_mut<F: Field>(arr: &mut [F], factor: &F) {
    cfg_iter_mut!(arr).for_each(|elem| *elem = *elem * factor);
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::misc::n_rand;
    use ark_bls12_381::Fr;
    use ark_ff::{Field, One, Zero};
    use ark_std::{
        rand::{prelude::StdRng, SeedableRng},
        UniformRand,
    };

    #[test]
    fn check_inner_product_and_norm() {
        let mut rng = StdRng::seed_from_u64(0u64);

        let count = 10;
        let a = n_rand(&mut rng, count).collect::<Vec<_>>();
        let b = n_rand(&mut rng, count).collect::<Vec<_>>();

        let weight = Fr::rand(&mut rng);

        let res1 = inner_product(&a, &b);
        let mut res2 = Fr::zero();
        for i in 0..count {
            res2 += a[i] * b[i];
        }
        assert_eq!(res1, res2);

        let res3 = weighted_inner_product(&a, &b, &weight);
        let mut res4 = Fr::zero();
        for i in 0..count {
            res4 += a[i] * b[i] * weight.pow(&[i as u64 + 1]);
        }
        assert_eq!(res3, res4);

        let res5 = weighted_norm(&a, &weight);
        let mut res6 = Fr::zero();
        for i in 0..count {
            res6 += a[i] * a[i] * weight.pow(&[i as u64 + 1]);
        }
        assert_eq!(res5, res6);
    }

    #[test]
    fn check_powers() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let start = Fr::rand(&mut rng);

        for base in [Fr::zero(), Fr::one(), -Fr::one(), Fr::rand(&mut rng)] {
            for num in [0u32, 1, 2, 3, 15, 16, 17, 31, 32, 100, 101] {
                let mut expected = Vec::with_capacity(num as usize);
                let mut acc = Fr::one();
                for _ in 0..num {
                    expected.push(acc);
                    acc *= base;
                }
                assert_eq!(powers(&base, num), expected);

                let expected_from: Vec<_> = expected.iter().map(|p| start * p).collect();
                assert_eq!(powers_starting_from(start, &base, num), expected_from);
            }
        }
    }
}

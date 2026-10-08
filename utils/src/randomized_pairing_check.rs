use crate::{
    checker_guard::{CheckerGuard, GuardedCheck},
    error::UtilsError,
};
#[cfg(feature = "ahash")]
use ahash::RandomState;
use ark_ec::{
    pairing::{MillerLoopOutput, Pairing, PairingOutput},
    AffineRepr, CurveGroup, PrimeGroup, VariableBaseMSM,
};
use ark_ff::{One, PrimeField, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::rand::{CryptoRng, RngCore};
use ark_std::{
    cfg_iter,
    hash::{Hash, Hasher},
    mem,
    ops::MulAssign,
    vec,
    vec::Vec,
    UniformRand,
};
use hashbrown::HashMap;
use itertools::Itertools;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Number of distinct targets from which they are combined with an MSM in the target group rather
/// than exponentiating each. The measured crossover is 32 on BLS12-381 and 24 on BN254.
const TARGET_MSM_THRESHOLD: usize = 32;

/// Bytes of a serialized `E::G2Prepared` fed to the hasher.
const PREPARED_HASH_PREFIX_SIZE: usize = 128;

/// x or y coordinate of a G2 point
type G2Coord<E> = <<E as Pairing>::G2Affine as AffineRepr>::BaseField;

#[cfg(feature = "ahash")]
type Map<K, V> = HashMap<K, V, RandomState>;
#[cfg(not(feature = "ahash"))]
type Map<K, V> = HashMap<K, V>;

#[cfg(feature = "ahash")]
fn new_map<K, V>() -> Map<K, V> {
    HashMap::with_hasher(RandomState::new())
}

#[cfg(not(feature = "ahash"))]
fn new_map<K, V>() -> Map<K, V> {
    HashMap::new()
}

/// Serialization of an `E::G2Prepared`, used to find the group it belongs to. Only a prefix is
/// hashed, equality compares all of it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedKey(Vec<u8>);

impl PreparedKey {
    fn new<P: CanonicalSerialize>(prepared: &P) -> Self {
        let mut bytes = Vec::with_capacity(prepared.compressed_size());
        prepared.serialize_compressed(&mut bytes).unwrap();
        Self(bytes)
    }
}

impl Hash for PreparedKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0[..self.0.len().min(PREPARED_HASH_PREFIX_SIZE)].hash(state)
    }
}

#[derive(Debug, Clone)]
enum GroupG2<E: Pairing> {
    Affine(E::G2Affine),
    Prepared(E::G2Prepared),
}

impl<E: Pairing> GroupG2<E> {
    fn prepare(self) -> E::G2Prepared {
        match self {
            Self::Affine(b) => E::G2Prepared::from(b),
            Self::Prepared(b) => b,
        }
    }
}

/// Pairs sharing a G2 element `b`. `\prod_i{e(a_i, b)^{m_i}} = e(\sum_i{m_i * a_i}, b)`, so the whole
/// group costs one multi-scalar multiplication in G1 and one miller loop.
#[derive(Debug, Clone)]
struct PairsWithSameG2<E: Pairing> {
    a: Vec<E::G1Affine>,
    b: GroupG2<E>,
    multipliers: Vec<E::ScalarField>,
}

impl<E: Pairing> PairsWithSameG2<E> {
    fn new(a: E::G1Affine, m: E::ScalarField, b: GroupG2<E>) -> Self {
        Self {
            a: vec![a],
            multipliers: vec![m],
            b,
        }
    }

    fn push(&mut self, a: E::G1Affine, m: E::ScalarField) {
        self.a.push(a);
        self.multipliers.push(m);
    }

    /// `\sum_i{m_i * a_i}`
    fn combine(&self) -> E::G1 {
        if self.a.len() == 1 {
            if self.multipliers[0].is_one() {
                self.a[0].into_group()
            } else {
                self.a[0].mul_bigint(self.multipliers[0].into_bigint())
            }
        } else {
            E::G1::msm_unchecked_full_width(&self.a, &self.multipliers)
        }
    }
}

/// A `CheckerGuard` holding a `RandomizedPairingChecker`.
pub type RandomizedPairingCheckerGuard<E> = CheckerGuard<RandomizedPairingChecker<E>>;

impl<E: Pairing> RandomizedPairingCheckerGuard<E> {
    /// Create a guard holding a checker using the given random value. If `lazy` is set to true, delays
    /// the miller loops till the end (unless overridden) trading off memory for CPU time.
    pub fn new(random: E::ScalarField, lazy: bool) -> Self {
        Self::wrap(RandomizedPairingChecker::_new(random, lazy))
    }

    /// Same as `Self::new` except that this generates a random value
    pub fn new_using_rng<R: RngCore + CryptoRng>(rng: &mut R, lazy: bool) -> Self {
        Self::new(E::ScalarField::rand(rng), lazy)
    }
}

impl<E: Pairing> GuardedCheck for RandomizedPairingChecker<E> {
    fn verify(&mut self) -> Result<(), UtilsError> {
        self.do_verify()
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }
}

/// Inspired from Snarkpack implementation - <https://github.com/nikkolasg/snarkpack/blob/main/src/pairing_check.rs>
/// RandomizedPairingChecker represents a check of the form `e(A,B) + e(C,D) + ... = T`. Checks can
/// be aggregated together using random linear combination. The efficiency comes
/// from keeping the results from the miller loop output before proceeding to a final
/// exponentiation when verifying if all checks are verified.
/// For each pairing equation, multiply by a power of a random element created during initialization
/// eg. to check 3 pairing equations `e(A1, B1) == O1, e(A2, B2) == O2 and e(A3, B3) == O3`, a single
/// equation can be checked as `e(A1, B1) + e(A2, B2)*r + e(A3, B3)*r^2 == O1 + O2*r + O3*r^2` which is
/// same as checking `e(A1, B1) + e(A2*r, B2) + e(A3*r^2, B3) == O1 + O2*r + O3*r^2`
/// Similarly to check 3 pairing equations `e(A1, B1) == e(C1, D1), e(A2, B2) == e(C2, D2) and e(A3, B3) == e(C3, D3)`,
/// a single check can done as `e(A1, B1) + e(A2, B2)*r + e(A3, B3)*r^2 == e(C1, D1) + e(C2, D2)*r + e(C3, D3)*r^2` which
/// is same as checking `e(A1, B1) + e(A2*r, B2) + e(A3*r^2, B3) + e(C1*-1, D1) + e(C2*-r, D2) + e(C3*-r^2, D3)== 1`
/// Pairs are not scaled and paired one by one but kept grouped by their G2 element, so the pairs
/// `e(A1*m1, B), e(A2*m2, B), ...` cost a single multi-scalar multiplication `A1*m1 + A2*m2 + ...`
/// and a single miller loop. The same is done on the target side where the multipliers of equal
/// targets are added before a single exponentiation.
/// The right hand side of the equation can be given either as a `E::G2Affine`, which is only prepared
/// once per distinct point at the end, or as an `E::G2Prepared`, which is grouped by its serialization.
/// The two forms of the same point do not share a group.
#[derive(Debug)]
pub struct RandomizedPairingChecker<E: Pairing> {
    /// a miller loop result that is to be multiplied by other miller loop results
    /// before going into a final exponentiation result
    left: MillerLoopOutput<E>,
    /// a right side result which is already in the right subgroup Gt which is to
    /// be compared to the left side when "final_exponentiatiat"-ed
    right: PairingOutput<E>,
    /// If true, delays the miller loops of the accumulated pairs till the end (unless overridden)
    /// trading off memory for CPU time.
    lazy: bool,
    /// Pairs accumulated since the last miller loop, grouped by their G2 element
    groups: Vec<PairsWithSameG2<E>>,
    /// Group of a G2 given as an affine point, keyed by its x coordinate. Holds the group's index
    /// and the y coordinate it was created with, so `b` and `-b` share a group.
    g2_by_affine: Map<G2Coord<E>, (usize, G2Coord<E>)>,
    /// Group of a G2 given in prepared form, keyed by its serialization
    g2_by_prepared: Map<PreparedKey, usize>,
    /// Multiplier of each distinct target not yet combined into `self.right`
    targets: Map<PairingOutput<E>, E::ScalarField>,
    random: E::ScalarField,
    /// For each pairing equation, its multiplied by `self.random`
    current_random: E::ScalarField,

    /// Flag to detect forgotten `verify()`.
    verified: bool,
    /// Flag to detect a canceled checker.
    cancelled: bool,
}

impl<E: Pairing> RandomizedPairingChecker<E> {
    fn _new(random: E::ScalarField, lazy: bool) -> Self {
        Self {
            left: MillerLoopOutput(E::TargetField::one()),
            right: PairingOutput::zero(),
            lazy,
            groups: vec![],
            g2_by_affine: new_map(),
            g2_by_prepared: new_map(),
            targets: new_map(),
            random,
            current_random: E::ScalarField::one(),
            verified: false,
            cancelled: false,
        }
    }

    /// Create a checker using given random number. If `lazy` is set to true, delays the computation
    /// of miller loops till the end (unless overridden) trading off memory for CPU time.
    #[deprecated = "Use `RandomizedPairingCheckerGuard::new` or `RandomizedPairingCheckerGuard::new_using_rng` instead"]
    pub fn new(random: E::ScalarField, lazy: bool) -> Self {
        Self::_new(random, lazy)
    }

    /// Same as `Self::new` except that this generates a random value
    #[deprecated = "Use `RandomizedPairingCheckerGuard::new` or `RandomizedPairingCheckerGuard::new_using_rng` instead"]
    pub fn new_using_rng<R: RngCore + CryptoRng>(rng: &mut R, lazy: bool) -> Self {
        Self::_new(E::ScalarField::rand(rng), lazy)
    }

    /// Add single elements from source and target groups
    pub fn add_sources_and_target(
        &mut self,
        a: &E::G1Affine,
        b: impl Into<E::G2Prepared>,
        out: &PairingOutput<E>,
    ) {
        let m = self.current_random;
        self.add_pair_prepared(a, b, m);
        self.add_target(out, m);
        self.check_added(self.lazy);
    }

    /// Same as `Self::add_sources_and_target` but takes the G2 element as an affine point
    pub fn add_sources_and_target_g2_affine(
        &mut self,
        a: &E::G1Affine,
        b: &E::G2Affine,
        out: &PairingOutput<E>,
    ) {
        let m = self.current_random;
        self.add_pair_affine(a, b, m);
        self.add_target(out, m);
        self.check_added(self.lazy);
    }

    /// Add a sequence of group elements whose pairing product must be equal to the given target field
    /// element, i.e. `\prod_{i}(e(a_i, b_i)) = out`
    pub fn add_multiple_sources_and_target(
        &mut self,
        a: &[E::G1Affine],
        b: impl IntoIterator<Item = impl Into<E::G2Prepared>>,
        out: &PairingOutput<E>,
    ) {
        self.add_multiple_sources_and_target_with_laziness_choice(a, b, out, self.lazy)
    }

    /// Same as `Self::add_multiple_sources_and_target` but takes the G2 elements as affine points
    pub fn add_multiple_sources_and_target_g2_affine(
        &mut self,
        a: &[E::G1Affine],
        b: &[E::G2Affine],
        out: &PairingOutput<E>,
    ) {
        let m = self.current_random;
        for (a, b) in a.iter().zip_eq(b) {
            self.add_pair_affine(a, b, m);
        }
        self.add_target(out, m);
        self.check_added(self.lazy);
    }

    /// Add a sequence of group elements whose pairing product must be equal to the another given sequence
    /// of group elements, i.e. `\prod_{i}(e(a_i, b_i)) = \prod_{i}(e(c_i, d_i))`
    pub fn add_multiple_sources(
        &mut self,
        a: &[E::G1Affine],
        b: impl IntoIterator<Item = impl Into<E::G2Prepared>>,
        c: &[E::G1Affine],
        d: impl IntoIterator<Item = impl Into<E::G2Prepared>>,
    ) {
        self.add_multiple_sources_with_laziness_choice(a, b, c, d, self.lazy)
    }

    /// Same as `Self::add_multiple_sources` but takes the G2 elements as affine points
    pub fn add_multiple_sources_g2_affine(
        &mut self,
        a: &[E::G1Affine],
        b: &[E::G2Affine],
        c: &[E::G1Affine],
        d: &[E::G2Affine],
    ) {
        let m = self.current_random;
        for (a, b) in a.iter().zip_eq(b) {
            self.add_pair_affine(a, b, m);
        }
        for (c, d) in c.iter().zip_eq(d) {
            self.add_pair_affine(c, d, -m);
        }
        self.check_added(self.lazy);
    }

    /// Add 2 group elements whose pairing should be equal to the pairing of another 2 given group
    /// elements, i.e. `e(a, b) = e(c, d)`
    pub fn add_sources(
        &mut self,
        a: &E::G1Affine,
        b: impl Into<E::G2Prepared>,
        c: &E::G1Affine,
        d: impl Into<E::G2Prepared>,
    ) {
        self.add_sources_with_laziness_choice(a, b, c, d, self.lazy)
    }

    /// Same as `Self::add_sources` but takes the G2 elements as affine points
    pub fn add_sources_g2_affine(
        &mut self,
        a: &E::G1Affine,
        b: &E::G2Affine,
        c: &E::G1Affine,
        d: &E::G2Affine,
    ) {
        let m = self.current_random;
        self.add_pair_affine(a, b, m);
        self.add_pair_affine(c, d, -m);
        self.check_added(self.lazy);
    }

    /// Same as `Self::add_multiple_sources_and_target` except that this accepts whether to be lazy or
    /// not and does not default to laziness decided during creation of the checker
    pub fn add_multiple_sources_and_target_with_laziness_choice(
        &mut self,
        a: &[E::G1Affine],
        b: impl IntoIterator<Item = impl Into<E::G2Prepared>>,
        out: &PairingOutput<E>,
        lazy: bool,
    ) {
        let m = self.current_random;
        for (a, b) in a.iter().zip_eq(b) {
            self.add_pair_prepared(a, b, m);
        }
        self.add_target(out, m);
        self.check_added(lazy);
    }

    /// Same as `Self::add_multiple_sources` except that this accepts whether to be lazy or
    /// not and does not default to laziness decided during creation of the checker
    pub fn add_multiple_sources_with_laziness_choice(
        &mut self,
        a: &[E::G1Affine],
        b: impl IntoIterator<Item = impl Into<E::G2Prepared>>,
        c: &[E::G1Affine],
        d: impl IntoIterator<Item = impl Into<E::G2Prepared>>,
        lazy: bool,
    ) {
        let m = self.current_random;
        for (a, b) in a.iter().zip_eq(b) {
            self.add_pair_prepared(a, b, m);
        }
        for (c, d) in c.iter().zip_eq(d) {
            self.add_pair_prepared(c, d, -m);
        }
        self.check_added(lazy);
    }

    /// Same as `Self::add_sources` except that this accepts whether to be lazy or
    /// not and does not default to laziness decided during creation of the checker
    pub fn add_sources_with_laziness_choice(
        &mut self,
        a: &E::G1Affine,
        b: impl Into<E::G2Prepared>,
        c: &E::G1Affine,
        d: impl Into<E::G2Prepared>,
        lazy: bool,
    ) {
        let m = self.current_random;
        self.add_pair_prepared(a, b, m);
        self.add_pair_prepared(c, d, -m);
        self.check_added(lazy);
    }

    /// Verify that all added pairing equations are satisfied.
    pub fn verify(&mut self) -> Result<(), UtilsError> {
        self.do_verify()
    }

    fn do_verify(&mut self) -> Result<(), UtilsError> {
        if self.cancelled {
            return Err(UtilsError::PairingCheckFailed);
        }
        self.verified = true;
        self.run_miller_loop();
        self.combine_targets();
        if E::final_exponentiation(self.left).unwrap() == self.right {
            Ok(())
        } else {
            Err(UtilsError::PairingCheckFailed)
        }
    }

    /// Cancel the checker. This is useful when the verifier wants to cancel the checker if it is not needed anymore,
    /// say a different check failed which fails verification anyway
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Number of distinct G2 elements among the pairs accumulated since the last miller loop. Each
    /// of them costs one multi-scalar multiplication and one miller loop.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// Number of distinct targets accumulated so far.
    pub fn num_targets(&self) -> usize {
        self.targets.len()
    }

    /// Add `e(a, b)^m` to the left side of the check, with `b` given as an affine point.
    fn add_pair_affine(&mut self, a: &E::G1Affine, b: &E::G2Affine, m: E::ScalarField) {
        if self.cancelled {
            return;
        }
        if a.is_zero() || b.is_zero() {
            return;
        }
        // unwrap is fine as point is not at infinity
        let (x, y) = b.xy().unwrap();
        match self.g2_by_affine.get(&x).copied() {
            Some((i, y_i)) => self.groups[i].push(*a, if y_i == y { m } else { -m }),
            None => {
                self.g2_by_affine.insert(x, (self.groups.len(), y));
                self.groups
                    .push(PairsWithSameG2::new(*a, m, GroupG2::Affine(*b)));
            }
        }
    }

    /// Add `e(a, b)^m` to the left side of the check, with `b` given in prepared form
    fn add_pair_prepared(&mut self, a: &E::G1Affine, b: impl Into<E::G2Prepared>, m: E::ScalarField) {
        if self.cancelled {
            return;
        }
        if a.is_zero() {
            return;
        }
        let b = b.into();
        let key = PreparedKey::new(&b);
        match self.g2_by_prepared.get(&key).copied() {
            Some(i) => self.groups[i].push(*a, m),
            None => {
                self.g2_by_prepared.insert(key, self.groups.len());
                self.groups
                    .push(PairsWithSameG2::new(*a, m, GroupG2::Prepared(b)));
            }
        }
    }

    /// Add `out^m` to the right side of the check
    fn add_target(&mut self, out: &PairingOutput<E>, m: E::ScalarField) {
        if self.cancelled || out.is_zero() {
            return;
        }
        *self
            .targets
            .entry(*out)
            .or_insert_with(E::ScalarField::zero) += m;
    }

    /// Move to the multiplier of the next check and, when not lazy, clear the accumulated pairs
    /// by running their miller loops.
    fn check_added(&mut self, lazy: bool) {
        self.current_random *= self.random;
        if !lazy {
            self.run_miller_loop();
        }
    }

    /// Combine each group into a single G1 point, run the miller loops of all of them and multiply
    /// the result into the left side.
    fn run_miller_loop(&mut self) {
        if self.cancelled || self.groups.is_empty() {
            return;
        }
        self.g2_by_affine.clear();
        self.g2_by_prepared.clear();
        let groups = mem::take(&mut self.groups);
        let combined = cfg_iter!(groups).map(|g| g.combine()).collect::<Vec<_>>();
        let (a, b): (Vec<_>, Vec<_>) = E::G1::normalize_batch(&combined)
            .into_iter()
            .zip(groups)
            .filter(|(a, _)| !a.is_zero())
            .map(|(a, g)| (E::G1Prepared::from(a), g.b.prepare()))
            .unzip();
        self.left.0.mul_assign(E::multi_miller_loop(a, b).0);
    }

    /// Exponentiate each distinct target by its multiplier and add the result to the right side.
    fn combine_targets(&mut self) {
        if self.targets.is_empty() {
            return;
        }
        let combined = if self.targets.len() >= TARGET_MSM_THRESHOLD {
            let (t, m): (Vec<_>, Vec<_>) = self.targets.drain().unzip();
            PairingOutput::msm_unchecked_full_width(&t, &m)
        } else {
            self.targets
                .drain()
                .fold(PairingOutput::zero(), |acc, (t, m)| {
                    acc + if m.is_one() {
                        t
                    } else {
                        t.mul_bigint(m.into_bigint())
                    }
                })
        };
        self.right += combined;
    }
}

impl<E: Pairing> Drop for RandomizedPairingChecker<E> {
    fn drop(&mut self) {
        if self.cancelled || self.verified {
            return;
        }
        // Panicking here while already unwinding aborts the process.
        #[cfg(feature = "std")]
        if std::thread::panicking() {
            return;
        }
        // Only panic if verify fails.
        if let Err(err) = self.do_verify() {
            log::error!("Skipped `verify` call returns error: err={err:?}");
            // Panic as this code path should never be reached in production and be caught in testing
            panic!(
                "RandomizedPairingChecker was dropped without calling `verify()`. \
                This means all accumulated multi-pairing checks were never performed. \
                This indicates a bug in caller's verifier code"
            );
        } else {
            // Log an error, since the code should be fixed to call `verify`.
            log::error!(
                "RandomizedPairingChecker was dropped without calling `verify()`. \
                This means all accumulated multi-pairing checks were never performed. \
                This indicates a bug in caller's verifier code"
            );
        }
    }
}
#[cfg(test)]
mod test {
    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_bn254::Bn254;
    use ark_ec::CurveGroup;
    use ark_std::{
        rand::{prelude::StdRng, SeedableRng},
        UniformRand,
    };
    use std::time::Instant;

    type Guard = RandomizedPairingCheckerGuard<Bls12_381>;

    fn rand_g1<E: Pairing>(n: usize, rng: &mut StdRng) -> Vec<E::G1Affine> {
        (0..n).map(|_| E::G1Affine::rand(rng)).collect()
    }

    fn rand_g2<E: Pairing>(n: usize, rng: &mut StdRng) -> Vec<E::G2Affine> {
        (0..n).map(|_| E::G2Affine::rand(rng)).collect()
    }

    fn prepare<E: Pairing>(b: &[E::G2Affine]) -> Vec<E::G2Prepared> {
        b.iter().map(|b| E::G2Prepared::from(*b)).collect()
    }

    fn rev_vec<T: Clone>(v: &[T]) -> Vec<T> {
        let mut x = v.to_vec();
        x.reverse();
        x
    }

    #[test]
    fn test_pairing_randomize() {
        fn check<E: Pairing>(curve: &str) {
            let mut rng = StdRng::seed_from_u64(0u64);
            let n = 10;
            let mut t1 = 0;

            let a1 = rand_g1::<E>(n, &mut rng);
            let b1 = rand_g2::<E>(n, &mut rng);
            let a2 = rand_g1::<E>(n + 5, &mut rng);
            let b2 = rand_g2::<E>(n + 5, &mut rng);
            let a3 = rand_g1::<E>(n - 2, &mut rng);
            let b3 = rand_g2::<E>(n - 2, &mut rng);

            let start = Instant::now();
            let out1 = E::multi_pairing(a1.clone(), b1.clone());
            t1 += start.elapsed().as_micros();

            let start = Instant::now();
            let out2 = E::multi_pairing(a2.clone(), b2.clone());
            t1 += start.elapsed().as_micros();

            let start = Instant::now();
            let out3 = E::multi_pairing(a3.clone(), b3.clone());
            t1 += start.elapsed().as_micros();

            println!("[{curve}] Time taken without checker {} us", t1);

            for lazy in [true, false] {
                let start = Instant::now();
                let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                    .with_err((), |checker| {
                        checker.add_multiple_sources_and_target(&a1, b1.iter().copied(), &out1);
                        checker.add_multiple_sources_and_target(&a2, b2.iter().copied(), &out2);
                        checker.add_multiple_sources_and_target(&a3, b3.iter().copied(), &out3);
                        Ok(())
                    });
                assert!(res.is_ok());
                let l_str = if lazy { "lazy-" } else { "" };
                println!(
                    "[{curve}] Time taken with {}checker {} us",
                    l_str,
                    start.elapsed().as_micros()
                );

                // Fail on wrong output
                let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                    .with_err((), |checker| {
                        checker.add_multiple_sources_and_target(&a1, b1.iter().copied(), &out2);
                        checker.add_multiple_sources_and_target(&a2, b2.iter().copied(), &out1);
                        Ok(())
                    });
                assert!(res.is_err());
            }

            let b1_prep = prepare::<E>(&b1);
            let b2_prep = prepare::<E>(&b2);
            let b3_prep = prepare::<E>(&b3);

            for lazy in [true, false] {
                let start = Instant::now();
                let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                    .with_err((), |checker| {
                        checker.add_multiple_sources_and_target(&a1, b1_prep.clone(), &out1);
                        checker.add_multiple_sources_and_target(&a2, b2_prep.clone(), &out2);
                        checker.add_multiple_sources_and_target(&a3, b3_prep.clone(), &out3);
                        Ok(())
                    });
                assert!(res.is_ok());
                let l_str = if lazy { "lazy-" } else { "" };
                println!(
                    "[{curve}] Time taken with prepared G2 and {}checker {} us",
                    l_str,
                    start.elapsed().as_micros()
                );
            }

            let a1_rev = rev_vec(&a1);
            let a2_rev = rev_vec(&a2);
            let a3_rev = rev_vec(&a3);
            let b1_rev = rev_vec(&b1);
            let b2_rev = rev_vec(&b2);
            let b3_rev = rev_vec(&b3);

            let b1_rev_prep = prepare::<E>(&b1_rev);
            let b2_rev_prep = prepare::<E>(&b2_rev);
            let b3_rev_prep = prepare::<E>(&b3_rev);

            for lazy in [true, false] {
                let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                    .with_err((), |checker| {
                        checker.add_multiple_sources(
                            &a1,
                            b1.iter().copied(),
                            &a1_rev,
                            b1_rev.iter().copied(),
                        );
                        Ok(())
                    });
                assert!(res.is_ok());
            }

            for lazy in [true, false] {
                let start = Instant::now();
                let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                    .with_err((), |checker| {
                        checker.add_multiple_sources(
                            &a1,
                            b1.iter().copied(),
                            &a1_rev,
                            b1_rev.iter().copied(),
                        );
                        checker.add_multiple_sources(
                            &a2,
                            b2.iter().copied(),
                            &a2_rev,
                            b2_rev.iter().copied(),
                        );
                        checker.add_multiple_sources(
                            &a3,
                            b3.iter().copied(),
                            &a3_rev,
                            b3_rev.iter().copied(),
                        );
                        Ok(())
                    });
                assert!(res.is_ok());
                let l_str = if lazy { "lazy-" } else { "" };
                println!(
                    "[{curve}] Time taken with {}checker {} us",
                    l_str,
                    start.elapsed().as_micros()
                );
            }

            for lazy in [true, false] {
                let start = Instant::now();
                let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                    .with_err((), |checker| {
                        checker.add_multiple_sources(
                            &a1,
                            b1_prep.clone(),
                            &a1_rev,
                            b1_rev_prep.clone(),
                        );
                        checker.add_multiple_sources(
                            &a2,
                            b2_prep.clone(),
                            &a2_rev,
                            b2_rev_prep.clone(),
                        );
                        checker.add_multiple_sources(
                            &a3,
                            b3_prep.clone(),
                            &a3_rev,
                            b3_rev_prep.clone(),
                        );
                        Ok(())
                    });
                assert!(res.is_ok());
                let l_str = if lazy { "lazy-" } else { "" };
                println!(
                    "[{curve}] Time taken with prepared G2 and {}checker {} us",
                    l_str,
                    start.elapsed().as_micros()
                );
            }

            for lazy in [true, false] {
                let start = Instant::now();
                let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                    .with_err((), |checker| {
                        checker.add_multiple_sources_and_target(&a1, b1.iter().copied(), &out1);
                        checker.add_multiple_sources_and_target(&a2, b2.iter().copied(), &out2);
                        checker.add_multiple_sources_and_target(&a3, b3.iter().copied(), &out3);
                        checker.add_multiple_sources(
                            &a1,
                            b1.iter().copied(),
                            &a1_rev,
                            b1_rev.iter().copied(),
                        );
                        checker.add_multiple_sources(
                            &a2,
                            b2.iter().copied(),
                            &a2_rev,
                            b2_rev.iter().copied(),
                        );
                        checker.add_multiple_sources(
                            &a3,
                            b3.iter().copied(),
                            &a3_rev,
                            b3_rev.iter().copied(),
                        );
                        Ok(())
                    });
                assert!(res.is_ok());
                let l_str = if lazy { "lazy-" } else { "" };
                println!(
                    "[{curve}] Time taken with {}checker {} us",
                    l_str,
                    start.elapsed().as_micros()
                );
            }

            for lazy in [true, false] {
                let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                    .with_err((), |checker| {
                        checker.add_sources(&a1[0], b1[0], &a1[0], b1[0]);
                        Ok(())
                    });
                assert!(res.is_ok());

                let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                    .with_err((), |checker| {
                        checker.add_sources(&a1[0], b1[0], &a1[0], b1[0]);
                        checker.add_sources(&a1[1], b1[1], &a1[1], b1[1]);
                        checker.add_sources(&a1[2], b1[2], &a1[2], b1[2]);
                        Ok(())
                    });
                assert!(res.is_ok());
            }

            for lazy in [true, false] {
                let out_0 = E::pairing(a1[0], b1[0]);
                let out_1 = E::pairing(a1[1], b1[1]);
                let out_2 = E::pairing(a1[2], b1[2]);

                let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                    .with_err((), |checker| {
                        checker.add_sources_and_target(&a1[0], b1[0], &out_0);
                        Ok(())
                    });
                assert!(res.is_ok());

                let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                    .with_err((), |checker| {
                        checker.add_sources_and_target(&a1[0], b1[0], &out_0);
                        checker.add_sources_and_target(&a1[1], b1[1], &out_1);
                        checker.add_sources_and_target(&a1[2], b1[2], &out_2);
                        Ok(())
                    });
                assert!(res.is_ok());

                // Fail on wrong output
                let wrong_out = E::pairing(a1[0], b1[1]);
                let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                    .with_err((), |checker| {
                        checker.add_sources_and_target(&a1[0], b1[0], &wrong_out);
                        Ok(())
                    });
                assert!(res.is_err());
            }

            // Boundary cases with `TARGET_MSM_THRESHOLD` - 1, `TARGET_MSM_THRESHOLD`, and
            // `TARGET_MSM_THRESHOLD` + 1 distinct targets for target-group MSM
            let at = rand_g1::<E>(TARGET_MSM_THRESHOLD + 1, &mut rng);
            let bt = rand_g2::<E>(TARGET_MSM_THRESHOLD + 1, &mut rng);
            let out_targets: Vec<_> = (0..TARGET_MSM_THRESHOLD + 1).map(|i| E::pairing(at[i], bt[i])).collect();

            for count in [TARGET_MSM_THRESHOLD - 1, TARGET_MSM_THRESHOLD, TARGET_MSM_THRESHOLD + 1] {
                for lazy in [true, false] {
                    // Valid equations pass using non-unit multipliers
                    let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                        .with_err((), |checker| {
                            for i in 0..count {
                                checker.add_sources_and_target(&at[i], bt[i], &out_targets[i]);
                            }
                            assert_eq!(checker.num_targets(), count);
                            Ok(())
                        });
                    assert!(res.is_ok());

                    // Corrupting one target fails
                    for corrupt_idx in [0, count - 1] {
                        let wrong_out = out_targets[corrupt_idx] + out_targets[corrupt_idx];
                        let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                            .with_err((), |checker| {
                                for i in 0..count {
                                    let target = if i == corrupt_idx {
                                        &wrong_out
                                    } else {
                                        &out_targets[i]
                                    };
                                    checker.add_sources_and_target(&at[i], bt[i], target);
                                }
                                assert_eq!(checker.num_targets(), count);
                                Ok(())
                            });
                        assert!(res.is_err());
                    }
                }
            }

            // Repeating the same target
            for lazy in [true, false] {
                let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                    .with_err((), |checker| {
                        for _ in 0..(TARGET_MSM_THRESHOLD + 1) {
                            checker.add_sources_and_target(&at[0], bt[0], &out_targets[0]);
                        }
                        assert_eq!(checker.num_targets(), 1);
                        assert!(checker.num_targets() < TARGET_MSM_THRESHOLD);
                        Ok(())
                    });
                res.unwrap();

                // Corrupting one target when repeating also fails
                let wrong_out = out_targets[0] + out_targets[0];
                let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, lazy)
                    .with_err((), |checker| {
                        for i in 0..(TARGET_MSM_THRESHOLD + 1) {
                            let target = if i == TARGET_MSM_THRESHOLD {
                                &wrong_out
                            } else {
                                &out_targets[0]
                            };
                            checker.add_sources_and_target(&at[0], bt[0], target);
                        }
                        assert_eq!(checker.num_targets(), 2);
                        assert!(checker.num_targets() < TARGET_MSM_THRESHOLD);
                        Ok(())
                    });
                assert!(res.is_err());
            }
        }

        check::<Bls12_381>("BLS12-381");
        check::<Bn254>("BN254");
    }

    #[test]
    fn grouping_by_g2() {
        fn check<E: Pairing>(curve: &str) {
            let mut rng = StdRng::seed_from_u64(0u64);
            let n = 16;

            let g2 = E::G2Affine::rand(&mut rng);
            let sk = E::ScalarField::rand(&mut rng);
            let pk = (g2 * sk).into_affine();
            let pk_prep = E::G2Prepared::from(pk);
            let g2_prep = E::G2Prepared::from(g2);

            // e(A', pk) = e(A_bar, g2) with A_bar = A' * sk
            let a_prime = rand_g1::<E>(n, &mut rng);
            let a_bar =
                E::G1::normalize_batch(&a_prime.iter().map(|a| *a * sk).collect::<Vec<_>>());

            let start = Instant::now();
            let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, true).with_err(
                (),
                |checker| {
                    for i in 0..n {
                        checker.add_sources(
                            &a_prime[i],
                            pk_prep.clone(),
                            &a_bar[i],
                            g2_prep.clone(),
                        );
                    }
                    assert_eq!(checker.num_groups(), 2);
                    Ok(())
                },
            );
            assert!(res.is_ok());
            println!(
                "[{curve}] {} equations with prepared G2 took {:?}",
                n,
                start.elapsed()
            );

            let start = Instant::now();
            let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, true).with_err(
                (),
                |checker| {
                    for i in 0..n {
                        checker.add_sources_g2_affine(&a_prime[i], &pk, &a_bar[i], &g2);
                    }
                    assert_eq!(checker.num_groups(), 2);
                    Ok(())
                },
            );
            assert!(res.is_ok());
            println!(
                "[{curve}] {} equations with affine G2 took {:?}",
                n,
                start.elapsed()
            );

            // Fail when one of the equations does not hold
            let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, true).with_err(
                (),
                |checker| {
                    for i in 0..n {
                        checker.add_sources_g2_affine(
                            &a_prime[i],
                            &pk,
                            &a_bar[(i + 1) % n],
                            &g2,
                        );
                    }
                    Ok(())
                },
            );
            assert!(res.is_err());

            // The two forms of the same G2 do not share a group but the check still holds
            let res = RandomizedPairingCheckerGuard::<E>::new_using_rng(&mut rng, true).with_err(
                (),
                |checker| {
                    for i in 0..n {
                        if i % 2 == 0 {
                            checker.add_sources(
                                &a_prime[i],
                                pk_prep.clone(),
                                &a_bar[i],
                                g2_prep.clone(),
                            );
                        } else {
                            checker.add_sources_g2_affine(&a_prime[i], &pk, &a_bar[i], &g2);
                        }
                    }
                    assert_eq!(checker.num_groups(), 4);
                    Ok(())
                },
            );
            assert!(res.is_ok());
        }

        check::<Bls12_381>("BLS12-381");
        check::<Bn254>("BN254");
    }

    /// A cancelled checker neither verifies nor panics when dropped
    #[test]
    fn cancelled() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let a = <Bls12_381 as Pairing>::G1Affine::rand(&mut rng);
        let b = <Bls12_381 as Pairing>::G2Affine::rand(&mut rng);
        let out = Bls12_381::pairing(a, b);
        let wrong_out = out + out;

        // The closure's error cancels the checker, so the failing check it holds is never verified
        let res: Result<(), &str> =
            Guard::new_using_rng(&mut rng, true).with_err("failed", |checker| {
                checker.add_sources_and_target(&a, b, &wrong_out);
                Err("failed")
            });
        assert_eq!(res, Err("failed"));
    }

    /// A panic inside the closure drops the unverified, failing checker during unwinding without
    /// a second panic
    #[test]
    #[should_panic(expected = "closure panicked")]
    fn panic_in_closure() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let a = <Bls12_381 as Pairing>::G1Affine::rand(&mut rng);
        let b = <Bls12_381 as Pairing>::G2Affine::rand(&mut rng);
        let out = Bls12_381::pairing(a, b);
        let wrong_out = out + out;

        let _ = Guard::new_using_rng(&mut rng, true).with_err((), |checker| {
            checker.add_sources_and_target(&a, b, &wrong_out);
            panic!("closure panicked");
            #[allow(unreachable_code)]
            Ok(())
        });
    }

    #[test]
    fn negated_g2() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let a1 = <Bls12_381 as Pairing>::G1Affine::rand(&mut rng);
        let a2 = <Bls12_381 as Pairing>::G1Affine::rand(&mut rng);
        let b = <Bls12_381 as Pairing>::G2Affine::rand(&mut rng);
        let minus_b = -b;
        let out1 = Bls12_381::pairing(a1, b);
        let out2 = Bls12_381::pairing(a2, minus_b);

        for lazy in [true, false] {
            let res = Guard::new_using_rng(&mut rng, lazy).with_err((), |checker| {
                checker.add_sources_and_target_g2_affine(&a1, &b, &out1);
                checker.add_sources_and_target_g2_affine(&a2, &minus_b, &out2);
                if lazy {
                    assert_eq!(checker.num_groups(), 1);
                }
                Ok(())
            });
            assert!(res.is_ok());
        }

        // Fail when the sign of the G2 is flipped
        let res = Guard::new_using_rng(&mut rng, true).with_err((), |checker| {
            checker.add_sources_and_target_g2_affine(&a1, &minus_b, &out1);
            checker.add_sources_and_target_g2_affine(&a2, &minus_b, &out2);
            Ok(())
        });
        assert!(res.is_err());
    }

    #[test]
    fn repeated_and_identity_targets() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let n = 12;
        let a = rand_g1::<Bls12_381>(n, &mut rng);
        let b = rand_g2::<Bls12_381>(n, &mut rng);
        // Same target for every equation
        let out = Bls12_381::pairing(a[0], b[0]);
        for lazy in [true, false] {
            let res = Guard::new_using_rng(&mut rng, lazy).with_err((), |checker| {
                for _ in 0..n {
                    checker.add_sources_and_target_g2_affine(&a[0], &b[0], &out);
                }
                Ok(())
            });
            assert!(res.is_ok());
        }

        // Identity target, i.e. `e(a, b) + e(-a, b) = 0`
        let minus_a = -a[0];
        for lazy in [true, false] {
            let res = Guard::new_using_rng(&mut rng, lazy).with_err((), |checker| {
                checker.add_multiple_sources_and_target_g2_affine(
                    &[a[0], minus_a],
                    &[b[0], b[0]],
                    &PairingOutput::zero(),
                );
                Ok(())
            });
            assert!(res.is_ok());
        }
    }

    #[test]
    fn zero_inputs() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let a = <Bls12_381 as Pairing>::G1Affine::rand(&mut rng);
        let b = <Bls12_381 as Pairing>::G2Affine::rand(&mut rng);
        let out = Bls12_381::pairing(a, b);
        let zero_g1 = <Bls12_381 as Pairing>::G1Affine::zero();
        let zero_g2 = <Bls12_381 as Pairing>::G2Affine::zero();

        for lazy in [true, false] {
            let res = Guard::new_using_rng(&mut rng, lazy).with_err((), |checker| {
                checker.add_sources_and_target_g2_affine(&zero_g1, &b, &PairingOutput::zero());
                checker.add_sources_and_target_g2_affine(&a, &zero_g2, &PairingOutput::zero());
                checker.add_sources_and_target_g2_affine(&a, &b, &out);
                Ok(())
            });
            assert!(res.is_ok());
        }
    }

    #[test]
    #[should_panic]
    fn safety() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let n = 2;

        let a1 = rand_g1::<Bls12_381>(n, &mut rng);
        let b1 = rand_g2::<Bls12_381>(n, &mut rng);
        let a2 = rand_g1::<Bls12_381>(n + 5, &mut rng);
        let b2 = rand_g2::<Bls12_381>(n + 5, &mut rng);

        let out1 = Bls12_381::multi_pairing(a1.clone(), b1.clone());
        let out2 = Bls12_381::multi_pairing(a2.clone(), b2.clone());

        // The verification should work
        let res = Guard::new_using_rng(&mut rng, true).with_err((), |checker| {
            checker.add_multiple_sources_and_target(&a1, b1.iter().copied(), &out1);
            checker.add_multiple_sources_and_target(&a2, b2.iter().copied(), &out2);
            Ok(())
        });
        assert!(res.is_ok());

        // This should panic since a checker holding a failing check is dropped without calling `verify()`
        #[allow(deprecated)]
        let mut checker = RandomizedPairingChecker::<Bls12_381>::new_using_rng(&mut rng, true);
        checker.add_multiple_sources_and_target(&a1, b1.iter().copied(), &out2);
        checker.add_multiple_sources_and_target(&a2, b2.iter().copied(), &out1);
    }
}

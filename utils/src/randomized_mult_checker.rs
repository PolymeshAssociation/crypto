#[cfg(feature = "ahash")]
use ahash::RandomState;
use ark_ec::{AffineRepr, VariableBaseMSM};
use ark_ff::{One, Zero};
use ark_std::{
    iter::{IntoIterator, Iterator},
    rand::Rng,
    vec::Vec,
    UniformRand,
};
use hashbrown::{hash_map::Entry, HashMap};

use crate::error::UtilsError;

/// A guard for `RandomizedMultChecker` that ensures `verify()` is called at the end of the scope.
///
/// The guard pattern is used to ensure that `verify()` is always called, even if the caller forgets to call it explicitly.
/// The `with` and `with_err` methods take a closure that performs the checks and returns a result.
/// If the closure returns an error, the checker is cancelled and the error is returned.
/// If the closure returns `Ok`, the checker is verified and the result is returned.
/// This pattern ensures that `verify()` is always called, and if it fails, it will return an error.
///
/// Example usage:
/// ```rust
/// let mut rng = StdRng::seed_from_u64(0u64);
/// let checker = RandomizedMultCheckerGuard::new_using_rng(&mut rng);
/// checker.with(|checker| {
///     proof.verify(checker)?;
///     Ok(())
/// })?;
/// ```
#[derive(Debug)]
pub struct RandomizedMultCheckerGuard<G: AffineRepr> {
    inner: RandomizedMultChecker<G>,
}

impl<G: AffineRepr> RandomizedMultCheckerGuard<G> {
    /// Create a new `RandomizedMultCheckerGuard` with the given random value.
    pub fn new(random: G::ScalarField) -> Self {
        Self {
            inner: RandomizedMultChecker::_new(random),
        }
    }

    /// Create a new `RandomizedMultCheckerGuard` with a random value generated using the given RNG.
    pub fn new_using_rng<R: Rng>(rng: &mut R) -> Self {
        Self::new(G::ScalarField::rand(rng))
    }

    /// Run the given closure with the inner `RandomizedMultChecker`.
    pub fn with<O, E: From<UtilsError>>(
        mut self,
        f: impl FnOnce(&mut RandomizedMultChecker<G>) -> Result<O, E>,
    ) -> Result<O, E> {
        match f(&mut self.inner) {
            Ok(result) => {
                self.inner.verify()?;
                Ok(result)
            }
            Err(err) => {
                self.inner.cancel();
                Err(err)
            }
        }
    }

    /// Run the given closure with the inner `RandomizedMultChecker`, and return the given error if verification fails.
    pub fn with_err<O, E>(
        mut self,
        err: E,
        f: impl FnOnce(&mut RandomizedMultChecker<G>) -> Result<O, E>,
    ) -> Result<O, E> {
        match f(&mut self.inner) {
            Ok(result) => {
                self.inner.verify().map_err(|_| err)?;
                Ok(result)
            }
            Err(err) => {
                self.inner.cancel();
                Err(err)
            }
        }
    }
}

/// A guard for a pair of `RandomizedMultChecker` that ensures `verify()` is called for both checkers at the end of the scope.
///
/// This is useful when there are two separate multi-scalar multiplication checks that need to be performed together, say one for G1 and one for G2 in a pairing check.
/// The `with` and `with_err` methods take a closure that performs the checks and returns a result.
/// If the closure returns an error, both checkers are cancelled and the error is returned.
/// If the closure returns `Ok`, both checkers are verified and the result is returned.
/// This pattern ensures that `verify()` is always called for both checkers, and if it fails, it will return an error.
///
/// Example usage:
/// ```rust
/// let mut rng = StdRng::seed_from_u64(0u64);
/// let checker = PairRandomizedMultCheckerGuard::new_using_rng(&mut rng);
/// checker.with(|(checker0, checker1)| {
///     proof.verify(checker0, checker1)?;
///     Ok(())
/// })?;
/// ```
#[derive(Debug)]
pub struct PairRandomizedMultCheckerGuard<G0: AffineRepr, G1: AffineRepr> {
    inner0: RandomizedMultChecker<G0>,
    inner1: RandomizedMultChecker<G1>,
    parallel: bool,
}

impl<G0: AffineRepr, G1: AffineRepr> PairRandomizedMultCheckerGuard<G0, G1> {
    /// Create a new `PairRandomizedMultCheckerGuard` with the given random values.
    pub fn new(random0: G0::ScalarField, random1: G1::ScalarField) -> Self {
        Self {
            inner0: RandomizedMultChecker::_new(random0),
            inner1: RandomizedMultChecker::_new(random1),
            parallel: cfg!(feature = "parallel"),
        }
    }

    /// Create a new `PairRandomizedMultCheckerGuard` with the random values generated using the given RNG.
    pub fn new_using_rng<R: Rng>(rng: &mut R) -> Self {
        Self::new(G0::ScalarField::rand(rng), G1::ScalarField::rand(rng))
    }

    /// Set whether to verify the two checkers in parallel. By default, it is set to true if the "parallel" feature is enabled.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Run the given closure with the pair of inner `RandomizedMultChecker`.
    pub fn with<O, E: From<UtilsError>>(
        mut self,
        f: impl FnOnce(&mut RandomizedMultChecker<G0>, &mut RandomizedMultChecker<G1>) -> Result<O, E>,
    ) -> Result<O, E> {
        match f(&mut self.inner0, &mut self.inner1) {
            Ok(result) => {
                self.verify()?;
                Ok(result)
            }
            Err(err) => {
                self.inner0.cancel();
                self.inner1.cancel();
                Err(err)
            }
        }
    }

    /// Run the given closure with the pair of inner `RandomizedMultChecker`, and return the given error if verification fails.
    pub fn with_err<O, E: Clone>(
        mut self,
        err: E,
        f: impl FnOnce(&mut RandomizedMultChecker<G0>, &mut RandomizedMultChecker<G1>) -> Result<O, E>,
    ) -> Result<O, E> {
        match f(&mut self.inner0, &mut self.inner1) {
            Ok(result) => {
                self.verify().map_err(|_| err)?;
                Ok(result)
            }
            Err(err) => {
                self.inner0.cancel();
                self.inner1.cancel();
                Err(err)
            }
        }
    }

    fn verify(self) -> Result<(), UtilsError> {
        let inner0 = self.inner0;
        let mut inner1 = self.inner1;
        let _parallel = self.parallel;

        #[cfg(feature = "parallel")]
        if _parallel {
            let (even_verify, odd_verify) = rayon::join(|| inner0.verify(), || inner1.verify());
            even_verify?;
            odd_verify?;
            return Ok(());
        }

        match inner0.verify() {
            Ok(_) => {}
            Err(err) => {
                inner1.cancel();
                return Err(err);
            }
        }
        inner1.verify()?;

        Ok(())
    }
}

/// Represents a scalar multiplication check of the form `G1 * a1 + G2 * a2 + G3 * a3 + ... = T`.
/// Several checks can be added of forms either `G1 * a1 = T1` or `G1 * a1 + H1 * b1 = T2` or `G1 * a1 + H1 * b1 + J1 * c1 = T3`
/// These checks can be aggregated together using random linear combination. The efficiency comes from converting all these
/// scalar multiplications in a single multi-scalar multiplication.
/// For each check, multiply the check by a power of a random element created during initialization.
/// eg. for these 4 checks `G1 * a1 = T1, G1 * a2 + H1 * b2 = T2` and `G1 * a3 + H2 * b3 + J1 * c3 = T3`, `G1 * a4 + H2 * b4 + J2 * c4 = T4`,
/// a single check is created as `G1 * a1 - T1 + G1 * a2 * r + H1 * b2 * r - T2 * r + G1 * a3 * r^2 + H2 * b3 * r^2 + J1 * c3 * r^2 - T3 * r^2 + G1 * a4 * r^3 + H2 * b4 * r^3 + J2 * c4 * r^3 - T4 * r^3 = 0`
/// where `r` is a random value and so are`r^2`, `r^3`
/// The single check above is simplified by combining terms of `G1`, `H1`, etc to reduce the size of the multi-scalar multiplication
#[derive(Debug)]
pub struct RandomizedMultChecker<G: AffineRepr> {
    /// Verification will expect the multi-scalar multiplication of key-value pairs to be one.
    /// x-coordinate -> (scalar, point, y-coordinate)
    // This trick is taken from halo2 code (MSM) but keeping both the point and y coordinate in value since there is no way to convert back from x, y coordinates for AffineRepr
    #[cfg(feature = "ahash")]
    pub args: HashMap<G::BaseField, (G::ScalarField, G, G::BaseField), RandomState>,
    #[cfg(not(feature = "ahash"))]
    pub args: HashMap<G::BaseField, (G::ScalarField, G, G::BaseField)>,
    /// The random value chosen during creation
    pub random: G::ScalarField,
    /// The random value to be used for current check. After each check, set `current_random = current_random * random`
    pub current_random: G::ScalarField,
    /// Flag to detect forgotten `verify()`.
    verified: bool,
    /// Flag to detect a canceled checker.
    cancelled: bool,
}

impl<G: AffineRepr> RandomizedMultChecker<G> {
    fn _new(random: G::ScalarField) -> Self {
        Self {
            #[cfg(feature = "ahash")]
            args: HashMap::with_hasher(RandomState::new()),
            #[cfg(not(feature = "ahash"))]
            args: HashMap::new(),
            random,
            current_random: G::ScalarField::one(),
            verified: false,
            cancelled: false,
        }
    }

    #[deprecated = "Use `RandomizedMultCheckerGuard::new` or `RandomizedMultCheckerGuard::new_using_rng` instead"]
    pub fn new(random: G::ScalarField) -> Self {
        Self::_new(random)
    }

    #[deprecated = "Use `RandomizedMultCheckerGuard::new` or `RandomizedMultCheckerGuard::new_using_rng` instead"]
    pub fn new_using_rng<R: Rng>(rng: &mut R) -> Self {
        Self::_new(G::ScalarField::rand(rng))
    }

    /// Add a check of the form `p * s = t`. Converts it to `p * s * r - t * r = 0` where `r` is the current randomness.
    pub fn add_1(&mut self, p: G, s: &G::ScalarField, t: G) {
        if self.cancelled {
            return;
        }
        self.add(p, self.current_random * s);
        self.add(t, -self.current_random);
        self.update_random();
    }

    /// Add a check of the form `p1 * s1 + p2 * s2 = t`. Converts it to `p1 * s1 * r + p2 * s2 * r - t * r = 0` where `r` is the current randomness.
    pub fn add_2(&mut self, p1: G, s1: &G::ScalarField, p2: G, s2: &G::ScalarField, t: G) {
        if self.cancelled {
            return;
        }
        self.add(p1, self.current_random * s1);
        self.add(p2, self.current_random * s2);
        self.add(t, -self.current_random);
        self.update_random();
    }

    /// Add a check of the form `p1 * s1 + p2 * s2 + p3 * s3 = t`. Converts it to `p1 * s1 * r + p2 * s2 * r + p3 * s3 * r - t * r = 0` where `r` is the current randomness.
    pub fn add_3(
        &mut self,
        p1: G,
        s1: &G::ScalarField,
        p2: G,
        s2: &G::ScalarField,
        p3: G,
        s3: &G::ScalarField,
        t: G,
    ) {
        if self.cancelled {
            return;
        }
        self.add(p1, self.current_random * s1);
        self.add(p2, self.current_random * s2);
        self.add(p3, self.current_random * s3);
        self.add(t, -self.current_random);
        self.update_random();
    }

    /// Add a check of the form `<a, b> = t`. Expects `a` and `b` to be of the same length
    pub fn add_many<'a>(
        &mut self,
        a: impl IntoIterator<Item = G>,
        b: impl IntoIterator<Item = &'a G::ScalarField>,
        t: G,
    ) {
        if self.cancelled {
            return;
        }
        for (a_i, b_i) in a.into_iter().zip(b) {
            self.add(a_i, self.current_random * b_i);
        }
        self.add(t, -self.current_random);
        self.update_random();
    }

    /// Combine all the checks into a multi-scalar multiplication and return `Ok` if the result is 0.
    pub fn verify(mut self) -> Result<(), UtilsError> {
        self.do_verify()
    }

    fn do_verify(&mut self) -> Result<(), UtilsError> {
        if self.cancelled {
            return Err(UtilsError::MultCheckFailed);
        }
        self.verified = true;
        if self.len() == 0 {
            return Ok(());
        }
        let (points, scalars) = self.points_and_scalars_for_msm();
        if G::Group::msm_unchecked(&points, &scalars).is_zero() {
            Ok(())
        } else {
            Err(UtilsError::MultCheckFailed)
        }
    }

    /// Cancel the checker. This is useful when the verifier wants to cancel the checker if it is not needed anymore,
    /// say a different check failed which fails verification anyway
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    pub fn len(&self) -> usize {
        self.args.len()
    }

    pub fn points_and_scalars_for_msm(&self) -> (Vec<G>, Vec<G::ScalarField>) {
        let mut points = Vec::with_capacity(self.len());
        let mut scalars = Vec::with_capacity(self.len());
        for (_, (s, point, _)) in self.args.iter() {
            points.push(*point);
            scalars.push(*s);
        }
        (points, scalars)
    }

    pub fn update_random(&mut self) {
        self.current_random *= self.random;
    }

    #[inline(always)]
    pub fn add(&mut self, p: G, s: G::ScalarField) {
        if self.cancelled {
            return;
        }
        if p.is_zero() {
            return;
        }

        // unwrap is fine as point is not at infinity
        let (p_x, p_y) = p.xy().unwrap();

        match self.args.entry(p_x) {
            Entry::Occupied(mut entry) => {
                let (old_scalar, point, y) = entry.get_mut();
                if *y == p_y {
                    *old_scalar += s;
                } else {
                    *old_scalar -= s;
                    debug_assert_eq!(point.into_group(), -p.into_group());
                }
            }
            Entry::Vacant(entry) => {
                entry.insert((s, p, p_y));
            }
        }
    }
}

impl<G: AffineRepr> Drop for RandomizedMultChecker<G> {
    fn drop(&mut self) {
        if self.cancelled || self.verified {
            return;
        }
        // Only panic if verify fails.
        if let Err(err) = self.do_verify() {
            log::error!("Skipped `verify` call returns error: err={err:?}");
            // Panic as this code path should never be reached in production and be caught in testing
            panic!(
                "RandomizedMultChecker was dropped without calling `verify()`. \
                This means all accumulated scalar multiplications were never performed. \
                This indicates a bug in caller's verifier code"
            );
        } else {
            // Log an error, since the code should be fixed to call `verify`.
            log::error!(
                "RandomizedMultChecker was dropped without calling `verify()`. \
                This means all accumulated scalar multiplications were never performed. \
                This indicates a bug in caller's verifier code"
            );
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bls12_381::{Fr, G1Affine};
    use ark_ec::CurveGroup;
    use ark_std::{
        rand::{rngs::StdRng, SeedableRng},
        UniformRand,
    };
    use std::time::Instant;

    #[test]
    fn basic() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let g1 = G1Affine::rand(&mut rng);
        let g2 = G1Affine::rand(&mut rng);
        let g3 = G1Affine::rand(&mut rng);
        let h1 = G1Affine::rand(&mut rng);
        let h2 = G1Affine::rand(&mut rng);
        let h3 = G1Affine::rand(&mut rng);

        let a1 = Fr::rand(&mut rng);
        let a2 = Fr::rand(&mut rng);
        let a3 = Fr::rand(&mut rng);
        let a4 = Fr::rand(&mut rng);
        let a5 = Fr::rand(&mut rng);
        let a6 = Fr::rand(&mut rng);

        let c1 = (g1 * a1).into_affine();
        let c2 = (g1 * a2).into_affine();
        let c3 = (g1 * a3).into_affine();

        let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
            checker.add_1(g1, &a1, c1);
            checker.add_1(g1, &a2, c2);
            checker.add_1(g1, &a3, c3);
            Ok(())
        });
        assert!(res.is_ok());

        // Checking if g1 * a2 == c3 fails
        let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
            checker.add_1(g1, &a1, c1);
            checker.add_1(g1, &a2, c2); // this is invalid
            checker.add_1(g1, &a2, c3);
            Ok(())
        });
        assert!(res.is_err());

        let c1 = (g1 * a1).into_affine();
        let c2 = (g2 * a2).into_affine();
        let c3 = (g3 * a3).into_affine();

        let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
            checker.add_1(g1, &a1, c1);
            checker.add_1(g2, &a2, c2);
            checker.add_1(g3, &a3, c3);
            Ok(())
        });
        assert!(res.is_ok());

        // Checking if g2 * a3 == c3 fails
        let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
            checker.add_1(g1, &a1, c1);
            checker.add_1(g2, &a2, c2); // this is invalid
            checker.add_1(g2, &a3, c3);
            Ok(())
        });
        assert!(res.is_err());

        let c1 = (g1 * a1 + h1 * a4).into_affine();
        let c2 = (g1 * a2 + h1 * a5).into_affine();
        let c3 = (g1 * a3 + h1 * a6).into_affine();

        let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
            checker.add_2(g1, &a1, h1, &a4, c1);
            checker.add_2(g1, &a2, h1, &a5, c2);
            checker.add_2(g1, &a3, h1, &a6, c3);
            Ok(())
        });
        assert!(res.is_ok());

        // Checking if g1 * a3 + h1 * a3 == c3 fails
        let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
            checker.add_2(g1, &a1, h1, &a4, c1);
            checker.add_2(g1, &a2, h1, &a5, c2);
            checker.add_2(g1, &a3, h1, &a3, c3); // this is invalid
            Ok(())
        });
        assert!(res.is_err());

        let c1 = (g1 * a1 + h1 * a4).into_affine();
        let c2 = (g2 * a2 + h2 * a5).into_affine();
        let c3 = (g3 * a3 + h3 * a6).into_affine();

        let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
            checker.add_2(g1, &a1, h1, &a4, c1);
            checker.add_2(g2, &a2, h2, &a5, c2);
            checker.add_2(g3, &a3, h3, &a6, c3);
            Ok(())
        });
        assert!(res.is_ok());

        let c1 = (g1 * a1 + g2 * a2 + g3 * a3).into_affine();
        let c2 = (h1 * a4 + h2 * a5 + h3 * a6).into_affine();
        let c3 = (g2 * a3 + h1 * a1 + h2 * a2).into_affine();

        let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
            checker.add_3(g1, &a1, g2, &a2, g3, &a3, c1);
            checker.add_3(h1, &a4, h2, &a5, h3, &a6, c2);
            checker.add_3(g2, &a3, h1, &a1, h2, &a2, c3);
            Ok(())
        });
        assert!(res.is_ok());

        // Checking if g2 * a3 + h1 * a1 + h2 * a1 == c3 fails
        let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
            checker.add_3(g1, &a1, g2, &a2, g3, &a3, c1);
            checker.add_3(h1, &a4, h2, &a5, h3, &a6, c2);
            checker.add_3(g2, &a3, h1, &a1, h2, &a1, c3); // this is invalid
            Ok(())
        });
        assert!(res.is_err());

        let c1 = (g1 * a1).into_affine();
        let c2 = (g2 * a2).into_affine();
        let c3 = (g1 * a1 + h1 * a4).into_affine();
        let c4 = (g1 * a2 + h1 * a5).into_affine();
        let c5 = (g1 * a3 + h1 * a6).into_affine();
        let c6 = (g1 * a1 + h1 * a4).into_affine();
        let c7 = (g2 * a2 + h2 * a5).into_affine();
        let c8 = (g3 * a3 + h3 * a6).into_affine();
        let c9 = (g1 * a1 + g2 * a2 + g3 * a3).into_affine();
        let c10 = (h1 * a4 + h2 * a5 + h3 * a6).into_affine();
        let c11 = (h1 * a2 + h2 * a3 + h3 * a4).into_affine();

        let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
            checker.add_1(g1, &a1, c1);
            checker.add_1(g2, &a2, c2);
            checker.add_2(g1, &a1, h1, &a4, c3);
            checker.add_2(g1, &a2, h1, &a5, c4);
            checker.add_2(g1, &a3, h1, &a6, c5);
            checker.add_2(g1, &a1, h1, &a4, c6);
            checker.add_2(g2, &a2, h2, &a5, c7);
            checker.add_2(g3, &a3, h3, &a6, c8);
            checker.add_3(g1, &a1, g2, &a2, g3, &a3, c9);
            checker.add_3(h1, &a4, h2, &a5, h3, &a6, c10);
            checker.add_3(h1, &a2, h2, &a3, h3, &a4, c11);
            Ok(())
        });
        assert!(res.is_ok());

        let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
            checker.add_many([g3, h3], [&a3, &a6], c8);
            checker.add_many([g1, g2, g3], [&a1, &a2, &a3], c9);
            Ok(())
        });
        assert!(res.is_ok());

        let minus_g1 = -g1;
        let minus_g2 = -g2;
        let c1 = (g1 * a1).into_affine();
        let c2 = (minus_g1 * a2).into_affine();
        let c3 = (g2 * a3).into_affine();
        let c4 = (minus_g2 * a4).into_affine();

        let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
            checker.add_1(g1, &a1, c1);
            checker.add_1(minus_g1, &a2, c2);
            checker.add_1(g2, &a3, c3);
            checker.add_1(minus_g2, &a4, c4);
            Ok(())
        });
        assert!(res.is_ok());
    }

    /// Test the paired checkers together
    #[test]
    fn pair_checker() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let g1 = G1Affine::rand(&mut rng);
        let h1 = G1Affine::rand(&mut rng);
        let a1 = Fr::rand(&mut rng);
        let b1 = Fr::rand(&mut rng);
        let c1 = (g1 * a1 + h1 * b1).into_affine();

        // Both checkers should pass
        let res = PairRandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err(
            (),
            |checker0, checker1| {
                checker0.add_2(g1, &a1, h1, &b1, c1);
                checker1.add_2(h1, &b1, g1, &a1, c1);
                Ok(())
            },
        );
        assert!(res.is_ok());

        // The first checker fails, the second checker should be cancelled and not panic in its Drop impl
        let res = PairRandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err(
            (),
            |checker0, checker1| {
                checker0.add_2(h1, &a1, h1, &b1, c1); // this is invalid
                checker1.add_2(g1, &b1, g1, &a1, c1); // this is also invalid but should be cancelled and not panic in Drop
                Ok(())
            },
        );
        assert!(res.is_err());

        // The first checker is valid, but the second checker fails.
        let res = PairRandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err(
            (),
            |checker0, checker1| {
                checker0.add_2(g1, &a1, h1, &b1, c1); // this is valid
                checker1.add_2(g1, &b1, g1, &a1, c1); // this is invalid
                Ok(())
            },
        );
        assert!(res.is_err());
    }

    #[test]
    fn timing_comparison() {
        let mut rng = StdRng::seed_from_u64(0u64);

        for i in [40, 60, 80, 100] {
            let g = (0..i).map(|_| G1Affine::rand(&mut rng)).collect::<Vec<_>>();
            let h = (0..i).map(|_| G1Affine::rand(&mut rng)).collect::<Vec<_>>();
            let k = (0..i).map(|_| G1Affine::rand(&mut rng)).collect::<Vec<_>>();
            let a = (0..i).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
            let b = (0..i).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
            let c = (0..i).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

            let r = (0..i)
                .map(|j| (g[0] * a[j] + h[0] * b[j]).into_affine())
                .collect::<Vec<_>>();

            let start = Instant::now();
            for j in 0..i {
                assert_eq!((g[0] * a[j] + h[0] * b[j]).into_affine(), r[j]);
            }
            println!("For {} items, naive check took {:?}", i, start.elapsed());

            let start = Instant::now();
            let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
                for j in 0..i {
                    checker.add_2(g[0], &a[j], h[0], &b[j], r[j]);
                }
                Ok(())
            });
            assert!(res.is_ok());
            println!(
                "For {} items, RandomizedMultChecker took {:?}",
                i,
                start.elapsed()
            );

            let r = (0..i)
                .map(|j| (g[j] * a[j] + h[j] * b[j]).into_affine())
                .collect::<Vec<_>>();

            let start = Instant::now();
            for j in 0..i {
                assert_eq!((g[j] * a[j] + h[j] * b[j]).into_affine(), r[j]);
            }
            println!("For {} items, naive check took {:?}", i, start.elapsed());

            let start = Instant::now();
            let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
                for j in 0..i {
                    checker.add_2(g[j], &a[j], h[j], &b[j], r[j]);
                }
                Ok(())
            });
            assert!(res.is_ok());
            println!(
                "For {} items, RandomizedMultChecker took {:?}",
                i,
                start.elapsed()
            );

            let r = (0..i)
                .map(|j| (g[0] * a[j] + h[0] * b[j] + k[0] * c[j]).into_affine())
                .collect::<Vec<_>>();

            let start = Instant::now();
            for j in 0..i {
                assert_eq!(
                    (g[0] * a[j] + h[0] * b[j] + k[0] * c[j]).into_affine(),
                    r[j]
                );
            }
            println!("For {} items, naive check took {:?}", i, start.elapsed());

            let start = Instant::now();
            let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
                for j in 0..i {
                    checker.add_3(g[0], &a[j], h[0], &b[j], k[0], &c[j], r[j]);
                }
                Ok(())
            });
            assert!(res.is_ok());
            println!(
                "For {} items, RandomizedMultChecker took {:?}",
                i,
                start.elapsed()
            );

            let r = (0..i)
                .map(|j| (g[j] * a[j] + h[j] * b[j] + k[j] * c[j]).into_affine())
                .collect::<Vec<_>>();

            let start = Instant::now();
            for j in 0..i {
                assert_eq!(
                    (g[j] * a[j] + h[j] * b[j] + k[j] * c[j]).into_affine(),
                    r[j]
                );
            }
            println!("For {} items, naive check took {:?}", i, start.elapsed());

            let start = Instant::now();
            let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
                for j in 0..i {
                    checker.add_3(g[j], &a[j], h[j], &b[j], k[j], &c[j], r[j]);
                }
                Ok(())
            });
            assert!(res.is_ok());
            println!(
                "For {} items, RandomizedMultChecker took {:?}",
                i,
                start.elapsed()
            );
        }
    }

    #[test]
    #[should_panic]
    fn safety() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let g1 = G1Affine::rand(&mut rng);
        let g2 = G1Affine::rand(&mut rng);
        let g3 = G1Affine::rand(&mut rng);

        let a1 = Fr::rand(&mut rng);
        let a2 = Fr::rand(&mut rng);
        let a3 = Fr::rand(&mut rng);

        let c1 = (g1 * a1).into_affine();
        let c2 = (g2 * a2).into_affine();
        let c3 = (g3 * a3).into_affine();

        // The verification should work
        let res = RandomizedMultCheckerGuard::new_using_rng(&mut rng).with_err((), |checker| {
            checker.add_1(g1, &a1, c1);
            checker.add_1(g2, &a2, c2);
            checker.add_1(g3, &a3, c3);
            Ok(())
        });
        assert!(res.is_ok());

        // This should panic since the checker is dropped without calling `verify()`
        #[allow(deprecated)]
        let mut checker = RandomizedMultChecker::new_using_rng(&mut rng);
        checker.add_1(g1, &a1, c1);
        checker.add_1(g2, &a3, c2);
        checker.add_1(g3, &a2, c3);
    }
}

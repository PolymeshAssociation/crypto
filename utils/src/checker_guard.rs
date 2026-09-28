use crate::error::UtilsError;

/// A checker accumulating checks that are all verified at once, driven by a `CheckerGuard`.
pub trait GuardedCheck {
    /// Verify everything accumulated so far.
    fn verify(&mut self) -> Result<(), UtilsError>;

    /// Cancel the accumulated checks.
    fn cancel(&mut self);
}

/// Ensures `verify` is called on the checker it holds. `with` and `with_err` take a closure that adds
/// the checks. If the closure returns `Ok`, the checker is verified, else it is cancelled and the
/// error returned.
#[derive(Debug)]
pub struct CheckerGuard<C: GuardedCheck> {
    inner: C,
}

impl<C: GuardedCheck> CheckerGuard<C> {
    pub(crate) fn wrap(inner: C) -> Self {
        Self { inner }
    }

    /// Run the given closure with the inner checker.
    pub fn with<O, E: From<UtilsError>>(
        mut self,
        f: impl FnOnce(&mut C) -> Result<O, E>,
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

    /// Same as `Self::with` except that this returns the given error if verification fails.
    pub fn with_err<O, E>(
        mut self,
        err: E,
        f: impl FnOnce(&mut C) -> Result<O, E>,
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

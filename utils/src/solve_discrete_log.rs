//! Baby-step giant-step solvers for small discrete logs.
//! Based on the paper [Computing Elliptic Curve Discrete Logarithms with Improved Baby-step Giant-step Algorithm](https://eprint.iacr.org/2015/605)
//!
//! [`old`] holds the original solvers keyed by the string encoding of a group element, [`affine`] the
//! baby-step giant-step solvers over a cached [`BabyStepsTable`], [`grumpy`] the grumpy-giants variants,
//! [`pairing`] the target-group solvers, and [`setup`] the precomputation and caching they share.

pub mod affine;
pub mod pairing;
pub mod setup;

#[cfg(test)]
pub mod tests;

pub use affine::{
    solve_discrete_log_bsgs_precomputed, solve_discrete_log_bsgs_precomputed_batch,
    solve_discrete_log_bsgs_precomputed_batch_large,
    solve_discrete_log_bsgs_precomputed_batch_with_table_size,
    solve_discrete_log_bsgs_precomputed_batch_with_table_size_large,
    solve_discrete_log_bsgs_precomputed_large, solve_discrete_log_bsgs_precomputed_with_table_size,
    solve_discrete_log_bsgs_precomputed_with_table_size_large, solve_discrete_log_given_table,
    solve_discrete_log_given_table_batch, solve_discrete_log_given_table_batch_large,
    solve_discrete_log_given_table_large,
};
pub use pairing::{
    solve_discrete_log_bsgs_precomputed_pairing,
    solve_discrete_log_bsgs_precomputed_pairing_with_table_size,
};
pub use setup::{
    clear_cached_tables, BabyStepsTable, MAX_NUM_BABY_STEPS, MAX_NUM_BABY_STEPS_BATCH,
};

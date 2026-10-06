use ark_bls12_381::{Bls12_381, Fr, G1Projective, G2Projective};
use ark_ec::pairing::Pairing;
use ark_std::{
    rand::{rngs::StdRng, SeedableRng},
    UniformRand,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dock_crypto_utils::solve_discrete_log::{
    solve_discrete_log_bsgs_precomputed, solve_discrete_log_bsgs_precomputed_batch,
    solve_discrete_log_bsgs_precomputed_batch_large, solve_discrete_log_bsgs_precomputed_large,
    solve_discrete_log_bsgs_precomputed_pairing, solve_discrete_log_given_table,
    solve_discrete_log_given_table_large,
    BabyStepsTable, MAX_NUM_BABY_STEPS,
};

// `precomputed_large` is tuned for dlogs above 2^32, plain precomputed for below, so each is
// benchmarked in its intended range.
const SMALL_BITS: [u32; 3] = [16, 24, 32];
const LARGE_BITS: [u32; 5] = [32, 36, 40, 44, 48];

// 52 and 56 bits are not benchmarked: with the default table size, a 52-bit worst case already
// takes ~460s, so it is not solvable under a minute.

// Worst case: dlog = max, i.e. the search has to walk the whole range.
fn worst_case_target(base: G1Projective, bits: u32) -> (u64, G1Projective) {
    let max = (1u64 << bits) - 1;
    (max, base * Fr::from(max))
}

fn bsgs_precomputed_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0u64);
    let base = G1Projective::rand(&mut rng);

    let mut group = c.benchmark_group("BSGS Precomputed (G1)");
    group.sample_size(10);
    for &bits in &SMALL_BITS {
        let (max, target) = worst_case_target(base, bits);
        group.bench_with_input(BenchmarkId::new("precomputed", bits), &bits, |b, _| {
            b.iter(|| solve_discrete_log_bsgs_precomputed(max, 0, base, target))
        });
    }
    for &bits in &LARGE_BITS {
        let (max, target) = worst_case_target(base, bits);
        group.bench_with_input(
            BenchmarkId::new("precomputed_large", bits),
            &bits,
            |b, _| b.iter(|| solve_discrete_log_bsgs_precomputed_large(max, 0, base, target)),
        );
    }
    group.finish();
}

fn bsgs_given_table_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0u64);
    let base = G1Projective::rand(&mut rng);
    let table = BabyStepsTable::new(base, MAX_NUM_BABY_STEPS);
    println!("Baby steps table size: {}", MAX_NUM_BABY_STEPS);

    let mut group = c.benchmark_group("BSGS Given Table (G1)");
    group.sample_size(10);
    for &bits in &SMALL_BITS {
        let (max, target) = worst_case_target(base, bits);
        group.bench_with_input(BenchmarkId::new("given_table", bits), &bits, |b, _| {
            b.iter(|| solve_discrete_log_given_table(&table, base, 0, max, target))
        });
    }
    for &bits in &LARGE_BITS {
        let (max, target) = worst_case_target(base, bits);
        group.bench_with_input(
            BenchmarkId::new("given_table_large", bits),
            &bits,
            |b, _| b.iter(|| solve_discrete_log_given_table_large(&table, base, 0, max, target)),
        );
    }
    group.finish();
}

fn bsgs_precomputed_pairing_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0u64);
    let g1 = G1Projective::rand(&mut rng);
    let g2 = G2Projective::rand(&mut rng);
    let base = <Bls12_381 as Pairing>::pairing(g1, g2);

    let mut group = c.benchmark_group("BSGS Precomputed (GT)");
    group.sample_size(10);
    for bits in [16u32, 24, 32, 36] {
        let max = (1u64 << bits) - 1;
        let target = base * Fr::from(max);
        group.bench_with_input(
            BenchmarkId::new("precomputed_pairing", bits),
            &bits,
            |b, _| b.iter(|| solve_discrete_log_bsgs_precomputed_pairing(max, 0, base, target)),
        );
    }
    group.finish();
}

// Batch sizes per width for the shared-base solver. The batch sizes its table from the number of targets,
// so the comparison against a loop of single solves also measures that. 40 bits stops at 128 targets: the
// loop alone already takes seconds per iteration there.
const BATCH_CASES: [(u32, &[usize]); 3] = [
    (20, &[16, 128, 1024]),
    (32, &[16, 128, 1024]),
    (40, &[16, 128]),
];

fn batch_targets(base: G1Projective, bits: u32, n: usize) -> (u64, Vec<G1Projective>) {
    let max = (1u64 << bits) - 1;
    // Spread over the range so the walk length varies across the batch.
    let targets = (0..n)
        .map(|i| base * Fr::from(max / n as u64 * i as u64 + 1))
        .collect();
    (max, targets)
}

fn bsgs_batch_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0u64);

    let mut group = c.benchmark_group("BSGS Batch (G1)");
    group.sample_size(10);
    for (bits, sizes) in BATCH_CASES {
        for &n in sizes {
            // A fresh base per configuration: the cached table is keyed by base, and the batch grows it
            // past the default that the loop is measured against.
            let base = G1Projective::rand(&mut rng);
            let (max, targets) = batch_targets(base, bits, n);
            let id = format!("{bits}bits/{n}");
            let large = bits > 32;
            group.bench_function(BenchmarkId::new("loop", &id), |b| {
                b.iter(|| {
                    targets
                        .iter()
                        .map(|t| {
                            if large {
                                solve_discrete_log_bsgs_precomputed_large(max, 0, base, *t)
                            } else {
                                solve_discrete_log_bsgs_precomputed(max, 0, base, *t)
                            }
                        })
                        .collect::<Vec<_>>()
                })
            });
            group.bench_function(BenchmarkId::new("batch", &id), |b| {
                b.iter(|| {
                    if large {
                        solve_discrete_log_bsgs_precomputed_batch_large(max, 0, base, &targets)
                    } else {
                        solve_discrete_log_bsgs_precomputed_batch(max, 0, base, &targets)
                    }
                })
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bsgs_precomputed_benchmark,
    bsgs_given_table_benchmark,
    bsgs_batch_benchmark,
    bsgs_precomputed_pairing_benchmark,
);
criterion_main!(benches);

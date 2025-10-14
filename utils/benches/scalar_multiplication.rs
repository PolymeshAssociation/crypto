use ark_bls12_381::{Bls12_381, G1Affine};
use ark_ec::{pairing::Pairing, VariableBaseMSM};
use ark_ff::UniformRand;
use ark_pallas::Affine as PallasAffine;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

fn scalar_multiplication_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0u64);

    // Create benchmark group for BLS12-381
    {
        let mut group_bls12 = c.benchmark_group("BLS12-381 Scalar Multiplication");

        // Test single scalar multiplication
        {
            let g1 = G1Affine::rand(&mut rng);
            let e = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);

            group_bls12.bench_function("single_scalar_mul", |b| {
                b.iter(|| g1 * e)
            });
        }

        // Test batches of scalar multiplications
        for batch_size in [10, 20, 50, 100, 1000].iter() {
            let g1 = G1Affine::rand(&mut rng);
            let scalars = (0..*batch_size)
                .map(|_| <Bls12_381 as Pairing>::ScalarField::rand(&mut rng))
                .collect::<Vec<_>>();

            // Sequential batch scalar multiplication
            group_bls12.bench_with_input(
                BenchmarkId::new("batch_scalar_mul_seq", batch_size),
                batch_size,
                |b, _| {
                    b.iter(|| {
                        scalars.iter().map(|e| g1 * e).collect::<Vec<_>>()
                    })
                }
            );

            // Parallel batch scalar multiplication (only when parallel feature is enabled)
            #[cfg(feature = "parallel")]
            group_bls12.bench_with_input(
                BenchmarkId::new("batch_scalar_mul_par", batch_size),
                batch_size,
                |b, _| {
                    b.iter(|| {
                        scalars.par_iter().map(|e| g1 * e).collect::<Vec<_>>()
                    })
                }
            );
        }

        group_bls12.finish();
    }

    // Create benchmark group for Pallas
    {
        let mut group_pallas = c.benchmark_group("Pallas Scalar Multiplication");

        // Test single scalar multiplication
        {
            let g1 = PallasAffine::rand(&mut rng);
            let e = <PallasAffine as ark_ec::AffineRepr>::ScalarField::rand(&mut rng);

            group_pallas.bench_function("single_scalar_mul", |b| {
                b.iter(|| g1 * e)
            });
        }

        // Test batches of scalar multiplications
        for batch_size in [10, 20, 50, 100, 1000].iter() {
            let g1 = PallasAffine::rand(&mut rng);
            let scalars = (0..*batch_size)
                .map(|_| <PallasAffine as ark_ec::AffineRepr>::ScalarField::rand(&mut rng))
                .collect::<Vec<_>>();

            // Sequential batch scalar multiplication
            group_pallas.bench_with_input(
                BenchmarkId::new("batch_scalar_mul_seq", batch_size),
                batch_size,
                |b, _| {
                    b.iter(|| {
                        scalars.iter().map(|e| g1 * e).collect::<Vec<_>>()
                    })
                }
            );

            // Parallel batch scalar multiplication (only when parallel feature is enabled)
            #[cfg(feature = "parallel")]
            group_pallas.bench_with_input(
                BenchmarkId::new("batch_scalar_mul_par", batch_size),
                batch_size,
                |b, _| {
                    b.iter(|| {
                        scalars.par_iter().map(|e| g1 * e).collect::<Vec<_>>()
                    })
                }
            );
        }

        group_pallas.finish();
    }
}

fn msm_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0u64);

    // Test different MSM sizes for BLS12-381
    {
        let mut group_bls12 = c.benchmark_group("BLS12-381 MSM");

        for msm_size in [10, 20, 50, 100, 1000].iter() {
            let points = (0..*msm_size)
                .map(|_| G1Affine::rand(&mut rng))
                .collect::<Vec<_>>();
            let scalars = (0..*msm_size)
                .map(|_| <Bls12_381 as Pairing>::ScalarField::rand(&mut rng))
                .collect::<Vec<_>>();

            group_bls12.bench_with_input(
                BenchmarkId::new("msm_unchecked", msm_size), 
                msm_size,
                |b, _| {
                    b.iter(|| <Bls12_381 as Pairing>::G1::msm_unchecked(&points, &scalars))
                }
            );
        }

        group_bls12.finish();
    }

    // Test different MSM sizes for Pallas
    {
        let mut group_pallas = c.benchmark_group("Pallas MSM");

        for msm_size in [10, 20, 50, 100, 1000].iter() {
            let points = (0..*msm_size)
                .map(|_| PallasAffine::rand(&mut rng))
                .collect::<Vec<_>>();
            let scalars = (0..*msm_size)
                .map(|_| <PallasAffine as ark_ec::AffineRepr>::ScalarField::rand(&mut rng))
                .collect::<Vec<_>>();

            group_pallas.bench_with_input(
                BenchmarkId::new("msm_unchecked", msm_size), 
                msm_size,
                |b, _| {
                    b.iter(|| <PallasAffine as ark_ec::AffineRepr>::Group::msm_unchecked(&points, &scalars))
                }
            );
        }

        group_pallas.finish();
    }
}

criterion_group!(
    benches,
    scalar_multiplication_benchmark,
    msm_benchmark
);
criterion_main!(benches);

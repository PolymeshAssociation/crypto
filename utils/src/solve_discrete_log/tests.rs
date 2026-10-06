use super::{
    new::BaseTable,
    setup::{bsgs_hasher, BsgsHasher},
    *,
};
use std::{
    mem::size_of,
    ops::Mul,
    sync::Arc,
    time::{Duration, Instant},
};

use ark_bls12_381::{Bls12_381, Fr, G1Projective, G2Projective};
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    AffineRepr, CurveGroup, PrimeGroup,
};
use ark_ff::AdditiveGroup;
use ark_std::{
    rand::{prelude::StdRng, SeedableRng},
    UniformRand,
};
use hashbrown::HashMap;
use integer_sqrt::IntegerSquareRoot;

#[test]
fn solving_discrete_log() {
    let mut rng = StdRng::seed_from_u64(0u64);

    fn check<G: AdditiveGroup + Mul<Fr, Output = G>>(
        rng: &mut StdRng,
        base: G,
        check_large_value: bool,
    ) {
        let checks_per_max = 10;
        let mut total_checks = 0;
        let mut duration_naive = Duration::default();
        let mut duration_bsgs = Duration::default();
        let mut duration_bsgs_alt = Duration::default();

        for max in [1, 2, 3, 4, 5, 6, 8, 15, 16, 31, 32, 255, 256, 65535] {
            for _ in 0..checks_per_max {
                let dl = (u16::rand(rng) as u64 % max) + 1;
                let target = base * Fr::from(dl);

                // println!("For max={} and discrete log={}", max, dl);
                let start = Instant::now();
                let dl_naive = solve_discrete_log_brute_force(max, base, target);
                let time = start.elapsed();
                assert_eq!(dl, dl_naive.unwrap());
                // println!("Time for naive approach: {:?}", time);
                duration_naive += time;

                let start = Instant::now();
                let dl_bsgs = solve_discrete_log_bsgs(max, base, target);
                let time = start.elapsed();
                assert_eq!(dl, dl_bsgs.unwrap());
                // println!("Time for BSGS approach: {:?}", time);
                duration_bsgs += time;

                let start = Instant::now();
                let dl_bsgs_alt = solve_discrete_log_bsgs_alt(max, base, target);
                let time = start.elapsed();
                assert_eq!(dl, dl_bsgs_alt.unwrap());
                // println!("Time for alt. BSGS approach: {:?}", time);
                duration_bsgs_alt += time;

                total_checks += 1;
            }
        }

        if check_large_value {
            for dl in [
                u32::MAX as u64,                  // 32-bit value
                u32::MAX as u64 * u8::MAX as u64, // 40-bit value
            ] {
                let target = base * Fr::from(dl);
                println!("For discrete log={}", dl);

                let start = Instant::now();
                let dl_bsgs = solve_discrete_log_bsgs(dl, base, target);
                let time = start.elapsed();
                assert_eq!(dl, dl_bsgs.unwrap());
                println!("Time for BSGS approach: {:?}", time);

                let start = Instant::now();
                let dl_bsgs_alt = solve_discrete_log_bsgs_alt(dl, base, target);
                let time = start.elapsed();
                assert_eq!(dl, dl_bsgs_alt.unwrap());
                println!("Time for alt. BSGS approach: {:?}", time);
            }
        }

        let target = base * Fr::from(10);
        assert!(solve_discrete_log_brute_force(8, base, target).is_none());
        assert!(solve_discrete_log_bsgs(8, base, target).is_none());

        println!("For total {} checks, brute force took {:?} and baby step giant step took {:?} and alt. baby step giant step took {:?}", total_checks, duration_naive, duration_bsgs, duration_bsgs_alt);
    }

    println!("\n\nTesting for group G1");
    let g1 = G1Projective::rand(&mut rng);
    check::<G1Projective>(&mut rng, g1, true);

    println!("\n\nTesting for group G2");
    let g2 = G2Projective::rand(&mut rng);
    check::<G2Projective>(&mut rng, g2, true);

    println!("\n\nTesting for group GT");
    let gt = <Bls12_381 as Pairing>::pairing(g1, g2);
    check::<PairingOutput<Bls12_381>>(&mut rng, gt, false);
}

#[test]
fn solving_discrete_log_precomputed() {
    let mut rng = StdRng::seed_from_u64(1u64);

    fn check_curve<G: CurveGroup + Send + Sync + 'static + Mul<Fr, Output = G>>(
        rng: &mut StdRng,
        base: G,
    ) {
        for &(min, max) in &[
            (0u64, 255u64),
            (1, 255),
            (0, 65535),
            (1000, 5000),
            (5, 5),
            (0, 0),
        ] {
            let span = max - min + 1;
            for _ in 0..5 {
                let dl = min + (u16::rand(rng) as u64 % span);
                let target = base * Fr::from(dl);
                assert_eq!(
                    Some(dl),
                    solve_discrete_log_bsgs_precomputed(max, min, base, target)
                );
                assert_eq!(
                    Some(dl),
                    solve_discrete_log_bsgs_precomputed_with_table_size(max, min, 64, base, target)
                );
            }
            // A discrete log below `min` is not returned.
            if min > 0 {
                let target = base * Fr::from(min - 1);
                assert_eq!(
                    None,
                    solve_discrete_log_bsgs_precomputed(max, min, base, target)
                );
            }
            // A discrete log above `max` is not returned.
            let target = base * Fr::from(max + 1);
            assert_eq!(
                None,
                solve_discrete_log_bsgs_precomputed(max, min, base, target)
            );
        }
    }

    fn check_gt(rng: &mut StdRng, base: PairingOutput<Bls12_381>) {
        for &(min, max) in &[(0u64, 255u64), (1, 255), (0, 65535), (100, 1000), (0, 0)] {
            let span = max - min + 1;
            for _ in 0..3 {
                let dl = min + (u16::rand(rng) as u64 % span);
                let target = base * Fr::from(dl);
                assert_eq!(
                    Some(dl),
                    solve_discrete_log_bsgs_precomputed_pairing(max, min, base, target)
                );
            }
            if min > 0 {
                let target = base * Fr::from(min - 1);
                assert_eq!(
                    None,
                    solve_discrete_log_bsgs_precomputed_pairing(max, min, base, target)
                );
            }
            let target = base * Fr::from(max + 1);
            assert_eq!(
                None,
                solve_discrete_log_bsgs_precomputed_pairing(max, min, base, target)
            );
        }
    }

    let g1 = G1Projective::rand(&mut rng);
    check_curve::<G1Projective>(&mut rng, g1);
    let g2 = G2Projective::rand(&mut rng);
    check_curve::<G2Projective>(&mut rng, g2);
    let gt = <Bls12_381 as Pairing>::pairing(g1, g2);
    check_gt(&mut rng, gt);

    // Amortized comparison against `solve_discrete_log_bsgs_alt` for repeated decryption in the target group.
    let base = <Bls12_381 as Pairing>::pairing(G1Projective::rand(&mut rng), g2);
    let max = 65535u64;
    let iters = 200usize;
    let dls: Vec<u64> = (0..iters).map(|_| u16::rand(&mut rng) as u64).collect();
    let targets: Vec<_> = dls.iter().map(|&d| base * Fr::from(d)).collect();

    let start = Instant::now();
    for (i, t) in targets.iter().enumerate() {
        assert_eq!(
            Some(dls[i]),
            solve_discrete_log_bsgs_precomputed_pairing_with_table_size(max, 0, 256, base, *t)
        );
    }
    let precomputed = start.elapsed();

    let start = Instant::now();
    for (i, t) in targets.iter().enumerate() {
        assert_eq!(Some(dls[i]), solve_discrete_log_bsgs_alt(max, base, *t));
    }
    let existing = start.elapsed();

    println!(
        "GT, {} solves up to {}: precomputed(+cache) {:?} vs bsgs_alt {:?}",
        iters, max, precomputed, existing
    );

    let c_base = G1Projective::rand(&mut rng);
    let c_targets: Vec<_> = dls.iter().map(|&d| c_base * Fr::from(d)).collect();

    let start = Instant::now();
    for (i, t) in c_targets.iter().enumerate() {
        assert_eq!(
            Some(dls[i]),
            solve_discrete_log_bsgs_precomputed(max, 0, c_base, *t)
        );
    }
    let c_precomputed = start.elapsed();

    let start = Instant::now();
    for (i, t) in c_targets.iter().enumerate() {
        assert_eq!(Some(dls[i]), solve_discrete_log_bsgs_alt(max, c_base, *t));
    }
    let c_existing = start.elapsed();

    println!(
        "G1, {} solves up to {}: precomputed(+cache) {:?} vs bsgs_alt {:?}",
        iters, max, c_precomputed, c_existing
    );

    let shot_max = 65535u64;
    let m_bal = shot_max.integer_sqrt();

    let mut t_old = Duration::default();
    let mut t_new = Duration::default();
    for _ in 0..iters {
        let b = <Bls12_381 as Pairing>::pairing(G1Projective::rand(&mut rng), g2);
        let dl = u16::rand(&mut rng) as u64;
        let target = b * Fr::from(dl);
        let s = Instant::now();
        assert_eq!(Some(dl), solve_discrete_log_bsgs(shot_max, b, target));
        t_old += s.elapsed();
        let s = Instant::now();
        assert_eq!(
            Some(dl),
            solve_discrete_log_bsgs_precomputed_pairing_with_table_size(
                shot_max, 0, m_bal, b, target
            )
        );
        t_new += s.elapsed();
    }
    println!(
        "GT single-shot (no reuse), {} solves up to {} (m={}): bsgs {:?} vs new {:?}",
        iters, shot_max, m_bal, t_old, t_new
    );

    let mut t_old = Duration::default();
    let mut t_new = Duration::default();
    for _ in 0..iters {
        let b = G1Projective::rand(&mut rng);
        let dl = u16::rand(&mut rng) as u64;
        let target = b * Fr::from(dl);
        let s = Instant::now();
        assert_eq!(Some(dl), solve_discrete_log_bsgs(shot_max, b, target));
        t_old += s.elapsed();
        let s = Instant::now();
        assert_eq!(
            Some(dl),
            solve_discrete_log_bsgs_precomputed_with_table_size(shot_max, 0, m_bal, b, target)
        );
        t_new += s.elapsed();
    }
    println!(
        "G1 single-shot (no reuse), {} solves up to {} (m={}): bsgs {:?} vs new {:?}",
        iters, shot_max, m_bal, t_old, t_new
    );
}

// Exercises the chunked path: `max` above `u32::MAX` and a table small enough that the center range spans
// more than one 2^32-wide chunk, so the discrete logs below land in later chunks.
#[test]
fn solving_discrete_log_chunked() {
    let mut rng = StdRng::seed_from_u64(2u64);
    let base = G1Projective::rand(&mut rng);
    let max = 1u64 << 33;
    let table_size = 1u64 << 16;
    for dl in [
        0u64,
        12345,
        (1u64 << 32) - 1,
        1u64 << 32,
        (1u64 << 32) + 6789,
        5_000_000_123,
        max,
    ] {
        let target = base * Fr::from(dl);
        assert_eq!(
            Some(dl),
            solve_discrete_log_bsgs_precomputed_with_table_size(max, 0, table_size, base, target)
        );
        assert_eq!(
            Some(dl),
            solve_discrete_log_bsgs_precomputed_with_table_size_large(
                max, 0, table_size, base, target
            )
        );
    }
}

// Exercises the direct baby-step fast path, a giant walk inside chunk 0, and the chunk-0 ->
// chunk-1 boundary, i.e. every branch of the fast-path / chunk-0-first restructure.
#[test]
fn solving_discrete_log_fast_path_and_at_boundary() {
    let mut rng = StdRng::seed_from_u64(3u64);
    let base = G1Projective::rand(&mut rng);
    let m: u64 = 1 << 16;
    let max = 1u64 << 33;
    for dl in [
        0u64, // identity fast path
        1,    // direct baby step
        m - 1,
        m,                // largest baby step (direct hit)
        m + 1,            // first value past the table (giant walk in chunk 0)
        1u64 << 20,       // mid chunk 0
        (1u64 << 32) - 1, // end of chunk 0
        1u64 << 32,       // start of chunk 1 (parallel path)
        (1u64 << 32) + 1,
    ] {
        let target = base * Fr::from(dl);
        assert_eq!(
            Some(dl),
            solve_discrete_log_bsgs_precomputed_with_table_size(max, 0, m, base, target),
            "dl={dl}"
        );
    }
    // A non-zero `min` shifts the same fast path / boundary.
    let min = 1000u64;
    for dl in [min, min + m, min + (1u64 << 32)] {
        let target = base * Fr::from(dl);
        assert_eq!(
            Some(dl),
            solve_discrete_log_bsgs_precomputed_with_table_size(max, min, m, base, target),
            "min={min}, dl={dl}"
        );
    }
}

#[test]
fn solving_discrete_log_full_u64_interval() {
    // The full `u64` interval, where `width + num_baby_steps` would overflow. Dlogs stay inside chunk 0
    // so the walk ends early.
    let mut rng = StdRng::seed_from_u64(13u64);
    let base = G1Projective::rand(&mut rng);
    let g2 = G2Projective::rand(&mut rng);
    let base_gt = <Bls12_381 as Pairing>::pairing(G1Projective::rand(&mut rng), g2);
    let m = 1u64 << 10;
    for dl in [0u64, 5, m, m + 1, 5000, 500_000] {
        let target = base * Fr::from(dl);
        assert_eq!(
            Some(dl),
            solve_discrete_log_bsgs_precomputed_with_table_size(u64::MAX, 0, m, base, target),
            "dl={dl}"
        );
        assert_eq!(
            vec![Some(dl)],
            solve_discrete_log_bsgs_precomputed_batch_with_table_size(
                u64::MAX,
                0,
                m,
                base,
                &[target]
            ),
            "batch dl={dl}"
        );
        assert_eq!(
            Some(dl),
            solve_discrete_log_bsgs_precomputed_pairing_with_table_size(
                u64::MAX,
                0,
                m,
                base_gt,
                base_gt * Fr::from(dl)
            ),
            "pairing dl={dl}"
        );
    }
}

#[cfg(feature = "std")]
#[test]
fn clearing_cached_tables() {
    let mut rng = StdRng::seed_from_u64(17u64);
    let base = G1Projective::rand(&mut rng).into_affine();
    let t1 = BabyStepsTable::<G1Projective>::get_or_build(base, 1 << 8).unwrap();
    let t2 = BabyStepsTable::<G1Projective>::get_or_build(base, 1 << 8).unwrap();
    assert!(Arc::ptr_eq(&t1, &t2));
    clear_cached_tables();
    let t3 = BabyStepsTable::<G1Projective>::get_or_build(base, 1 << 8).unwrap();
    assert!(!Arc::ptr_eq(&t1, &t3));
    let target = base * Fr::from(1000u64);
    assert_eq!(
        Some(1000),
        solve_discrete_log_bsgs_precomputed_with_table_size(1 << 16, 0, 1 << 8, base.into_group(), target)
    );
}

#[test]
fn base_mul_window_table() {
    let mut rng = StdRng::seed_from_u64(4u64);
    let base = G1Projective::rand(&mut rng);
    for window in [1u32, 3, 8, 12] {
        let table = BaseTable::new(base, 20, window);
        for v in [0u64, 1, 2, 255, 256, 1 << 19, (1u64 << 20) - 1] {
            assert_eq!(table.mul(v), base.mul_bigint([v]), "window={window}, v={v}");
        }
    }
    assert_eq!(
        BaseTable::empty(base).mul(12345),
        base.mul_bigint([12345u64])
    );
}

#[test]
fn solving_discrete_log_batch() {
    let mut rng = StdRng::seed_from_u64(5u64);
    let base = G1Projective::rand(&mut rng);
    let m: u64 = 1 << 16;
    let max = 1u64 << 33;

    let mut dls = vec![
        0u64,
        1,
        m - 1,
        m,
        m + 1,
        1 << 20,
        (1u64 << 32) - 1,
        1u64 << 32,
        max,
    ];
    dls.extend((0..40u64).map(|i| i * 1_234_567 + 7));
    let targets = dls.iter().map(|d| base * Fr::from(*d)).collect::<Vec<_>>();
    let expected = dls.iter().map(|d| Some(*d)).collect::<Vec<_>>();

    assert_eq!(
        expected,
        solve_discrete_log_bsgs_precomputed_batch_with_table_size(max, 0, m, base, &targets)
    );
    assert_eq!(
        expected,
        solve_discrete_log_bsgs_precomputed_batch_with_table_size_large(max, 0, m, base, &targets)
    );
    assert_eq!(
        expected,
        solve_discrete_log_bsgs_precomputed_batch(max, 0, base, &targets)
    );

    assert_eq!(
        expected[..2],
        solve_discrete_log_bsgs_precomputed_batch_with_table_size(max, 0, m, base, &targets[..2])[..]
    );

    let table = BabyStepsTable::new(base, m);
    assert_eq!(
        expected,
        solve_discrete_log_given_table_batch(&table, base, 0, max, &targets)
    );
    assert_eq!(
        expected,
        solve_discrete_log_given_table_batch_large(&table, base, 0, max, &targets)
    );

    // A non-zero `min` shifts the same values.
    let min = 1000u64;
    let dls = [min, min + 1, min + m, min + (1u64 << 32), max];
    let targets = dls.iter().map(|d| base * Fr::from(*d)).collect::<Vec<_>>();
    assert_eq!(
        dls.iter().map(|d| Some(*d)).collect::<Vec<_>>(),
        solve_discrete_log_bsgs_precomputed_batch_with_table_size(max, min, m, base, &targets)
    );

    let out_of_range = base * Fr::from(max + 1);
    assert_eq!(
        vec![None],
        solve_discrete_log_bsgs_precomputed_batch_with_table_size(max, 0, m, base, &[out_of_range])
    );
    assert!(solve_discrete_log_bsgs_precomputed_batch(max, 0, base, &[]).is_empty());
    assert_eq!(
        vec![None],
        solve_discrete_log_bsgs_precomputed_batch(5, 10, base, &[base])
    );
}

// Compares the grumpy-giants algorithms against the existing negation BSGS and prints timings.
#[test]
fn grumpy_vs_bsgs_negation_timings() {
    let mut rng = StdRng::seed_from_u64(42u64);
    let max = (1u64 << 20) - 1;
    let iters = 50usize;
    // Distinct bases so each cached-table algorithm builds its own table (no cross-subsidy).
    let base_ref = G1Projective::rand(&mut rng);
    let base_g = G1Projective::rand(&mut rng);
    let dls: Vec<u64> = (0..iters)
        .map(|_| (u32::rand(&mut rng) as u64) % (max + 1))
        .collect();

    // Correctness spot-check, and it warms `base_g`'s grumpy table in the cache.
    for dl in [0u64, 1, max] {
        let t = base_g * Fr::from(dl);
        assert_eq!(Some(dl), solve_discrete_log_grumpy(max, base_g, t));
        assert_eq!(
            Some(dl),
            solve_discrete_log_grumpy_precomputed(max, base_g, t)
        );
    }

    let targets_ref: Vec<_> = dls.iter().map(|&d| base_ref * Fr::from(d)).collect();
    // Warm `base_ref`'s bsgs table too, so the timed precomputed loops below both measure warm
    // scan time rather than charging a one-off table build to bsgs alone.
    let _ = solve_discrete_log_bsgs_precomputed(max, 0, base_ref, base_ref * Fr::from(1u64));
    let start = Instant::now();
    for (i, t) in targets_ref.iter().enumerate() {
        assert_eq!(
            Some(dls[i]),
            solve_discrete_log_bsgs_precomputed(max, 0, base_ref, *t)
        );
    }
    let t_bsgs = start.elapsed();

    // Same solves with the cached table passed in directly: isolates the cache lookup from the scan.
    let ref_table =
        BabyStepsTable::<G1Projective>::get_or_build(base_ref.into_affine(), MAX_NUM_BABY_STEPS)
            .unwrap();
    let start = Instant::now();
    for (i, t) in targets_ref.iter().enumerate() {
        assert_eq!(
            Some(dls[i]),
            solve_discrete_log_given_table(&ref_table, base_ref, 0, max, *t)
        );
    }
    let t_bsgs_direct = start.elapsed();

    let targets_g: Vec<_> = dls.iter().map(|&d| base_g * Fr::from(d)).collect();
    let start = Instant::now();
    for (i, t) in targets_g.iter().enumerate() {
        assert_eq!(Some(dls[i]), solve_discrete_log_grumpy(max, base_g, *t));
    }
    let t_grumpy = start.elapsed();

    let start = Instant::now();
    for (i, t) in targets_g.iter().enumerate() {
        assert_eq!(
            Some(dls[i]),
            solve_discrete_log_grumpy_precomputed(max, base_g, *t)
        );
    }
    let t_grumpy_pre = start.elapsed();

    println!(
        "max={max}, iters={iters}: bsgs_precomputed {t_bsgs:?} (avg {:?}), bsgs_given_table {t_bsgs_direct:?} (avg {:?}), grumpy {t_grumpy:?} (avg {:?}), grumpy_precomputed {t_grumpy_pre:?} (avg {:?})",
        t_bsgs / iters as u32,
        t_bsgs_direct / iters as u32,
        t_grumpy / iters as u32,
        t_grumpy_pre / iters as u32
    );

    // Same solves with already constructed tables: isolates scan time from table-build time.
    // Both tables hold 1024 baby steps (equal memory; grumpy stays complete at any size).
    let bsgs_table = BabyStepsTable::new(base_ref, 1024);
    let grumpy_table = BabyStepsTable::new(base_g, 1024);
    for dl in [0u64, 1, max, max + 1] {
        let t = base_g * Fr::from(dl);
        let expect = (dl <= max).then_some(dl);
        assert_eq!(
            expect,
            solve_discrete_log_grumpy_given_table(&grumpy_table, base_g, max, t)
        );
    }
    let start = Instant::now();
    for (i, t) in targets_ref.iter().enumerate() {
        assert_eq!(
            Some(dls[i]),
            solve_discrete_log_given_table(&bsgs_table, base_ref, 0, max, *t)
        );
    }
    let t_bsgs_given = start.elapsed();
    let start = Instant::now();
    for (i, t) in targets_g.iter().enumerate() {
        assert_eq!(
            Some(dls[i]),
            solve_discrete_log_grumpy_given_table(&grumpy_table, base_g, max, *t)
        );
    }
    let t_grumpy_given = start.elapsed();
    println!(
        "given tables: bsgs {t_bsgs_given:?} (avg {:?}), grumpy {t_grumpy_given:?} (avg {:?})",
        t_bsgs_given / iters as u32,
        t_grumpy_given / iters as u32
    );
}

// Worst-case (dl = max) comparison at large intervals. Only the cached-table algorithms are
// feasible here. Grumpy uses the minimal complete baby table M + 1 with M = ceil(sqrt(max/2)).
#[test]
#[ignore = "slow and memory-heavy (tens of millions of baby steps at 50 bits), run explicitly with --ignored"]
fn grumpy_large_interval_timings() {
    #[cfg(feature = "parallel")]
    println!("rayon threads: {}", rayon::current_num_threads());
    let mut rng = StdRng::seed_from_u64(7u64);
    let base_ref = G1Projective::rand(&mut rng);
    let base_g = G1Projective::rand(&mut rng);
    for bits in [32u32, 36, 40, 42, 44, 48, 50] {
        let max = (1u64 << bits) - 1;
        let m = ((max as f64) / 2.0).sqrt().ceil() as u64;
        let start = Instant::now();
        let bsgs_table = BabyStepsTable::new(base_ref, MAX_NUM_BABY_STEPS);
        let t_bsgs_build = start.elapsed();
        let target_ref = base_ref * Fr::from(max);
        let start = Instant::now();
        assert_eq!(
            Some(max),
            solve_discrete_log_given_table(&bsgs_table, base_ref, 0, max, target_ref)
        );
        let t_bsgs = start.elapsed();
        let start = Instant::now();
        let grumpy_table = BabyStepsTable::new(base_g, m + 1);
        let t_grumpy_build = start.elapsed();
        let target_g = base_g * Fr::from(max);
        let start = Instant::now();
        assert_eq!(
            Some(max),
            solve_discrete_log_grumpy_given_table(&grumpy_table, base_g, max, target_g)
        );
        let t_grumpy = start.elapsed();
        println!(
            "bits={bits}: bsgs build {t_bsgs_build:?} solve {t_bsgs:?}; grumpy table {} build {t_grumpy_build:?} solve {t_grumpy:?}",
            m + 1
        );
    }
}

// Exercises the multi-chunk grumpy path: `max` large enough that `lmax` spans more than one
// `GRUMPY_PAR_CHUNK`, so the parallel build takes the chunked scan (the interleaved scan handles
// it under a no-`parallel` build). Small table keeps the build cheap; completeness is unaffected.
#[test]
fn grumpy_multichunk() {
    let mut rng = StdRng::seed_from_u64(9u64);
    let base = G1Projective::rand(&mut rng);
    let max = 1u64 << 30;
    let table_size = 1u64 << 14;
    for dl in [0u64, 1, 1 << 10, 1 << 20, (1 << 30) - 12345, max] {
        let target = base * Fr::from(dl);
        assert_eq!(
            Some(dl),
            solve_discrete_log_grumpy_precomputed_with_table_size(max, table_size, base, target),
            "dl={dl}"
        );
    }
}

// Grumpy completeness does not depend on the baby table size: a single baby step still resolves
// every in-range dlog via the giant-1/giant-2 families. Exhaustive over small intervals.
#[test]
fn grumpy_complete_with_tiny_table() {
    let mut rng = StdRng::seed_from_u64(11u64);
    let base = G1Projective::rand(&mut rng);
    let table = BabyStepsTable::new(base, 1);
    for &max in &[7u64, 50, 255, 1000, 4095] {
        for dl in 0..=max {
            let target = base * Fr::from(dl);
            assert_eq!(
                Some(dl),
                solve_discrete_log_grumpy_given_table(&table, base, max, target),
                "max={max}, dl={dl}"
            );
        }
    }
}

#[test]
fn ff_index_map_shrinkage() {
    const NUM_STEPS: u64 = 1 << 16;

    // Bytes a hashbrown table of `len` entries of type `E` takes: `size_of::<E>()` plus one control
    // byte per bucket, over the bucket count hashbrown rounds `len` up to.
    fn table_bytes<E>(len: usize) -> usize {
        let buckets = if len < 8 {
            8
        } else {
            (len * 8 / 7).next_power_of_two()
        };
        buckets * (size_of::<E>() + 1)
    }

    fn check<G: CurveGroup + Send + Sync>(name: &str) {
        let base = G::generator();
        let table = BabyStepsTable::new(base, NUM_STEPS);
        let (map_len, spill_len) = table.map.lengths();

        // The map `FpIndex` replaces: the x-coordinate itself as the key.
        let mut pts = Vec::with_capacity(NUM_STEPS as usize);
        let mut cur = G::zero();
        for _ in 0..NUM_STEPS {
            cur += base;
            pts.push(cur);
        }
        let mut naive: HashMap<G::BaseField, u32, BsgsHasher> =
            HashMap::with_capacity_and_hasher(NUM_STEPS as usize, bsgs_hasher());
        for (i, p) in G::normalize_batch(&pts).iter().enumerate() {
            let (x, _) = p.xy().expect("baby step is never the identity");
            naive.insert(x, i as u32 + 1);
        }
        assert_eq!(naive.len(), map_len + spill_len);

        let naive_bytes = table_bytes::<(G::BaseField, u32)>(naive.len());
        let index_bytes =
            table_bytes::<(u64, (u64, u32))>(map_len) + spill_len * size_of::<(u64, u64, u32)>();
        let kb = |b: usize| b as f64 / 1024.0;
        println!(
            "{:<14} {:>6} {:>12.1} {:>12.1} {:>8.2}x {:>7}",
            name,
            size_of::<G::BaseField>(),
            kb(naive_bytes),
            kb(index_bytes),
            naive_bytes as f64 / index_bytes as f64,
            spill_len,
        );
        assert!(index_bytes < naive_bytes);
    }

    println!("Baby steps table of {NUM_STEPS} entries");
    println!(
        "{:<14} {:>6} {:>12} {:>12} {:>9} {:>7}",
        "curve", "field", "naive KB", "index KB", "shrinkage", "spill"
    );
    check::<G1Projective>("BLS12-381 G1");
    check::<G2Projective>("BLS12-381 G2");
    check::<ark_pallas::Projective>("Pallas");
    check::<ark_vesta::Projective>("Vesta");
}

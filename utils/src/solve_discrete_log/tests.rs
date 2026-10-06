use super::{
    affine::BaseTable,
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
use ark_std::{
    rand::{prelude::StdRng, SeedableRng},
    UniformRand,
};
use hashbrown::HashMap;
use integer_sqrt::IntegerSquareRoot;

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

    println!(
        "GT, {} solves up to {}: precomputed(+cache) {:?}",
        iters, max, precomputed
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

    println!(
        "G1, {} solves up to {}: precomputed(+cache) {:?}",
        iters, max, c_precomputed
    );

    let shot_max = 65535u64;
    let m_bal = shot_max.integer_sqrt();

    let mut t_new = Duration::default();
    for _ in 0..iters {
        let b = <Bls12_381 as Pairing>::pairing(G1Projective::rand(&mut rng), g2);
        let dl = u16::rand(&mut rng) as u64;
        let target = b * Fr::from(dl);
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
        "GT single-shot (no reuse), {} solves up to {} (m={}): bsgs {:?}",
        iters, shot_max, m_bal, t_new
    );

    let mut t_new = Duration::default();
    for _ in 0..iters {
        let b = G1Projective::rand(&mut rng);
        let dl = u16::rand(&mut rng) as u64;
        let target = b * Fr::from(dl);
        let s = Instant::now();
        assert_eq!(
            Some(dl),
            solve_discrete_log_bsgs_precomputed_with_table_size(shot_max, 0, m_bal, b, target)
        );
        t_new += s.elapsed();
    }
    println!(
        "G1 single-shot (no reuse), {} solves up to {} (m={}): bsgs {:?}",
        iters, shot_max, m_bal, t_new
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

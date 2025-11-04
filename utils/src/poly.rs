use ark_ff::{Field, Zero};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
use ark_std::{cfg_into_iter, vec::Vec, vec};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Naive multiplication (n^2) of 2 polynomials defined over prime fields
/// Note: Using multiply operator from ark-poly is orders of magnitude slower than naive multiplication
pub fn multiply_poly<F: Field>(
    left: &DensePolynomial<F>,
    right: &DensePolynomial<F>,
) -> DensePolynomial<F> {
    let mut product = (0..(left.degree() + right.degree() + 1))
        .map(|_| F::zero())
        .collect::<Vec<_>>();
    for i in 0..=left.degree() {
        for j in 0..=right.degree() {
            product[i + j] += left.coeffs[i] * right.coeffs[j];
        }
    }
    DensePolynomial::from_coefficients_vec(product)
}

/// Multiply given polynomials together
pub fn multiply_many_polys<F: Field>(polys: Vec<DensePolynomial<F>>) -> DensePolynomial<F> {
    #[cfg(not(feature = "parallel"))]
    let r = polys
        .into_iter()
        .reduce(|a, b| multiply_poly(&a, &b))
        .unwrap();

    #[cfg(feature = "parallel")]
    let r = {
        use ark_std::vec;
        let one = || DensePolynomial::from_coefficients_vec(vec![F::one()]);
        polys
            .into_par_iter()
            .reduce(one, |a, b| multiply_poly(&a, &b))
    };

    r
}

/// Given a vector of polynomials `polys` and scalars `coeffs`, return their inner product `polys[0] * coeffs[0] + polys[1] * coeffs[1] + ...`
pub fn inner_product_poly<F: Field>(
    polys: &[DensePolynomial<F>],
    coeffs: Vec<F>,
) -> DensePolynomial<F> {
    let product = cfg_into_iter!(coeffs)
        .zip(cfg_into_iter!(polys))
        .map(|(f, p)| p * f);

    let zero = DensePolynomial::zero;
    cfg_iter_sum!(product, zero)
}

/// Create a polynomial from given `roots` as `(x-roots[0])*(x-roots[1])*(x-roots[2])*..`
pub fn poly_from_roots_parallel<F: Field>(roots: &[F]) -> DensePolynomial<F> {
    if roots.is_empty() {
        return DensePolynomial::zero();
    }

    // [(x-roots[0]), (x-roots[1]), (x-roots[2]), ..., (x-roots[last])]
    let terms = cfg_into_iter!(roots)
        .map(|i| DensePolynomial::from_coefficients_slice(&[-*i, F::one()]))
        .collect::<Vec<_>>();

    // Product (x-roots[0]) * (x-roots[1]) * (x-roots[2]) * ... * (x-roots[last])
    multiply_many_polys(terms)
}

/// Create a polynomial from given `roots` as `(x-roots[0])*(x-roots[1])*(x-roots[2])*..`
/// Check https://stackoverflow.com/questions/33594384/find-the-coefficients-of-the-polynomial-given-its-roots for detailed explanation.
/// This function is sequential unlike poly_from_roots_parallel but still performs better on reasonably large size.
pub fn poly_from_roots<F: Field>(roots: &[F]) -> DensePolynomial<F> {
    if roots.is_empty() {
        return DensePolynomial::zero();
    }

    let n = roots.len() + 1;
    // coeffs is the array coefficients starting from highest degree to lowest
    let mut coeffs = vec![F::zero(); n];
    // Coefficient of highest degree is 1
    coeffs[0] = F::one();

    for (i, root) in roots.iter().enumerate() {
        // i roots can only make a polynomial with i+1 coefficients
        // Iterate from i+1 to 1 since coeffs[0] doesn't change because its for the highest degree
        for j in (1..i+2).rev() {
            // eg. (a_2.x^2 + a_1.x + a_0).(x-r) = a_2.x^3 + (a_1 - a_2).x^2 + ...
            coeffs[j] = coeffs[j] - (*root) * coeffs[j-1];
        }
    }
    // Convert the array to have coefficients starting from lowest degree to highest
    coeffs.reverse();

    DensePolynomial::from_coefficients_vec(coeffs)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::{One, Zero};
    use ark_ff::UniformRand;
    use ark_std::rand::rngs::StdRng;
    use ark_std::rand::SeedableRng;

    #[test]
    fn test_poly_from_roots() {
        let roots: Vec<Fr> = vec![];
        assert!(poly_from_roots(&roots).is_zero());
        assert!(poly_from_roots_parallel(&roots).is_zero());

        // Polynomial (x - 2), which has coefficients [-2, 1]
        let roots = vec![Fr::from(2)];
        let poly = poly_from_roots(&roots);
        assert_eq!(poly.degree(), 1, "Polynomial degree should be 1");
        assert_eq!(poly.coeffs[0], Fr::from(-2), "Constant coefficient should be -2");
        assert_eq!(poly.coeffs[1], Fr::one(), "Linear coefficient should be 1");
        assert_eq!(poly, poly_from_roots_parallel(&roots));

        // Polynomial (x - 1)(x - 2) = x^2 - 3x + 2
        // Coefficients: [2, -3, 1]
        let roots = vec![Fr::from(1), Fr::from(2)];
        let poly = poly_from_roots(&roots);
        assert_eq!(poly.degree(), 2, "Polynomial degree should be 2");
        assert_eq!(poly.coeffs[0], Fr::from(2), "Constant coefficient should be 2");
        assert_eq!(poly.coeffs[1], Fr::from(-3), "Linear coefficient should be -3");
        assert_eq!(poly.coeffs[2], Fr::one(), "Quadratic coefficient should be 1");
        assert_eq!(poly, poly_from_roots_parallel(&roots));

        // Test with random roots
        let mut rng = StdRng::from_entropy();
        for count in [20, 50, 100, 200, 300, 400, 500] {
            let roots: Vec<Fr> = (0..count).map(|_| Fr::rand(&mut rng)).collect();
            let start = Instant::now();
            let poly = poly_from_roots(&roots);
            println!("poly_from_roots with {count} roots: {:?}", start.elapsed());
            let start = Instant::now();
            let poly_naive = poly_from_roots_parallel(&roots);
            println!("naive poly_from_roots with {count} roots: {:?}", start.elapsed());

            assert_eq!(poly.degree(), count);
            assert_eq!(poly, poly_naive);

            for root in &roots {
                assert_eq!(poly.evaluate(root), Fr::zero());
            }
        }
    }
}

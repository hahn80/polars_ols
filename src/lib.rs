mod expressions;
pub mod least_squares;

use pyo3::types::PyModule;
use pyo3::{pymodule, PyResult, Python};

#[cfg(test)]
mod tests {
    use crate::expressions::convert_polars_to_ndarray;
    use crate::least_squares::{solve_ols_qr, solve_ridge, solve_elastic_net, solve_recursive_least_squares, solve_rolling_ols};
    use ndarray::prelude::*;
    use ndarray_linalg::assert_close_l2;
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use polars::datatypes::DataType::Float32;
    use polars::prelude::*;

    fn make_data() -> (Series, Series, Series) {
        let x1 = Series::from_vec(
            "x1",
            Array::random(10_000, Normal::new(0., 1.).unwrap()).to_vec(),
        )
        .cast(&Float32)
        .unwrap();
        let x2 = Series::from_vec(
            "x2",
            Array::random(10_000, Normal::new(0., 1.).unwrap()).to_vec(),
        )
        .cast(&Float32)
        .unwrap();
        let y = (&x1 + &x2).with_name("y");
        (y, x1, x2)
    }

    #[test]
    fn test_ols_qr() {
        let (y, x1, x2) = make_data();
        let (targets, features) = convert_polars_to_ndarray(&[y.clone(), x1, x2]);
        let coefficients = solve_ols_qr(&targets, &features);
        let expected = array![1., 1.];
        assert_close_l2!(&coefficients, &expected, 0.001);
    }

    #[test]
    fn test_ridge() {
        let (y, x1, x2) = make_data();
        let (targets, features) = convert_polars_to_ndarray(&[y.clone(), x1, x2]);
        let coefficients = solve_ridge(&targets, &features, 1_000.0);
        let expected = array![0.999, 0.999];
        assert_close_l2!(&coefficients, &expected, 0.001);
    }

    #[test]
    fn test_elastic_net() {
        let (y, x1, x2) = make_data();
        let (targets, features) = convert_polars_to_ndarray(&[y.clone(), x1, x2]);
        let coefficients = solve_elastic_net(&targets, &features, 0.1, Some(0.5), None, None, None);
        let expected = array![0.999, 0.999];
        assert_close_l2!(&coefficients, &expected, 0.001);
    }

    #[test]
    fn test_recursive_least_squares() {
        let (y, x1, x2) = make_data();
        let (targets, features) = convert_polars_to_ndarray(
            &[y.clone(), x1, x2]);
        let coefficients = solve_recursive_least_squares(
            &targets, &features, Some(252.0), Some(0.01), None);
        let expected = array![1.0, 1.0];
        println!("{:?}", coefficients.slice(s![0, ..]));
        println!("{:?}", coefficients.slice(s![-1, ..]));
        assert_close_l2!(&coefficients.slice(s![-1, ..]), &expected, 0.0001);
    }

    #[test]
    fn test_rolling_least_squares() {
        let (y, x1, x2) = make_data();
        let (targets, features) = convert_polars_to_ndarray
            (&[y.clone(), x1, x2]);
        let coefficients = solve_rolling_ols(&targets, &features,
                                             1_000usize,
                                             Some(100usize), Some(false));
        let expected: Array1<f32> = array![1.0, 1.0];
        println!("{:?}", coefficients.slice(s![0, ..]));
        println!("{:?}", coefficients.slice(s![-1, ..]));
        assert_close_l2!(&coefficients.slice(s![-1, ..]), &expected, 0.0001);
    }

}

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

#[pymodule]
#[pyo3(name = "polars_ols")]
fn _internal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

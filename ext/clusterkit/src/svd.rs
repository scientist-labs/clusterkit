use magnus::{function, prelude::*, Error, Value, RArray, Ruby};
use annembed::tools::svdapprox::{SvdApprox, RangeApproxMode, RangeRank, MatRepr};
use crate::utils::ruby_array_to_ndarray2;

pub fn init(parent: &magnus::RModule) -> Result<(), Error> {
    let svd_module = parent.define_module("SVD")?;

    svd_module.define_singleton_method(
        "randomized_svd_rust",
        function!(randomized_svd, 3),
    )?;

    Ok(())
}

fn randomized_svd(matrix: Value, k: usize, n_iter: usize) -> Result<RArray, Error> {
    let ruby = Ruby::get().unwrap();

    // Convert Ruby array to ndarray using shared helper
    let matrix_data = ruby_array_to_ndarray2(matrix)?;
    let (n_rows, n_cols) = matrix_data.dim();

    if k > n_rows.min(n_cols) {
        return Err(Error::new(
            ruby.exception_arg_error(),
            format!("k ({}) cannot be larger than min(rows, cols) = {}", k, n_rows.min(n_cols)),
        ));
    }

    // Create MatRepr for the full matrix
    let mat_repr = MatRepr::from_array2(matrix_data.clone());

    // Create SvdApprox instance
    let mut svd_approx = SvdApprox::new(&mat_repr);

    // Set up parameters for randomized SVD
    let params = RangeApproxMode::RANK(RangeRank::new(k, n_iter));

    // Perform SVD
    let svd_result = svd_approx.direct_svd(params)
        .map_err(|e| Error::new(ruby.exception_runtime_error(), e))?;

    // Extract U, S, V from the result
    let u_matrix = svd_result.u.ok_or_else(|| {
        Error::new(ruby.exception_runtime_error(), "No U matrix in SVD result")
    })?;

    let s_values = svd_result.s.ok_or_else(|| {
        Error::new(ruby.exception_runtime_error(), "No S values in SVD result")
    })?;

    let vt_matrix = svd_result.vt.ok_or_else(|| {
        Error::new(ruby.exception_runtime_error(), "No V^T matrix in SVD result")
    })?;

    // Convert results to Ruby arrays
    let u_ruby = ruby.ary_new();
    let u_shape = u_matrix.shape();
    for i in 0..u_shape[0] {
        let row = ruby.ary_new();
        for j in 0..u_shape[1] {
            row.push(u_matrix[[i, j]])?;
        }
        u_ruby.push(row)?;
    }

    let s_ruby = ruby.ary_new();
    for val in s_values.iter() {
        s_ruby.push(*val)?;
    }

    let v_ruby = ruby.ary_new();
    let vt_shape = vt_matrix.shape();
    for i in 0..vt_shape[0] {
        let row = ruby.ary_new();
        for j in 0..vt_shape[1] {
            row.push(vt_matrix[[i, j]])?;
        }
        v_ruby.push(row)?;
    }

    // Return [U, S, V^T] as a Ruby array
    let result = ruby.ary_new();
    result.push(u_ruby)?;
    result.push(s_ruby)?;
    result.push(v_ruby)?;

    Ok(result)
}

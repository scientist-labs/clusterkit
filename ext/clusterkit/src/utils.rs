use magnus::{function, prelude::*, Error, Value, RArray, TryConvert, Float, Integer, Ruby};
use ndarray::Array2;

pub fn init(parent: &magnus::RModule) -> Result<(), Error> {
    let utils_module = parent.define_module("Utils")?;

    utils_module.define_singleton_method(
        "estimate_intrinsic_dimension_rust",
        function!(estimate_intrinsic_dimension, 2),
    )?;

    utils_module.define_singleton_method(
        "estimate_hubness_rust",
        function!(estimate_hubness, 1),
    )?;

    Ok(())
}

fn estimate_intrinsic_dimension(_data: Value, _k_neighbors: usize) -> Result<f64, Error> {
    let ruby = Ruby::get().unwrap();
    Err(Error::new(
        ruby.exception_not_imp_error(),
        "Dimension estimation not implemented yet",
    ))
}

fn estimate_hubness(_data: Value) -> Result<Value, Error> {
    let ruby = Ruby::get().unwrap();
    Err(Error::new(
        ruby.exception_not_imp_error(),
        "Hubness estimation not implemented yet",
    ))
}

/// Convert Ruby 2D array to ndarray Array2<f64>
/// Handles validation and provides consistent error messages
pub fn ruby_array_to_ndarray2(data: Value) -> Result<Array2<f64>, Error> {
    let ruby = Ruby::get().unwrap();
    let rarray: RArray = TryConvert::try_convert(data)?;
    let n_samples = rarray.len();

    if n_samples == 0 {
        return Err(Error::new(
            ruby.exception_arg_error(),
            "Data cannot be empty",
        ));
    }

    // Get dimensions from first row
    let first_row: RArray = rarray.entry::<RArray>(0)?;
    let n_features = first_row.len();

    if n_features == 0 {
        return Err(Error::new(
            ruby.exception_arg_error(),
            "Data rows cannot be empty",
        ));
    }

    // Create ndarray and populate
    let mut data_array = Array2::<f64>::zeros((n_samples, n_features));
    for i in 0..n_samples {
        let row: RArray = rarray.entry(i as isize)?;

        // Validate row length consistency
        if row.len() != n_features {
            return Err(Error::new(
                ruby.exception_arg_error(),
                format!("Row {} has {} elements, expected {}", i, row.len(), n_features),
            ));
        }

        for j in 0..n_features {
            let val: f64 = row.entry(j as isize)?;
            data_array[[i, j]] = val;
        }
    }

    Ok(data_array)
}

/// Convert Ruby 2D array to Vec<Vec<f64>>
/// Handles validation and provides consistent error messages
pub fn ruby_array_to_vec_vec_f64(data: Value) -> Result<Vec<Vec<f64>>, Error> {
    let ruby = Ruby::get().unwrap();
    let rarray: RArray = TryConvert::try_convert(data)?;
    let n_samples = rarray.len();

    if n_samples == 0 {
        return Err(Error::new(
            ruby.exception_arg_error(),
            "Data cannot be empty",
        ));
    }

    let mut data_vec: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut expected_features: Option<usize> = None;

    for i in 0..n_samples {
        let row: RArray = rarray.entry(i as isize)?;
        let n_features = row.len();

        // Check row length consistency
        match expected_features {
            Some(expected) => {
                if n_features != expected {
                    return Err(Error::new(
                        ruby.exception_arg_error(),
                        format!("Row {} has {} elements, expected {}", i, n_features, expected),
                    ));
                }
            }
            None => expected_features = Some(n_features),
        }

        let mut row_vec: Vec<f64> = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let val: f64 = row.entry(j as isize)?;
            row_vec.push(val);
        }
        data_vec.push(row_vec);
    }

    Ok(data_vec)
}

/// Convert Ruby 2D array to Vec<Vec<f32>>
/// For algorithms that require f32 precision (like UMAP)
pub fn ruby_array_to_vec_vec_f32(data: Value) -> Result<Vec<Vec<f32>>, Error> {
    let ruby = Ruby::get().unwrap();
    let rarray: RArray = TryConvert::try_convert(data)?;
    let array_len = rarray.len();

    if array_len == 0 {
        return Err(Error::new(
            ruby.exception_arg_error(),
            "Input data cannot be empty",
        ));
    }

    let mut rust_data: Vec<Vec<f32>> = Vec::with_capacity(array_len);

    for i in 0..array_len {
        let row = rarray.entry::<Value>(i as isize)?;
        let row_array = RArray::try_convert(row).map_err(|_| {
            Error::new(
                ruby.exception_type_error(),
                "Expected array of arrays (2D array)",
            )
        })?;

        let mut rust_row: Vec<f32> = Vec::new();
        let row_len = row_array.len();

        for j in 0..row_len {
            let val = row_array.entry::<Value>(j as isize)?;
            let float_val = if let Ok(f) = Float::try_convert(val) {
                f.to_f64() as f32
            } else if let Ok(i) = Integer::try_convert(val) {
                i.to_i64()? as f32
            } else {
                return Err(Error::new(
                    ruby.exception_type_error(),
                    "All values must be numeric",
                ));
            };
            rust_row.push(float_val);
        }

        // Validate row length consistency
        if !rust_data.is_empty() && rust_row.len() != rust_data[0].len() {
            return Err(Error::new(
                ruby.exception_arg_error(),
                "All rows must have the same length",
            ));
        }

        rust_data.push(rust_row);
    }

    Ok(rust_data)
}

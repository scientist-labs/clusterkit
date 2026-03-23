use magnus::{function, prelude::*, Error, Value, RHash, Ruby};
use hdbscan::{Hdbscan, HdbscanHyperParams};
use crate::utils::ruby_array_to_vec_vec_f64;

/// Perform HDBSCAN clustering
/// Returns a hash with labels and basic statistics
pub fn hdbscan_fit(
    data: Value,
    min_samples: usize,
    min_cluster_size: usize,
    metric: String,
) -> Result<RHash, Error> {
    let ruby = Ruby::get().unwrap();

    // Convert Ruby array to Vec<Vec<f64>> using shared helper
    let data_vec = ruby_array_to_vec_vec_f64(data)?;
    let n_samples = data_vec.len();

    if metric != "euclidean" && metric != "l2" {
        eprintln!("Warning: Current hdbscan version only supports Euclidean distance. Using Euclidean.");
    }

    // Adjust parameters to avoid index out of bounds errors
    let adjusted_min_samples = min_samples.min(n_samples.saturating_sub(1)).max(1);
    let adjusted_min_cluster_size = min_cluster_size.min(n_samples).max(2);

    // Create hyperparameters
    let hyper_params = HdbscanHyperParams::builder()
        .min_cluster_size(adjusted_min_cluster_size)
        .min_samples(adjusted_min_samples)
        .build();

    // Create HDBSCAN instance and run clustering
    let clusterer = Hdbscan::new(&data_vec, hyper_params);

    let labels = clusterer.cluster().map_err(|e| {
        Error::new(
            ruby.exception_runtime_error(),
            format!("HDBSCAN clustering failed: {:?}", e)
        )
    })?;

    // Convert results to Ruby types
    let result = ruby.hash_new();

    let labels_array = ruby.ary_new();
    for &label in labels.iter() {
        labels_array.push(ruby.integer_from_i64(label as i64))?;
    }
    result.aset("labels", labels_array)?;

    let probs_array = ruby.ary_new();
    for &label in labels.iter() {
        let prob = if label == -1 { 0.0 } else { 1.0 };
        probs_array.push(prob)?;
    }
    result.aset("probabilities", probs_array)?;

    let outlier_array = ruby.ary_new();
    for &label in labels.iter() {
        let score = if label == -1 { 1.0 } else { 0.0 };
        outlier_array.push(score)?;
    }
    result.aset("outlier_scores", outlier_array)?;

    let persistence_hash = ruby.hash_new();
    result.aset("cluster_persistence", persistence_hash)?;

    Ok(result)
}

/// Initialize HDBSCAN module functions
pub fn init(clustering_module: &magnus::RModule) -> Result<(), Error> {
    clustering_module.define_singleton_method(
        "hdbscan_rust",
        function!(hdbscan_fit, 4),
    )?;

    Ok(())
}

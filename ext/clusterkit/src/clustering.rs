use magnus::{function, prelude::*, Error, Value, RArray, Ruby};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use crate::utils::ruby_array_to_ndarray2;

mod hdbscan_wrapper;

pub fn init(parent: &magnus::RModule) -> Result<(), Error> {
    let clustering_module = parent.define_module("Clustering")?;

    clustering_module.define_singleton_method(
        "kmeans_rust",
        function!(kmeans, 4),
    )?;

    clustering_module.define_singleton_method(
        "kmeans_predict_rust",
        function!(kmeans_predict, 2),
    )?;

    // Initialize HDBSCAN functions
    hdbscan_wrapper::init(&clustering_module)?;

    Ok(())
}

/// Perform K-means clustering
/// Returns (labels, centroids, inertia)
fn kmeans(data: Value, k: usize, max_iter: usize, random_seed: Option<i64>) -> Result<(RArray, RArray, f64), Error> {
    let ruby = Ruby::get().unwrap();

    // Convert Ruby array to ndarray using shared helper
    let data_array = ruby_array_to_ndarray2(data)?;
    let (n_samples, n_features) = data_array.dim();

    if k > n_samples {
        return Err(Error::new(
            ruby.exception_arg_error(),
            format!("k ({}) cannot be larger than number of samples ({})", k, n_samples),
        ));
    }

    // Initialize centroids using K-means++
    let mut centroids = kmeans_plusplus(&data_array, k, random_seed)?;
    let mut labels = vec![0usize; n_samples];
    let mut prev_labels = vec![0usize; n_samples];

    // K-means iterations
    for iteration in 0..max_iter {
        // Assign points to nearest centroid
        let mut changed = false;
        for i in 0..n_samples {
            let point = data_array.row(i);
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;

            for (j, centroid) in centroids.axis_iter(Axis(0)).enumerate() {
                let dist = euclidean_distance(&point, &centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }

            if labels[i] != best_cluster {
                changed = true;
            }
            labels[i] = best_cluster;
        }

        // Check for convergence
        if !changed && iteration > 0 {
            break;
        }

        // Update centroids
        for j in 0..k {
            let mut sum = Array1::<f64>::zeros(n_features);
            let mut count = 0;

            for i in 0..n_samples {
                if labels[i] == j {
                    sum += &data_array.row(i);
                    count += 1;
                }
            }

            if count > 0 {
                centroids.row_mut(j).assign(&(sum / count as f64));
            }
        }

        prev_labels.clone_from(&labels);
    }

    // Calculate inertia (sum of squared distances to nearest centroid)
    let mut inertia = 0.0;
    for i in 0..n_samples {
        let point = data_array.row(i);
        let centroid = centroids.row(labels[i]);
        inertia += euclidean_distance(&point, &centroid).powi(2);
    }

    // Convert results to Ruby arrays
    let labels_array = ruby.ary_new();
    for label in labels {
        labels_array.push(ruby.integer_from_i64(label as i64))?;
    }

    let centroids_array = ruby.ary_new();
    for i in 0..k {
        let row_array = ruby.ary_new();
        for j in 0..n_features {
            row_array.push(centroids[[i, j]])?;
        }
        centroids_array.push(row_array)?;
    }

    Ok((labels_array, centroids_array, inertia))
}

/// Predict cluster labels for new data given centroids
fn kmeans_predict(data: Value, centroids: Value) -> Result<RArray, Error> {
    let ruby = Ruby::get().unwrap();

    // Convert inputs using shared helpers
    let data_matrix = ruby_array_to_ndarray2(data)?;
    let centroids_matrix = ruby_array_to_ndarray2(centroids)?;

    let (n_samples, _) = data_matrix.dim();

    // Predict labels
    let labels_array = ruby.ary_new();

    for i in 0..n_samples {
        let point = data_matrix.row(i);
        let mut min_dist = f64::INFINITY;
        let mut best_cluster = 0;

        for (j, centroid) in centroids_matrix.axis_iter(Axis(0)).enumerate() {
            let dist = euclidean_distance(&point, &centroid);
            if dist < min_dist {
                min_dist = dist;
                best_cluster = j;
            }
        }

        labels_array.push(ruby.integer_from_i64(best_cluster as i64))?;
    }

    Ok(labels_array)
}

/// K-means++ initialization
fn kmeans_plusplus(data: &Array2<f64>, k: usize, random_seed: Option<i64>) -> Result<Array2<f64>, Error> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    // Use seeded RNG if seed is provided, otherwise use thread_rng
    let mut rng: Box<dyn RngCore> = match random_seed {
        Some(seed) => {
            let seed_u64 = seed as u64;
            Box::new(StdRng::seed_from_u64(seed_u64))
        },
        None => Box::new(thread_rng()),
    };

    let mut centroids = Array2::<f64>::zeros((k, n_features));

    // Choose first centroid randomly
    let first_idx = rng.gen_range(0..n_samples);
    centroids.row_mut(0).assign(&data.row(first_idx));

    // Choose remaining centroids
    for i in 1..k {
        let mut distances = vec![f64::INFINITY; n_samples];

        for j in 0..n_samples {
            for c in 0..i {
                let dist = euclidean_distance(&data.row(j), &centroids.row(c));
                if dist < distances[j] {
                    distances[j] = dist;
                }
            }
        }

        let total: f64 = distances.iter().map(|d| d * d).sum();
        if total == 0.0 {
            if i < n_samples {
                centroids.row_mut(i).assign(&data.row(i));
            } else {
                centroids.row_mut(i).assign(&data.row(0));
            }
            continue;
        }

        let mut cumsum = 0.0;
        let rand_val: f64 = rng.gen::<f64>() * total;

        for j in 0..n_samples {
            cumsum += distances[j] * distances[j];
            if cumsum >= rand_val {
                centroids.row_mut(i).assign(&data.row(j));
                break;
            }
        }
    }

    Ok(centroids)
}

/// Calculate Euclidean distance between two points
fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

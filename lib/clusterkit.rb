# frozen_string_literal: true

require_relative "clusterkit/version"

# Load the native extension. Precompiled platform gems install one .so/.bundle per
# Ruby ABI under lib/clusterkit/<major.minor>/, while a source build (and the native
# darwin gem) produce the flat lib/clusterkit/clusterkit.{so,bundle}. Try the
# versioned path first, then fall back to flat. Use `require` (not
# `require_relative`) so RubyGems can resolve the extension through $LOAD_PATH when
# it installs native artifacts outside lib/.
begin
  RUBY_VERSION =~ /(\d+\.\d+)/
  require "clusterkit/#{Regexp.last_match(1)}/clusterkit"
rescue LoadError
  require "clusterkit/clusterkit"
end

require_relative "clusterkit/configuration"

# Main module for ClusterKit gem
# Provides high-performance dimensionality reduction algorithms
module ClusterKit
  class Error < StandardError; end


  # Core error classes
  class DimensionError < Error; end
  class ConvergenceError < Error; end
  class InvalidParameterError < Error; end
  
  # Data-related errors
  class DataError < Error; end
  class IsolatedPointError < DataError; end
  class DisconnectedGraphError < DataError; end
  class InsufficientDataError < DataError; end

  # Autoload utilities
  autoload :Utils, "clusterkit/utils"
  autoload :Preprocessing, "clusterkit/preprocessing"
  autoload :Silence, "clusterkit/silence"
  
  # Load modules that depend on the extension
  require_relative "clusterkit/dimensionality"
  require_relative "clusterkit/clustering"
  require_relative "clusterkit/hnsw"
  
  # Make RustUMAP private - it's an implementation detail
  # Users should use Dimensionality::UMAP instead
  private_constant :RustUMAP if const_defined?(:RustUMAP)

  class << self
    # Quick UMAP embedding
    # @param data [Array] Input data
    # @param n_components [Integer] Number of dimensions in output
    # @return [Array] Embedded data
    def umap(data, n_components: 2, **options)
      umap = Dimensionality::UMAP.new(n_components: n_components, **options)
      umap.fit_transform(data)
    end
    
    # Quick PCA
    # @param data [Array] Input data
    # @param n_components [Integer] Number of dimensions in output
    # @return [Array] Transformed data
    def pca(data, n_components: 2)
      pca = Dimensionality::PCA.new(n_components: n_components)
      pca.fit_transform(data)
    end

    # t-SNE is not yet implemented
    # @deprecated Not implemented - use UMAP instead
    def tsne(data, n_components: 2, **options)
      raise NotImplementedError, "t-SNE is not yet implemented. Please use UMAP instead, which provides similar dimensionality reduction capabilities."
    end

    # Estimate intrinsic dimension of data
    # @param data [Array, Numo::NArray] Input data
    # @param k [Integer] Number of neighbors to consider
    # @return [Float] Estimated intrinsic dimension
    def estimate_dimension(data, k: 10)
      Utils.estimate_intrinsic_dimension(data, k_neighbors: k)
    end

    # Perform SVD
    # @param matrix [Array] Input matrix
    # @param k [Integer] Number of components
    # @param n_iter [Integer] Number of iterations for randomized algorithm
    # @return [Array] U, S, V matrices
    def svd(matrix, k, n_iter: 2)
      svd = Dimensionality::SVD.new(n_components: k, n_iter: n_iter)
      svd.fit_transform(matrix)
    end
    
    # Quick K-means with automatic k detection
    # @param data [Array] Input data
    # @param k [Integer, nil] Number of clusters (auto-detect if nil)
    # @param k_range [Range] Range for auto-detection
    # @return [Array] Cluster labels
    def kmeans(data, k: nil, k_range: 2..10, **options)
      k ||= Clustering::KMeans.optimal_k(data, k_range: k_range)
      kmeans = Clustering::KMeans.new(k: k, **options)
      kmeans.fit_predict(data)
    end
  end
end
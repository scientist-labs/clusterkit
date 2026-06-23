require 'spec_helper'
require 'fileutils'

RSpec.describe ClusterKit::HNSW do
  describe '#initialize' do
    it 'creates an index with required dimension' do
      index = described_class.new(dim: 10)
      expect(index).to be_a(described_class)
      expect(index.config[:dim]).to eq(10)
    end

    it 'accepts optional parameters' do
      index = described_class.new(
        dim: 5,
        space: :euclidean,  # Only euclidean currently supported
        max_elements: 1000,
        m: 32,
        ef_construction: 400,
        random_seed: 42
      )
      expect(index.config[:space]).to eq('euclidean')
      expect(index.config[:dim]).to eq(5)
    end

    it 'raises error for invalid dimension' do
      # Note: Rust currently accepts 0 but rejects negative
      expect { described_class.new(dim: -1) }.to raise_error(ArgumentError)
    end

    it 'raises error for invalid space' do
      expect { described_class.new(dim: 10, space: :invalid) }.to raise_error(ArgumentError, /space must be/)
    end
  end

  describe '#add_item' do
    let(:index) { described_class.new(dim: 3) }

    it 'adds a single vector' do
      index.add_item([1.0, 2.0, 3.0], {})
      expect(index.size).to eq(1)
    end

    it 'adds a vector with a label' do
      index.add_item([1.0, 2.0, 3.0], label: 'test_item')
      results = index.search([1.0, 2.0, 3.0], k: 1)
      expect(results).to include('test_item')
    end

    it 'adds a vector with metadata' do
      metadata = { category: 'test', score: '0.95' }
      index.add_item([1.0, 2.0, 3.0], label: 'item1', metadata: metadata)
      results = index.search_with_metadata([1.0, 2.0, 3.0], k: 1)
      expect(results.first[:label]).to eq('item1')
      # Metadata values are converted to strings in Rust
      expect(results.first[:metadata]).to eq({ 'category' => 'test', 'score' => '0.95' })
    end

    it 'raises error for wrong dimension' do
      expect { index.add_item([1.0, 2.0], {}) }.to raise_error(ArgumentError, /dimension mismatch/)
      expect { index.add_item([1.0, 2.0, 3.0, 4.0], {}) }.to raise_error(ArgumentError, /dimension mismatch/)
    end

    it 'raises error for duplicate labels' do
      index.add_item([1.0, 2.0, 3.0], label: 'dup')
      expect { index.add_item([4.0, 5.0, 6.0], label: 'dup') }.to raise_error(ArgumentError, /already exists/)
    end
  end

  describe '#add_batch' do
    let(:index) { described_class.new(dim: 2) }
    let(:vectors) { [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] }

    it 'adds multiple vectors' do
      index.add_batch(vectors, {})
      expect(index.size).to eq(3)
    end

    it 'adds vectors with labels' do
      labels = ['a', 'b', 'c']
      index.add_batch(vectors, labels: labels)
      results = index.search([1.0, 2.0], k: 3)
      expect(results).to match_array(labels)
    end

    it 'processes in parallel by default' do
      large_vectors = Array.new(100) { [rand, rand] }
      index.add_batch(large_vectors, {})
      expect(index.size).to eq(100)
    end

    it 'can process sequentially' do
      index.add_batch(vectors, parallel: false)
      expect(index.size).to eq(3)
    end
  end

  describe '#search' do
    let(:index) do
      idx = described_class.new(dim: 2)
      idx.add_batch([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [10.0, 10.0]], 
                    labels: ['a', 'b', 'c', 'd'])
      idx
    end

    it 'finds k nearest neighbors' do
      results = index.search([1.5, 1.5], k: 2)
      expect(results.size).to eq(2)
      expect(results).to include('a', 'b')
    end

    it 'returns distances when requested' do
      indices, distances = index.search([1.0, 1.0], k: 2, include_distances: true)
      expect(indices.size).to eq(2)
      expect(distances.size).to eq(2)
      expect(distances.first).to eq(0.0)  # Exact match
    end

    it 'respects ef parameter for search quality' do
      results_low = index.search([2.0, 2.0], k: 3, ef: 10)
      results_high = index.search([2.0, 2.0], k: 3, ef: 100)
      # Both should return results
      expect(results_low.size).to eq(3)
      expect(results_high.size).to eq(3)
    end

    it 'handles k larger than index size' do
      # Use ef: 50 to ensure comprehensive graph traversal in a tiny 4-item index.
      # Without a generous ef the HNSW search graph may not visit all nodes,
      # causing a non-deterministic result < index_size on Ruby 3.3.
      results = index.search([1.0, 1.0], k: 10, ef: 50)
      expect(results.size).to eq(4)  # Only 4 items in index
    end
  end

  describe '#search_with_metadata' do
    let(:index) do
      idx = described_class.new(dim: 2)
      idx.add_item([1.0, 1.0], label: 'a', metadata: { type: 'first' })
      idx.add_item([2.0, 2.0], label: 'b', metadata: { type: 'second' })
      idx
    end

    it 'returns results with metadata' do
      results = index.search_with_metadata([1.0, 1.0], k: 2)
      expect(results).to be_an(Array)
      expect(results.first).to have_key(:label)
      expect(results.first).to have_key(:distance)
      expect(results.first).to have_key(:metadata)
    end
  end

  describe '#knn_query' do
    let(:index) do
      idx = described_class.new(dim: 2)
      idx.add_batch([[1.0, 1.0], [2.0, 2.0]], labels: ['a', 'b'])
      idx
    end

    it 'is an alias for search with distances' do
      indices, distances = index.knn_query([1.5, 1.5], k: 2)
      expect(indices).to be_an(Array)
      expect(distances).to be_an(Array)
    end
  end

  describe '#size and #empty?' do
    let(:index) { described_class.new(dim: 2) }

    it 'reports correct size' do
      expect(index.size).to eq(0)
      expect(index.empty?).to be true

      index.add_item([1.0, 2.0], {})
      expect(index.size).to eq(1)
      expect(index.empty?).to be false
    end
  end

  describe '#set_ef' do
    let(:index) { described_class.new(dim: 2) }

    it 'updates search ef parameter' do
      index.set_ef(100)
      expect(index.config[:ef]).to eq(100)
    end
  end

  describe '#config and #stats' do
    let(:index) { described_class.new(dim: 3, space: :euclidean) }

    it 'returns configuration' do
      config = index.config
      expect(config[:dim]).to eq(3)
      expect(config[:space]).to eq('euclidean')
    end

    it 'returns statistics' do
      index.add_batch([[1, 2, 3], [4, 5, 6]], labels: ['a', 'b'])
      stats = index.stats
      expect(stats[:size]).to eq(2)
      expect(stats[:dim]).to eq(3)
    end
  end

  describe 'seeded construction' do
    it 'produces consistent graph structure with seed' do
      # Generate deterministic test data with well-separated points
      # to avoid ties in distance calculations
      vectors = []
      labels = []
      
      # Create a grid of well-separated points
      5.times do |i|
        10.times do |j|
          # Points are spaced far apart to avoid distance ties
          vector = Array.new(10) { |k| i * 10.0 + j * 0.5 + k * 0.01 }
          vectors << vector
          labels << "#{i}_#{j}"
        end
      end
      
      # Build two indices with the same seed
      index1 = described_class.new(dim: 10, random_seed: 42, m: 8, ef_construction: 100)
      index1.add_batch(vectors, labels: labels)
      
      index2 = described_class.new(dim: 10, random_seed: 42, m: 8, ef_construction: 100)
      index2.add_batch(vectors, labels: labels)
      
      # Test multiple queries to verify consistent behavior
      consistent = true
      5.times do |q|
        query = Array.new(10) { |i| q * 5.0 + i * 0.3 }
        results1 = index1.search(query, k: 3)
        results2 = index2.search(query, k: 3)
        
        # At minimum, the nearest neighbor should be the same
        if results1.first != results2.first
          consistent = false
          break
        end
      end
      
      expect(consistent).to be true
    end
  end

  describe 'integration with Ruby types' do
    let(:index) { described_class.new(dim: 3) }

    it 'accepts Ruby arrays' do
      index.add_item([1, 2, 3], {})
      expect(index.size).to eq(1)
    end
    
    # TODO: Implement Numo::NArray support
    # Future feature: accept Numo arrays directly without conversion
  end

  describe '#save and .load' do
    let(:index) do
      idx = described_class.new(dim: 2, random_seed: 42)
      # Use more points to ensure a stable HNSW graph
      vectors = [
        [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0],
        [1.5, 1.5], [2.5, 2.5], [3.5, 3.5], [4.5, 4.5],
        [1.0, 2.0], [2.0, 3.0]
      ]
      labels = ('a'..'j').to_a
      idx.add_batch(vectors, labels: labels)
      idx
    end

    it 'saves index to file' do
      path = '/tmp/test_hnsw'
      index.save(path)
      expect(File.exist?("#{path}.metadata")).to be true
      expect(Dir.exist?("#{path}_hnsw_data")).to be true
      expect(File.exist?("#{path}_hnsw_data/hnsw.hnsw.data")).to be true
      expect(File.exist?("#{path}_hnsw_data/hnsw.hnsw.graph")).to be true
      
      # Clean up
      File.delete("#{path}.metadata") if File.exist?("#{path}.metadata")
      FileUtils.rm_rf("#{path}_hnsw_data") if Dir.exist?("#{path}_hnsw_data")
    end

    it 'loads index from file' do
      path = '/tmp/test_hnsw'
      index.save(path)
      
      loaded = described_class.load(path)
      expect(loaded.size).to eq(10)
      
      # Test that loaded index works - search for nearest neighbors
      results = loaded.search([1.0, 1.0], k: 3)
      expect(results.size).to eq(3)
      # The exact nearest point should always be found
      expect(results.first).to eq('a')
      
      # Clean up  
      File.delete("#{path}.metadata") if File.exist?("#{path}.metadata")
      FileUtils.rm_rf("#{path}_hnsw_data") if Dir.exist?("#{path}_hnsw_data")
    end
  end
end
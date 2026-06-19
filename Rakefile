require "bundler/gem_tasks"
require "rake/extensiontask"

# rspec is a DEVELOPMENT-only dependency. The cross-compile build container
# (rb-sys-dock, via scientist-labs/rust-gem-release) installs the runtime bundle
# only, so this require would raise LoadError and abort `rake` before the native
# build task can run. Guard it so this Rakefile always loads; the spec task simply
# isn't available in a build-only environment.
begin
  require "rspec/core/rake_task"
  RSpec::Core::RakeTask.new(:spec)
rescue LoadError
  desc "run specs (rspec unavailable in this environment)"
  task(:spec) { abort "rspec is a development dependency and is not installed here" }
end

# Define the Rust extension. Passing the loaded gemspec to ExtensionTask makes
# rake-compiler expose the `native:<platform>` tasks rb-sys-dock invokes for each
# precompiled leg; without it the cross build fails with "Don't know how to build task".
GEMSPEC = Gem::Specification.load("clusterkit.gemspec")
Rake::ExtensionTask.new("clusterkit", GEMSPEC) do |ext|
  ext.lib_dir = "lib/clusterkit"
  ext.source_pattern = "*.{rs,toml}"
  ext.cross_compile = true
  # Union of the precompiled targets this gem ships: both glibc linux arches
  # (assembled by oxidize-rb cross-gem in rb-sys-dock) plus Apple Silicon
  # (built natively on a macOS runner).
  ext.cross_platform = %w[x86_64-linux aarch64-linux arm64-darwin]
end

task default: [:compile, :spec]

# Documentation task
begin
  require "yard"
  YARD::Rake::YardocTask.new do |t|
    t.files = ["lib/**/*.rb"]
    t.options = ["--no-private", "--readme", "README.md"]
  end
rescue LoadError
  desc "YARD documentation task not available"
  task :yard do
    puts "YARD is not available. Please install it with: gem install yard"
  end
end

# Benchmarking task
desc "Run benchmarks"
task :benchmark do
  ruby "test/benchmark/benchmarks.rb"
end

# Console task for interactive testing
desc "Open an interactive console with the gem loaded"
task :console do
  require "irb"
  require "clusterkit"
  ARGV.clear
  IRB.start
end

# Rust-specific tasks
namespace :rust do
  desc "Run cargo fmt"
  task :fmt do
    Dir.chdir("ext/clusterkit") do
      sh "cargo fmt"
    end
  end

  desc "Run cargo clippy"
  task :clippy do
    Dir.chdir("ext/clusterkit") do
      sh "cargo clippy -- -D warnings"
    end
  end

  desc "Run cargo test"
  task :test do
    Dir.chdir("ext/clusterkit") do
      sh "cargo test"
    end
  end
end

# Coverage task
desc "Run specs with code coverage"
task :coverage do
  ENV['COVERAGE'] = 'true'
  Rake::Task["spec"].invoke
end

# Coverage report task
desc "Open coverage report in browser"
task :"coverage:report" => :coverage do
  if RUBY_PLATFORM =~ /darwin/
    sh "open coverage/index.html"
  elsif RUBY_PLATFORM =~ /linux/
    sh "xdg-open coverage/index.html"
  else
    puts "Coverage report generated at coverage/index.html"
  end
end

# CI task that runs all checks
desc "Run all CI checks"
task ci: ["rust:fmt", "rust:clippy", "compile", "spec", "coverage"]

# Load custom rake tasks
Dir.glob('lib/tasks/*.rake').each { |r| load r }

# Test fixture generation
namespace :fixtures do
  desc "Generate real embedding fixtures for tests using red-candle"
  task :generate_embeddings do
    begin
      require 'red-candle'
      require 'json'
      require 'fileutils'
      
      puts "Loading embedding model..."
      # Use a small, fast model for generating test embeddings
      model = Candle::EmbeddingModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
      
      # Create fixtures directory
      fixtures_dir = File.join(__dir__, 'spec', 'fixtures', 'embeddings')
      FileUtils.mkdir_p(fixtures_dir)
      
      # Generate embeddings for different test scenarios
      test_cases = {
        # Basic test set - 15 sentences for general testing
        'basic_15' => [
          "The quick brown fox jumps over the lazy dog.",
          "Machine learning is transforming how we process data.",
          "Ruby is a dynamic programming language.",
          "Natural language processing enables computers to understand text.",
          "The weather today is sunny and warm.",
          "Coffee is a popular morning beverage.",
          "Books are a gateway to knowledge and imagination.",
          "The ocean waves crash against the shore.",
          "Technology continues to evolve rapidly.",
          "Music has the power to evoke emotions.",
          "The mountain peak was covered in snow.",
          "Cooking is both an art and a science.",
          "Exercise is important for maintaining health.",
          "The stars shine brightly in the night sky.",
          "History teaches us valuable lessons."
        ],
        
        # Clustered data - 3 distinct topics for testing clustering
        'clusters_30' => [
          # Technology cluster (10 items)
          "Artificial intelligence is revolutionizing industries.",
          "Python is widely used for data science.",
          "Cloud computing provides scalable infrastructure.",
          "Cybersecurity is crucial for protecting data.",
          "Blockchain technology enables decentralized systems.",
          "Quantum computing may solve complex problems.",
          "APIs enable software integration.",
          "DevOps practices improve deployment efficiency.",
          "Microservices architecture promotes modularity.",
          "Machine learning models require training data.",
          
          # Nature cluster (10 items)
          "The rainforest ecosystem is incredibly diverse.",
          "Mountains are formed by tectonic activity.",
          "Coral reefs support marine biodiversity.",
          "Rivers flow from highlands to the sea.",
          "Deserts have adapted to water scarcity.",
          "Forests produce oxygen and absorb carbon.",
          "The arctic ice is melting due to climate change.",
          "Volcanoes release molten rock from Earth's interior.",
          "Wetlands filter water naturally.",
          "The savanna supports large herbivore populations.",
          
          # Food/Cooking cluster (10 items)
          "Italian cuisine features pasta and tomatoes.",
          "Sushi is a traditional Japanese dish.",
          "Baking bread requires yeast for fermentation.",
          "Spices add flavor and aroma to dishes.",
          "Vegetarian diets exclude meat products.",
          "Wine is produced through grape fermentation.",
          "Chocolate comes from cacao beans.",
          "Grilling imparts a smoky flavor to food.",
          "Fresh herbs enhance culinary creations.",
          "Fermented foods contain beneficial probiotics."
        ],
        
        # Small set for minimum viable dataset testing
        'minimal_6' => [
          "Data science involves statistical analysis.",
          "The sunset painted the sky orange.",
          "Coffee beans are roasted before brewing.",
          "Programming requires logical thinking.",
          "Gardens need water and sunlight.",
          "Music festivals bring people together."
        ],
        
        # Large set for high-dimensional testing
        'large_100' => (1..100).map { |i| "This is test sentence number #{i} with some variation in content." }
      }
      
      puts "Generating embeddings for test cases..."
      test_cases.each do |name, texts|
        puts "  Generating #{name} (#{texts.length} texts)..."
        
        # Generate embeddings
        # Each embedding is a 1x384 tensor, so we need to extract the array
        embeddings_array = texts.map { |text| model.embedding(text).to_a.first.to_a }
        
        # Save as JSON
        output_file = File.join(fixtures_dir, "#{name}.json")
        File.write(output_file, JSON.pretty_generate({
          'description' => "Test embeddings for #{name}",
          'model' => 'sentence-transformers/all-MiniLM-L6-v2',
          'dimension' => embeddings_array.first.length,
          'count' => embeddings_array.length,
          'embeddings' => embeddings_array
        }))
        
        puts "    Saved to #{output_file}"
      end
      
      puts "\nEmbedding fixtures generated successfully!"
      puts "You can now use these in your specs with:"
      puts '  embeddings = JSON.parse(File.read("spec/fixtures/embeddings/basic_15.json"))["embeddings"]'
      
    rescue LoadError
      puts "Error: red-candle gem not found."
      puts "Please run: bundle install --with development"
      exit 1
    rescue => e
      puts "Error generating embeddings: #{e.message}"
      puts e.backtrace.first(5)
      exit 1
    end
  end
  
  desc "List available embedding fixtures"
  task :list do
    require 'json'
    fixtures_dir = File.join(__dir__, 'spec', 'fixtures', 'embeddings')
    if Dir.exist?(fixtures_dir)
      files = Dir.glob(File.join(fixtures_dir, '*.json'))
      if files.empty?
        puts "No embedding fixtures found. Run 'rake fixtures:generate_embeddings' to create them."
      else
        puts "Available embedding fixtures:"
        files.each do |file|
          data = JSON.parse(File.read(file))
          basename = File.basename(file)
          puts "  #{basename}: #{data['count']} embeddings, #{data['dimension']}D"
        end
      end
    else
      puts "Fixtures directory not found. Run 'rake fixtures:generate_embeddings' to create fixtures."
    end
  end
end
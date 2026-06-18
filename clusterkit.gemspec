require_relative "lib/clusterkit/version"

Gem::Specification.new do |spec|
  spec.name = "clusterkit"
  spec.version = ClusterKit::VERSION
  spec.authors = ["Chris Petersen"]
  spec.email = ["chris@petersen.io"]

  spec.summary = "High-performance clustering and dimensionality reduction for Ruby"
  spec.description = "A comprehensive clustering toolkit for Ruby, providing UMAP, PCA, K-means, HDBSCAN and more. Built on top of annembed and hdbscan Rust crates for blazing-fast performance."
  spec.homepage = "https://github.com/scientist-labs/clusterkit"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 2.7.0"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/main/CHANGELOG.md"

  # Specify which files should be added to the gem when it is released.
  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0").reject do |f|
      (f == __FILE__) || f.match(%r{\A(?:(?:bin|test|spec|features)/|\.(?:git|travis|circleci)|appveyor)})
    end + Dir["ext/**/*.rs", "ext/**/*.toml"]
  end
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  # Precompiled platform gems (arm64-darwin built natively on a macOS runner; the
  # x86_64-linux/aarch64-linux gems assembled by rb_sys via oxidize-rb cross-gem) carry
  # one compiled extension per Ruby ABI under lib/clusterkit/<major.minor>/ and must NOT
  # declare extensions, or RubyGems would recompile from Rust source on install —
  # defeating the precompiled gem. .gitignore excludes **.bundle/**.so, so the
  # git-ls-files-based spec.files above drops the native artifacts; the explicit Dir[]
  # globs below re-add them for the platform gem. Unset env => normal source gem.
  if (platform_gem = ENV["RUST_GEM_PLATFORM"])
    spec.platform = platform_gem
    spec.extensions = []
    spec.files += Dir["lib/clusterkit/*/clusterkit.bundle"] + Dir["lib/clusterkit/*/clusterkit.so"]
  else
    spec.extensions = ["ext/clusterkit/extconf.rb"]
  end

  # Runtime dependencies
  # Numo is optional but recommended for better performance
  # spec.add_dependency "numo-narray", "~> 0.9"
  spec.add_dependency "rb_sys", "~> 0.9"

  # Development dependencies
  spec.add_development_dependency "benchmark"
  spec.add_development_dependency "csv"
  spec.add_development_dependency "rake", "~> 13.0"
  spec.add_development_dependency "rake-compiler", "~> 1.2"
  spec.add_development_dependency "rspec", "~> 3.0"
  spec.add_development_dependency "simplecov", "~> 0.22"
  spec.add_development_dependency "yard", "~> 0.9"

  # For more information and examples about making a new gem, check out our
  # guide at: https://bundler.io/guides/creating_gem.html
end

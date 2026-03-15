require "spec_helper"

# These specs verify that the native extension loads correctly via $LOAD_PATH
# resolution (using `require`) rather than relative path resolution
# (using `require_relative`).
#
# Background:
# RubyGems installs native extensions into a separate extensions directory
# (e.g., ~/.gem/ruby/3.4.0/extensions/...) and adds that directory to
# $LOAD_PATH. Using `require_relative` bypasses $LOAD_PATH and looks only
# in the gem's lib/ directory, where the compiled .so/.bundle file does not
# exist. Using `require` resolves via $LOAD_PATH and finds the extension
# in the correct location.

RSpec.describe "Native extension loading" do
  it "loads the ClusterKit module successfully" do
    expect(defined?(ClusterKit)).to eq("constant")
  end

  it "makes ClusterKit::Error available" do
    expect(defined?(ClusterKit::Error)).to eq("constant")
  end

  it "makes ClusterKit::DimensionError available" do
    expect(defined?(ClusterKit::DimensionError)).to eq("constant")
  end

  it "makes ClusterKit::DataError available" do
    expect(defined?(ClusterKit::DataError)).to eq("constant")
  end

  it "can instantiate a UMAP object (proves native extension is functional)" do
    umap = ClusterKit::Dimensionality::UMAP.new(n_components: 2)
    expect(umap).to be_a(ClusterKit::Dimensionality::UMAP)
  end

  it "loads clusterkit/clusterkit via require (not require_relative)" do
    # Verify that lib/clusterkit.rb uses `require` for the native extension.
    # This is critical because RubyGems places compiled extensions in the
    # extensions directory, not in the gem's lib/ directory.
    clusterkit_rb = File.read(File.expand_path("../lib/clusterkit.rb", __dir__))
    expect(clusterkit_rb).to include('require "clusterkit/clusterkit"')
    expect(clusterkit_rb).not_to include('require_relative "clusterkit/clusterkit"')
  end
end

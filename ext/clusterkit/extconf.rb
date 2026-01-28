require "mkmf"
require "rb_sys/mkmf"

create_rust_makefile("clusterkit/clusterkit") do |r|
  if ENV["CLUSTERKIT_FEATURES"]
    r.extra_cargo_args += ["--no-default-features"]
    r.features = ENV["CLUSTERKIT_FEATURES"].split(",")
  elsif RUBY_PLATFORM =~ /darwin/
    r.extra_cargo_args += ["--no-default-features"]
    r.features = ["macos-accelerate"]
  end
end

require "mkmf"
require "rb_sys/mkmf"

create_rust_makefile("clusterkit/clusterkit") do |r|
  if ENV["CLUSTERKIT_FEATURES"]
    # Explicit override wins (set CLUSTERKIT_FEATURES=openblas-static,... to force a backend).
    r.extra_cargo_args += ["--no-default-features"]
    r.features = ENV["CLUSTERKIT_FEATURES"].split(",")
  elsif RUBY_PLATFORM =~ /darwin/
    # macOS links the system Accelerate framework — no OpenBLAS build needed.
    r.extra_cargo_args += ["--no-default-features"]
    r.features = ["macos-accelerate"]
  elsif RUBY_PLATFORM =~ /linux/
    # Linux: link the SYSTEM OpenBLAS/LAPACK (apt: libopenblas-dev liblapack-dev
    # gfortran, provided by the rust-gem-cross image) instead of the default
    # `openblas-static` feature, which compiles OpenBLAS from C+Fortran source.
    # rb-sys-dock does NOT forward host env to extconf and a .cargo/config.toml
    # [env] only reaches cargo-spawned procs (not mkmf), so this backend choice
    # must live in committed code — it cannot be passed via a workflow input.
    r.extra_cargo_args += ["--no-default-features"]
    r.features = ["openblas-system"]
  end
end

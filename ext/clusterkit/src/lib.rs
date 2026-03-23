use magnus::{Error, Ruby};

mod embedder;
mod svd;
mod utils;
mod clustering;
mod hnsw;

#[cfg(test)]
mod tests;

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let module = ruby.define_module("ClusterKit")?;

    // Initialize submodules
    embedder::init(&module)?;
    svd::init(&module)?;
    utils::init(&module)?;
    clustering::init(&module)?;
    hnsw::init(&module)?;

    Ok(())
}
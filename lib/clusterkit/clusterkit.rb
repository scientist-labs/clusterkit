# frozen_string_literal: true

# Load the compiled Rust extension. Precompiled (platform) gems install it into a
# Ruby-ABI-versioned subdir (lib/clusterkit/<major.minor>/clusterkit.{so,bundle}) so a
# single fat gem can carry a binary per Ruby version; source/dev builds place it flat at
# lib/clusterkit/clusterkit.{so,bundle}. Try the versioned path first, fall back to the
# flat one. The versioned path goes through $LOAD_PATH (`require`) because RubyGems may
# install native extensions outside lib/; the flat fallback uses `require_relative` to
# directly load the bundle alongside this file, avoiding a circular `require`.
begin
  RUBY_VERSION =~ /(\d+\.\d+)/
  require "clusterkit/#{Regexp.last_match(1)}/clusterkit"
rescue LoadError
  begin
    require_relative "clusterkit.bundle"
  rescue LoadError
    require_relative "clusterkit.so"
  end
end

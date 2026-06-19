# frozen_string_literal: true

# Load the compiled Rust extension. Precompiled (platform) gems install it into a
# Ruby-ABI-versioned subdir (lib/clusterkit/<major.minor>/clusterkit.{so,bundle}) so a
# single fat gem can carry a binary per Ruby version; source/dev builds place it flat at
# lib/clusterkit/clusterkit.{so,bundle}. Try the versioned path first, fall back to the
# flat one. Resolution goes through $LOAD_PATH (`require`, never `require_relative`)
# because RubyGems installs native extensions outside the gem's lib/ dir.
begin
  RUBY_VERSION =~ /(\d+\.\d+)/
  require "clusterkit/#{Regexp.last_match(1)}/clusterkit"
rescue LoadError
  require "clusterkit/clusterkit"
end

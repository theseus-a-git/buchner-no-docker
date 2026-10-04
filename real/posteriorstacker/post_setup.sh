#!/usr/bin/env bash
# Clone PosteriorStacker (AGPLv3, so it is cloned here instead of being shipped in this package).
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -d "$HERE/PosteriorStacker" ] || git clone --depth 1 https://github.com/JohannesBuchner/PosteriorStacker "$HERE/PosteriorStacker" \
  || { echo "!! could not clone PosteriorStacker"; exit 1; }
echo "PosteriorStacker ready in $HERE/PosteriorStacker"

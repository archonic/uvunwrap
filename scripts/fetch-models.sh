#!/usr/bin/env bash
# Downloads the benchmark models into the gitignored models/ directory, from
# https://github.com/alecjacobson/common-3d-test-models pinned to one commit
# so benchmark inputs never change. Existing files with the right checksum
# are kept. Usage: scripts/fetch-models.sh
set -euo pipefail

commit=8a4f8642acaf43f9cd7b67858a1502e1055ef202
base="https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/$commit/data"
dir="$(cd "$(dirname "$0")/.." && pwd)/models"

# Local name, upstream name, SHA-256.
models=(
    "spot spot 0738b5e8608fed74e5e8c7aa8dd0af97b4b74f9f6cbf7aac84cd7e40b2e44a75"
    "beetle beetle 46ee9dcbec84b8d1b8f0d743ab1c91b1ed81378672fc0516e7afff5172d7c9b7"
    "bunny stanford-bunny 1eb35d1e21ce99e5ce911353b6be278990713448dd9e8f5c9387f9de39b32205"
    "armadillo armadillo d2fd040334c12b16804fbcb9962391232cc24708244617f931baaa871210563e"
    "nefertiti nefertiti 19e39748b145092878cc4faafd050505eb5991c5d3e497be2b4575673fa87968"
    "igea igea 13dd57979295049c73830313772c379da4508e035551921686e2edc7de9bff0e"
)

sha256() {
    if command -v sha256sum >/dev/null; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

mkdir -p "$dir"
for entry in "${models[@]}"; do
    read -r name upstream sum <<<"$entry"
    file="$dir/$name.obj"
    if [[ -f "$file" && "$(sha256 "$file")" == "$sum" ]]; then
        echo "$name.obj: up to date"
        continue
    fi
    echo "$name.obj: downloading $upstream.obj"
    curl -fsSL -o "$file.part" "$base/$upstream.obj"
    if [[ "$(sha256 "$file.part")" != "$sum" ]]; then
        rm -f "$file.part"
        echo "$name.obj: checksum mismatch" >&2
        exit 1
    fi
    mv "$file.part" "$file"
done

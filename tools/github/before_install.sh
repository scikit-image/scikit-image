#!/usr/bin/env bash
set -ex

export PIP_DEFAULT_TIMEOUT=60

if [[ $MINIMUM_REQUIREMENTS == 1 ]]; then
    for filename in requirements/*.txt; do
        sed -i 's/>=/==/g' "$filename"
    done
fi

python -m pip install --upgrade pip

# Install build time requirements
python -m pip install $PIP_FLAGS -r requirements/build.txt

# Show what's installed
python -m pip list

set +ex

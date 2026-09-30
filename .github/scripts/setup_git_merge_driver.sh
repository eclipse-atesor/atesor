#!/bin/sh
# Set up the recipe cache merge driver in this clone.
#
# .gitattributes marks data/recipe_cache.json "merge=binary", so git stops
# at a conflict instead of merging the JSON line by line (a line merge left
# duplicate and mixed packages in commit 6b50080). After this script, git
# merges the file by meaning with no conflict: a recipe that one side
# changed takes that change, and when both sides changed it, the newest
# recipe wins.
#
# The script changes only this clone: .git/config and .git/info/attributes.
# Run it once, from any folder. It is safe to run again.
set -eu

# The clone that holds this script, whatever the current folder is.
root=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd -P)
cd "$root"
if [ "$(git rev-parse --show-toplevel)" != "$root" ]; then
    echo "error: $root is not the top folder of a git clone" >&2
    exit 1
fi

git config merge.recipe-cache.name "Atesor recipe cache: newest recipe wins"
git config merge.recipe-cache.driver \
    "python3 .github/scripts/merge_data_artifacts.py --merge-driver %O %A %B"

# info/attributes wins over .gitattributes, for this clone only.
attributes=$(git rev-parse --git-path info/attributes)
mkdir -p "$(dirname "$attributes")"
line="data/recipe_cache.json merge=recipe-cache"
if ! grep -qxF "$line" "$attributes" 2>/dev/null; then
    printf '%s\n' "$line" >> "$attributes"
fi

echo "Recipe cache merge driver set up in $root"
git check-attr merge -- data/recipe_cache.json

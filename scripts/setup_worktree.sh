#!/usr/bin/env bash
#
# Point a git worktree's heavy, gitignored data directories at the main checkout.
#
# WHY THIS EXISTS
#
# `git worktree remove` deletes the worktree directory outright, including everything
# gitignored inside it. That is the intended behaviour and git's own safety checks cannot
# help: they look at commits and tracked files, so a clean `git status` and "0 commits
# unmerged" both report all-clear while gigabytes of uncommitted derived data sit in the
# directory about to be erased.
#
# This has already cost this repo once: a worktree that had built
# `benchset_v1_{small_test,large_set_a,large_set_b}.parquet` (2.7 GB) and
# `papers_benchset_v1.parquet` was removed after its PR merged, and all four went with it.
# `embeddings_cache/` survived only because it was already symlinked here by hand - which
# is the whole idea, generalised.
#
# After running this, anything a worktree writes into data/processed lands in the main
# checkout and outlives the worktree.
#
# THE TRADE-OFF, STATED PLAINLY
#
# Worktrees now SHARE processed data instead of isolating it. Two agents rebuilding
# `papers_fe.parquet` at the same time will collide. That is the right call here because
# every file involved is a gitignored derived artifact rebuilt from tracked inputs, and
# `embeddings_cache/` was already shared this way - but it is a real change, not a free
# win. If you want an isolated worktree, don't run this.
#
# Usage:  bash scripts/setup_worktree.sh          # from inside the worktree
#         bash scripts/setup_worktree.sh <path>   # or point it at one
#
# Idempotent - safe to re-run.

set -euo pipefail

WORKTREE="${1:-$PWD}"
cd "$WORKTREE"

# The main checkout is the parent of the *common* git dir; a worktree's own .git is a file
# pointing back at it, so this resolves correctly from either side.
MAIN="$(dirname "$(cd "$(git rev-parse --git-common-dir)" && pwd)")"
HERE="$(pwd -P)"

if [ "$HERE" = "$MAIN" ]; then
    echo "Refusing to run: $HERE is the main checkout, not a worktree." >&2
    echo "Its data directories are the link targets - there is nothing to point elsewhere." >&2
    exit 1
fi

# Everything large, gitignored, and expensive or slow to rebuild. Order does not matter.
LINKS=(
    "data/processed"      # embeddings cache, feature tables, the benchset sets
    "data/benchsets_v1"   # 92 MB external corpus, re-downloadable but tedious
    ".venv"               # not data, but rebuilding it per worktree is pure waste
)

echo "worktree : $HERE"
echo "main     : $MAIN"
echo

for rel in "${LINKS[@]}"; do
    target="$MAIN/$rel"

    if [ ! -e "$target" ]; then
        echo "skip  $rel  (no such path in the main checkout)"
        continue
    fi
    if [ -L "$rel" ]; then
        echo "ok    $rel  (already a symlink)"
        continue
    fi

    # Tracked files living under this path would be reported as deleted once the directory
    # becomes a symlink, because git does not follow symlinks when resolving tracked paths.
    # They are still readable through the link with the right content, so the fix is to
    # stop git checking them rather than to move them.
    tracked="$(git ls-files "$rel")"

    rm -rf "$rel"
    mkdir -p "$(dirname "$rel")"
    ln -s "$target" "$rel"
    echo "link  $rel  ->  $target"

    if [ -n "$tracked" ]; then
        echo "$tracked" | tr '\n' '\0' | xargs -0 git update-index --skip-worktree
        echo "      skip-worktree set on $(echo "$tracked" | wc -l | tr -d ' ') tracked file(s) beneath it"
    fi
done

# The symlinks themselves are untracked entries that `git add -A` would otherwise commit as
# machine-specific absolute paths. info/exclude is shared with the main checkout, where
# these paths are real directories whose contents .gitignore already covers - so excluding
# them there is a no-op, and tracked files are never affected by ignore rules anyway.
EXCLUDE="$(git rev-parse --path-format=absolute --git-path info/exclude)"
for rel in "${LINKS[@]}"; do
    if ! grep -qxF "/$rel" "$EXCLUDE" 2>/dev/null; then
        printf '/%s\n' "$rel" >> "$EXCLUDE"
    fi
done

echo
echo "Done. git status in this worktree:"
git status --short || true
echo "(no output above means clean)"
echo
echo "To undo in this worktree:"
echo "  git update-index --no-skip-worktree \$(git ls-files data/processed)"
echo "  rm data/processed && git checkout -- data/processed"
echo
echo "KNOWN GOTCHA: skip-worktree makes git ignore local changes to those tracked files,"
echo "so if one of them (e.g. papers_combined.parquet) is updated upstream, a pull in this"
echo "worktree can refuse to overwrite it. Undo as above, pull, then re-run this script."

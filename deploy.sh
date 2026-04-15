#!/usr/bin/env bash
# Deploy a Quarto site to a per-branch subdirectory on gh-pages.
# This script DOES NOT RENDER the Quarto site, so make sure to run `quarto render` first.
# Usage:
#   deploy.sh                                  Deploy full site
set -euo pipefail

# Check dependencies
command -v git >/dev/null 2>&1 || { echo "Error: git is required but not installed."; exit 1; }
command -v realpath >/dev/null 2>&1 || { echo "Error: realpath is required but not installed."; exit 1; }

# Check we're in a git repository
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "Error: Not in a git repository. Run this from within a git repo."
  exit 1
}

# Check this is a Quarto project (has _quarto.yml)
[[ -f "$REPO_ROOT/_quarto.yml" ]] || {
  echo "Error: No _quarto.yml found at repository root. Is this a Quarto project?"
  exit 1
}

BRANCH="$(git branch --show-current)"
DEST=$([[ "$BRANCH" == "main" ]] && echo "." || echo "preview/$(echo "$BRANCH" | sed 's/[^a-zA-Z0-9._-]/-/g')")
TMPDIR=""

cleanup() {
  # Remove worktree if created, then the temp directory (may still exist if
  # we fell into the `git init` branch where no worktree was registered).
  # Each line ends with `|| true` because this runs as an EXIT trap under
  # `set -e` — a failing conditional must not propagate as a script failure.
  [[ -n "$TMPDIR" ]] && git -C "$REPO_ROOT" worktree remove --force "$TMPDIR" 2>/dev/null || true
  [[ -n "$TMPDIR" && -d "$TMPDIR" ]] && rm -rf "$TMPDIR" || true
}
trap cleanup EXIT

# Delete any stale `_deploy_tmp` branch from a previous run (worktrees share
# branch refs with the main repo, so the branch persists across invocations)
git -C "$REPO_ROOT" branch -D _deploy_tmp 2>/dev/null || true

# Fetch the latest gh-pages from the remote so the deploy is always based on
# the current remote state, not a stale or missing local branch
git -C "$REPO_ROOT" fetch --quiet origin gh-pages 2>/dev/null || true

# Deploy to gh-pages subdirectory via worktree based on the latest origin/gh-pages
# (or fresh repo if gh-pages doesn't exist remotely yet)
TMPDIR="$(mktemp -d)"
if git -C "$REPO_ROOT" rev-parse --verify --quiet origin/gh-pages >/dev/null; then
  git -C "$REPO_ROOT" worktree add -q --detach "$TMPDIR" origin/gh-pages
else
  git init -q "$TMPDIR"
  git -C "$TMPDIR" remote add origin "$(git -C "$REPO_ROOT" remote get-url origin)"
fi

# Full-site deploy
[[ -d "$REPO_ROOT/_site" ]] || { echo "No _site/ found — render first."; exit 1; }
mkdir -p "$TMPDIR/$DEST"
rsync -a --delete --exclude='preview' --exclude='.nojekyll' --exclude='.git' "$REPO_ROOT/_site/" "$TMPDIR/$DEST/"
touch "$TMPDIR/.nojekyll"

# If no main deployment exists yet, redirect root to the preview index
if [[ ! -f "$TMPDIR/index.html" && -d "$TMPDIR/preview" ]]; then
  cat > "$TMPDIR/index.html" <<'REDIRECT'
<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta http-equiv="refresh" content="0; url=preview/">
<title>Redirecting…</title></head>
<body><p>Redirecting to <a href="preview/">branch previews</a>…</p></body></html>
REDIRECT
fi

# Generate preview index listing all branch subdirectories
if [[ -d "$TMPDIR/preview" ]]; then
  {
    echo '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Branch Previews</title>'
    echo '<style>body{font-family:system-ui,sans-serif;max-width:600px;margin:2rem auto;padding:0 1rem}a{display:block;padding:.4rem 0}</style>'
    echo '</head><body><h1>Branch Previews</h1>'
    for d in "$TMPDIR"/preview/*/; do
      [[ -d "$d" ]] || continue
      name="$(basename "$d")"
      echo "<a href=\"$name/\">$name</a>"
    done
    echo '</body></html>'
  } > "$TMPDIR/preview/index.html"
fi

cd "$TMPDIR"
git add -A
git diff --cached --quiet && { echo "gh-pages/$DEST is already up to date."; exit 0; }
# Create a fresh orphan commit (no history)
git checkout --orphan _deploy_tmp
git add -A
git commit -m "Deploy $BRANCH -> $DEST"
# --force-with-lease refuses the push if origin/gh-pages moved since we fetched,
# so a concurrent deploy/cleanup can't be silently clobbered
if EXPECTED_SHA=$(git -C "$REPO_ROOT" rev-parse --verify --quiet origin/gh-pages); then
  git push --force-with-lease="gh-pages:$EXPECTED_SHA" origin HEAD:gh-pages
else
  # First-ever deploy into an empty remote gh-pages: no lease to check
  git push origin HEAD:gh-pages
fi

echo "Deployed to /$DEST/"

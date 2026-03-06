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
  # Remove worktree if created
  [[ -n "$TMPDIR" ]] && git -C "$REPO_ROOT" worktree remove --force "$TMPDIR" 2>/dev/null || true
}
trap cleanup EXIT

# Deploy to gh-pages subdirectory via worktree (or fresh repo if gh-pages doesn't exist yet)
TMPDIR="$(mktemp -d)"
if git rev-parse --verify gh-pages >/dev/null 2>&1; then
  git -C "$REPO_ROOT" worktree add -q "$TMPDIR" gh-pages
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
git push --force origin HEAD:gh-pages

echo "Deployed to /$DEST/"

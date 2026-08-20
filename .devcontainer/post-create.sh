#!/usr/bin/env bash
# One-command setup for the nearstack pnpm workspace.
# Runs once, after the container is created and the workspace is mounted.
set -euo pipefail

echo "node $(node --version)"

# Claude Code's config directory is a named volume so credentials and settings
# survive rebuilds. CLAUDE_CONFIG_DIR (set in devcontainer.json) relocates the
# whole config directory here, including ~/.claude.json which holds the OAuth
# session -- a bare ~/.claude mount would persist everything except the login.
#
# Docker creates that volume root-owned, so hand it to the container user the
# first time it comes up. The ! -w guard keeps this a no-op once correct;
# without it every rebuild recursively chowns accumulated session history.
CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
if [ -d "$CLAUDE_DIR" ] && [ ! -w "$CLAUDE_DIR" ]; then
  if command -v sudo > /dev/null; then
    echo "claiming $CLAUDE_DIR for $(id -un)"
    sudo chown -R "$(id -u):$(id -g)" "$CLAUDE_DIR"
  else
    echo "warning: $CLAUDE_DIR is not writable and sudo is unavailable;" \
      "Claude Code sign-in will not persist" >&2
  fi
fi

# Restore any skills this repo pins. The mdm binary comes from the dev container
# feature; it only reads the repo once the workspace is mounted, which is why
# this lives here rather than in the feature. No lock file yet means nothing to
# install -- create one with `mdm skills add <source> --skill <name>`.
if command -v mdm > /dev/null; then
  echo "mdm $(mdm --version 2>/dev/null || echo '(version unavailable)')"
  if [ -f skills-lock.json ]; then
    mdm skills install
  else
    echo "no skills-lock.json; skipping 'mdm skills install'"
  fi
fi

# corepack activates the pnpm version pinned by "packageManager" in package.json.
corepack enable
corepack prepare --activate

echo "pnpm $(pnpm --version)"

pnpm install -r

echo
echo "Ready. Next: pnpm build, pnpm test, pnpm dev"
echo "Claude Code: run 'claude' (sign-in persists across rebuilds)"

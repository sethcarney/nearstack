#!/usr/bin/env bash
# One-command setup for the nearstack pnpm workspace.
# Runs once, after the container is created.
set -euo pipefail

echo "node $(node --version)"

# CLAUDE_CONFIG_DIR points at a named volume so Claude Code's OAuth session,
# settings, and history survive rebuilds. Docker creates that volume root-owned,
# so hand it to the container user the first time it comes up. The ! -w guard
# keeps this a no-op afterwards instead of re-chowning accumulated history.
claude_dir="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
if [ -d "$claude_dir" ] && [ ! -w "$claude_dir" ]; then
  if command -v sudo > /dev/null; then
    echo "claiming $claude_dir for $(id -un)"
    sudo chown -R "$(id -u):$(id -g)" "$claude_dir"
  else
    echo "warning: $claude_dir is not writable and sudo is unavailable;" \
      "Claude Code sign-in will not persist" >&2
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

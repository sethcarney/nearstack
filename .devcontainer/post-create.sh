#!/usr/bin/env bash
# One-command setup for the nearstack pnpm workspace.
# Runs once, after the container is created.
set -euo pipefail

echo "node $(node --version)"

# ~/.claude is a named volume so Claude Code's credentials and settings survive
# rebuilds. Docker creates that volume root-owned, so hand it to the container
# user the first time it comes up.
if [ -d "$HOME/.claude" ] && [ ! -w "$HOME/.claude" ]; then
  if command -v sudo > /dev/null; then
    echo "claiming $HOME/.claude for $(id -un)"
    sudo chown -R "$(id -u):$(id -g)" "$HOME/.claude"
  else
    echo "warning: $HOME/.claude is not writable and sudo is unavailable;" \
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

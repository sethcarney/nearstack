#!/usr/bin/env bash
# One-command setup for the nearstack pnpm workspace.
# Runs once, after the container is created.
set -euo pipefail

echo "node $(node --version)"

# corepack activates the pnpm version pinned by "packageManager" in package.json.
corepack enable
corepack prepare --activate

echo "pnpm $(pnpm --version)"

pnpm install -r

echo
echo "Ready. Next: pnpm build, pnpm test, pnpm dev"

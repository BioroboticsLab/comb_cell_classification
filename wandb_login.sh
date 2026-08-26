#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/." && pwd)"
KEY_FILE="$ROOT_DIR/.wandb_key"

if [[ ! -f "$KEY_FILE" ]]; then
    echo "Missing .wandb_key in project root"
    echo "Paste the wandb API key in .wandb_key file and try again."
    exit 1
fi

export WANDB_API_KEY=$(tr -d '[:space:]' < "$KEY_FILE")

wandb login "$WANDB_API_KEY"
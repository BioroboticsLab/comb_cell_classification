#!/usr/bin/env bash
set -euo pipefail

#######################
# author: Vinzent Jörß
# project: Bachelor Thesis - Comb Cell Classification (CCC)
#
# This script sets up the Python virtual environment for the project using uv.
# It checks if uv is installed, and if not, offers to install it (can be auto-confirmed with option --yes).
# Then it syncs the virtual environment based on the pyproject.toml config.
#
# Usage:
#   ./setup.sh [--yes] [--add-commands] [--rm-commands] [--no-commands]
#              [--uninstall] [-s|--sync] [--no-cache] [--clean-cache]
# Options:
#   --yes                 Auto-confirm prompts (uv installation, --uninstall).
#   --add-commands        Link the console scripts (ccc, hbcsp, the annotation tools) into
#                         ~/.local/bin so they work without `uv run`. Done on a normal run too.
#   --rm-commands         Remove those links.
#   --no-commands         Sync without linking them.
#   --uninstall           Remove venv, command links and the pipeline checkout.
#   -s, --sync            Alias for a normal run (kept for completeness): a normal run already
#                         syncs and clears stale __pycache__.
#   --no-cache            Run uv with no cache (UV_NO_CACHE). Avoids piling a multi-GB
#                         uv cache into a quota-limited home dir.
#   --clean-cache         Run `uv cache clean` to free the uv cache, then exit.
#
#######################

VENV_DIR=".venv"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Project's console scripts ([project.scripts] in pyproject.toml), linked into BIN_DIR so they can be called bare, without `uv run` or activating the venv.
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
PROJECT_COMMANDS=(ccc hbcsp annotation-tool_v2 annotation-tool)

HBCSP_REPO="https://github.com/BioroboticsLab/honeybee_cell_segmentation_pipeline.git"
HBCSP_REV="d5d73bf32a073d89e6cfe531b9c066412a2bc947"
HBCSP_DIR="external/honeybee_cell_segmentation_pipeline"

usage() {
  cat <<'EOF'
Usage:
  ./setup.sh [--yes] [--add-commands] [--rm-commands] [--no-commands]
             [--uninstall] [-s|--sync] [--no-cache] [--clean-cache]

Options:
  --yes                 Auto-confirm prompts (uv installation, --uninstall).
  --add-commands        Link ccc, hbcsp and the annotation tools into ~/.local/bin, then exit.
                        A normal run does this anyway; use --no-commands to skip it.
  --rm-commands         Remove those links, then exit.
  --no-commands         Sync without linking the commands.
  --uninstall           Remove the venv, the command links and the pipeline checkout,
                        then exit. Asks first unless --yes.
  -s, --sync            Alias for a normal run (kept for muscle memory): a normal run already
                        syncs and clears stale __pycache__.
  --no-cache            Run uv with no cache (avoids piling a multi-GB cache into home).
  --clean-cache         Run `uv cache clean` to free the uv cache, then exit.
EOF
}

# True if $1 is a symlink into this repo's venv (so we only ever remove our own).
is_our_link() {
  [[ -L "$1" ]] || return 1
  [[ "$(readlink "$1")" == "$REPO/$VENV_DIR/bin/"* ]]
}

# Drop links of ours whose target is gone — the repo was moved or renamed, or the venv was deleted. 
# moved repo cannot clean up after itself from the old path, so this runs on every setup.sh call from wherever the repo lives now.
prune_stale_commands() {
  local cmd link
  for cmd in "${PROJECT_COMMANDS[@]}"; do
    link="$BIN_DIR/$cmd"
    if [[ -L "$link" && ! -e "$link" ]]; then
      rm -f "$link"
      echo "  removed stale link $link (its target no longer exists)"
    fi
  done
}

# Link the console scripts into BIN_DIR. Never clobbers a file or a foreign symlink of the same name — that would hijack someone else's command.
add_commands() {
  mkdir -p "$BIN_DIR"
  prune_stale_commands
  local cmd link target
  for cmd in "${PROJECT_COMMANDS[@]}"; do
    target="$REPO/$VENV_DIR/bin/$cmd"
    link="$BIN_DIR/$cmd"
    if [[ ! -x "$target" ]]; then
      echo "  skipped $cmd (not installed in the venv)"
      continue
    fi
    if [[ -e "$link" || -L "$link" ]] && ! is_our_link "$link"; then
      if [[ -L "$link" && "$(readlink "$link")" == *"/uv/tools/ccc-bachelor-thesis/"* ]]; then
        echo "  replacing $cmd: was a frozen 'uv tool install' snapshot" \
             "(remove it fully with: uv tool uninstall ccc-bachelor-thesis)"
      else
        echo "  skipped $cmd: $link already exists and is not ours"
        continue
      fi
    fi
    ln -sfn "$target" "$link"
    echo "  linked $cmd -> $target"
  done
  case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) echo "  note: $BIN_DIR is not on your PATH, so the commands are not callable bare yet." ;;
  esac
}

# Remove links (including stale ones, whose target string no longer matches).
rm_commands() {
  local cmd link removed=0
  for cmd in "${PROJECT_COMMANDS[@]}"; do
    link="$BIN_DIR/$cmd"
    if [[ -L "$link" ]] && { is_our_link "$link" || [[ ! -e "$link" ]]; }; then
      rm -f "$link"
      echo "  removed $link"
      removed=1
    fi
  done
  [[ "$removed" -eq 1 ]] || echo "  no command links of this repo in $BIN_DIR"
}

# Drop compiled bytecode. Stale __pycache__ can shadow edited sources (rsync -a preserves mtimes, so Python's mtime-based invalidation may keep loading old .pyc files) removing it forces a clean compile on the next start.
clean_pycache() {
  local n
  # ponytail: `|| true` because find exits 1 when tests/ is absent, and set -e + pipefail would kill the script here.
  n="$(find "$REPO/src" "$REPO/tests" -type d -name __pycache__ 2>/dev/null | wc -l | tr -d ' ')" || true
  if [[ "$n" -gt 0 ]]; then
    find "$REPO/src" "$REPO/tests" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
  fi
  echo "  removed $n __pycache__ dir(s)"
}

# Undo full setup: the venv, the command links and the pipeline checkout, repo and any data will not be touched
uninstall() {
  echo "This removes:"
  echo "  - the virtual environment   $REPO/$VENV_DIR"
  echo "  - the command links         $BIN_DIR/{$(IFS=,; echo "${PROJECT_COMMANDS[*]}")}"
  echo "  - the pipeline checkout     $REPO/$HBCSP_DIR  (kept if it has local changes)"
  echo "It does not touch the repository itself, your data, or ~/ccc."
  if [[ "$AUTO_YES" -ne 1 ]]; then
    read -r -p "Proceed? [y/N] " answer
    case "${answer:-}" in
      y|Y|yes|YES) ;;
      *) echo "Aborted."; exit 1 ;;
    esac
  fi

  echo "Removing command links..."
  rm_commands
  if [[ -d "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
    echo "Removed $VENV_DIR"
  fi
  if [[ -d "$HBCSP_DIR/.git" ]]; then
    if [[ -n "$(git -C "$HBCSP_DIR" status --porcelain 2>/dev/null)" ]]; then
      echo "Kept $HBCSP_DIR: it has local changes. Delete it yourself if you are sure."
    else
      rm -rf "$HBCSP_DIR"
      echo "Removed $HBCSP_DIR"
    fi
  fi
  echo "Uninstalled."
}

AUTO_YES=0
NO_CACHE=0
LINK_COMMANDS=1

# --yes may appear after the flag it is meant to confirm, so read it up front.
for arg in "$@"; do
  [[ "$arg" == "--yes" ]] && AUTO_YES=1
done

for arg in "$@"; do
  case "$arg" in
    -h|--help)             usage; exit 0 ;;
    --yes)                 ;;  # already handled above
    --add-commands)        add_commands; exit 0 ;;
    --rm-commands)         rm_commands; exit 0 ;;
    --no-commands)         LINK_COMMANDS=0 ;;
    --uninstall)           uninstall; exit 0 ;;
    -s|--sync)             ;;  # alias for a normal run (kept for muscle memory)
    --no-cache)            NO_CACHE=1 ;;
    --clean-cache)         command -v uv >/dev/null 2>&1 && uv cache clean || echo "uv not found"; exit 0 ;;
    *)
      echo "Unknown option: $arg"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "pyproject.toml" ]]; then
  echo "Missing pyproject.toml in $(pwd)"
  exit 1
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin) PLATFORM="macOS" ;;
  Linux)  PLATFORM="Linux" ;;
  *)
    echo "Unsupported OS: $OS"
    exit 1
    ;;
esac

echo "Platform: $PLATFORM ($ARCH)"
echo "Venv dir: $VENV_DIR"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

install_uv() {
  echo "Installing uv..."

  if need_cmd curl; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif need_cmd wget; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Neither curl nor wget is available. Install one of them and re-run."
    exit 1
  fi

  export PATH="$HOME/.local/bin:$PATH"

  if ! need_cmd uv; then
    echo "uv installation completed, but uv is still not in PATH."
    echo "Please add ~/.local/bin to PATH and re-run this script."
    exit 1
  fi

  echo "uv installed."
}

ensure_hbcsp_checkout() {
  if [[ -d "$HBCSP_DIR/.git" ]]; then
    local rev; rev="$(git -C "$HBCSP_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "Pipeline checkout: $HBCSP_DIR (at ${rev:0:12})"
    if [[ "$rev" != "$HBCSP_REV" ]]; then
      echo "  note: pyproject pins ${HBCSP_REV:0:12}; leaving your checkout as it is."
      echo "        To match the pin:  git -C $HBCSP_DIR checkout $HBCSP_REV"
    fi
    return 0
  fi
  if [[ -e "$HBCSP_DIR" ]]; then
    echo "$HBCSP_DIR exists but is not a git checkout. Move it aside and re-run."
    exit 1
  fi
  if ! need_cmd git; then
    echo "git is required to fetch the honeybee pipeline. Install git and re-run."
    exit 1
  fi
  echo "Cloning the honeybee pipeline into $HBCSP_DIR (~50 MB, includes the segmentation weights)..."
  git clone --quiet "$HBCSP_REPO" "$HBCSP_DIR"
  git -C "$HBCSP_DIR" checkout --quiet "$HBCSP_REV"
  echo "  cloned at ${HBCSP_REV:0:12}"
}

# Ensure uv exists, otherwise install (after asking)
if ! need_cmd uv; then
  echo "uv not found in PATH."

  if [[ "$AUTO_YES" -eq 1 ]]; then
    answer="y"
  else
    read -r -p "Install uv now? [y/N] " answer
  fi

  case "${answer:-}" in
    y|Y|yes|YES)
      install_uv
      ;;
    *)
      echo "Please install uv, then re-run this script."
      exit 1
      ;;
  esac
fi

if [[ "$NO_CACHE" -eq 1 ]]; then
  export UV_NO_CACHE=1
  echo "Cache disabled (UV_NO_CACHE=1)."
fi

ensure_hbcsp_checkout

echo "Clearing compiled bytecode (stale __pycache__ can shadow edited sources)..."
clean_pycache

echo ""
echo "──────────────────────────────────────────────"
echo "Syncing virtual environment from pyproject.toml"
echo "──────────────────────────────────────────────"
echo "  Python: $(uv python find 2>/dev/null || echo 'will be downloaded')"
echo "  Venv:   $(pwd)/$VENV_DIR"
echo ""

# Show which packages will be installed/updated before syncing
echo "Resolving dependencies..."
uv sync --dry-run 2>&1 | while IFS= read -r line; do
  echo "  $line"
done
echo ""

echo "Installing/syncing packages..."
uv sync --verbose 2>&1 | while IFS= read -r line; do
  case "$line" in
    *"already installed"*) ;;  # skip noisy "already installed" lines
    *) echo "  $line" ;;
  esac
done

echo ""
echo "──────────────────────────────────────────────"
echo "Installed packages:"
echo "──────────────────────────────────────────────"
uv pip list 2>/dev/null | head -5
echo "  ... ($(uv pip list 2>/dev/null | tail -n +3 | wc -l) packages total)"
echo ""

if [[ "$LINK_COMMANDS" -eq 1 ]]; then
  echo "──────────────────────────────────────────────"
  echo "Linking the project commands into $BIN_DIR"
  echo "──────────────────────────────────────────────"
  add_commands
  echo ""
fi

echo "Done. The commands are:"
echo "  ccc --help            # project commands"
echo "  hbcsp --help          # honeybee pipeline tools"
echo "  annotation-tool_v2 <folder>   # the napari annotation GUI"
echo "Without the links, prefix them with uv, e.g. \`uv run ccc --help\`,"
echo "or activate the venv: source $VENV_DIR/bin/activate"
echo ""
echo "Remove the command links with:     ./setup.sh --rm-commands"
echo "Undo the whole setup with:         ./setup.sh --uninstall"

# Bachelor Thesis — Comb Cell Classification

Author: Vinzent Jörß.

A pipeline for classifying honeybee comb cells: train a classifier, run inference to label annotations, and post-process predictions using the further developed Annotation Tool from [honeybee_cell_segmentation_pipeline](https://github.com/BioroboticsLab/honeybee_cell_segmentation_pipeline).

## Setup

```bash
./setup.sh                      # install uv (if missing) and sync the venv from pyproject.toml
./wandb_login.sh                # once per machine: log in to W&B with the key in .wandb_key
```

`setup.sh` links `ccc`, `hbcsp` and the annotation tools into `~/.local/bin`, which must be on your `PATH`.

What you get depends on the platform:

- **macOS:** a ~1.3 GB venv with no TensorFlow, which is gated to Linux in `pyproject.toml`. The annotation tool works, but its **Classify sequence…** button is disabled. Training and `ccc classify` need Linux.
- **Linux (x86_64):** a ~6.7 GB venv, because `tensorflow[and-cuda]` is a core dependency. This happens even if you only annotate. If `annotation-tool_v2` still reports `Could not load the Qt platform plugin "xcb"`, run it once with `QT_DEBUG_PLUGINS=1` to see which system library is missing. Then install it, e.g. on Ubuntu: `sudo apt install libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xfixes0 libxcb-xkb1`. The tool already works around the conflicting Qt plugins bundled with opencv-python (`_repair_qt_plugin_path` in `src/napari_tools/annotation_tool_v2/__main__.py`).
- **Windows:** not supported.

`./wandb_login.sh` reads `.wandb_key` (project root, gitignored — paste your W&B API key into it) and runs `wandb login`, which stores the key in `~/.netrc`. Every later online training run and every `wandb sync` picks it up from there. `ccc training --offline` needs no wandb key to run.

## Commands

Commands are grouped under the entry points — `ccc` for this project's commands and `hbcsp` for the honeybee_cell_segmentation_pipeline tools. Run commands directly through `uv run`:

| Command | Purpose |
|---|---|
| `uv run ccc training [--options]` | Launch/manage a training run in the background, in its own tmux session. |
| `uv run ccc classify --images-dir <dir>` | Load a trained `.keras` model and write predicted labels into each image's `<stem>.json` in place. Picks the model interactively (see below) unless `--model-dir`/`--model-id` is given. `--images-dir` accepts a glob (`.../cam-*`); `--reset` blanks all labels back to `unlabeled`. |
| `uv run ccc postprocess <subcommand>` | Correction of prediction JSONs. Subcommands: `correct-spatial`, `correct-temporal`, `list-rules`; `--revert PATH` restores from `bkp/`. |
| `annotation-tool_v2 <folder>` | napari tool for annotating a time-ordered image sequence. |

Run `uv run ccc <command> --help` for a command's full flag list.

### `uv run ccc training`

Starts the training driver (the `_train_driver` sweep config in [src/cli/train.py](src/cli/train.py)) in its own detached tmux session, `ccc-training-<timestamp>` (`tmux` must be installed!). Everything a run produces lives under its own folder `output/training/<timestamp>/`. The logs in `log/` subfolder, the wandb files in the subfolder `wandb/`, and the trained checkpoints in `model/`.

Start / manage the training process:

```bash
uv run ccc training            # start a training run with the built-in default sweep
uv run ccc training --config configs/train.yaml  # train the sweep from a YAML
uv run ccc training --debug    # start in debug mode (ccc_debug wandb / DEBUG data)
uv run ccc training --offline  # log fully offline (WANDB_MODE=offline, no W&B API key)
uv run ccc training --kill     # stop the running training
uv run ccc training --restart  # kill the running run and start a fresh one
```

The start command prints the run's tmux session. `tmux attach -t <session>` joins it live (`Ctrl+B D` detaches again), `tmux ls` lists every running one. A run is "active" for `--kill`/`--restart`/`log` exactly as long as its session exists — it ends when the driver exits.

Inspect, list, and delete run logs under the `log` subcommand:

```bash
uv run ccc training log                    # tail the current run's log (table-select if several)
uv run ccc training log --currently-running <ID>  # tail a specific currently-running run's log
uv run ccc training log --list-all         # table of all runs: start time, status, model variants
uv run ccc training log --show <ID>        # print a past run's full log by its --list-all ID
uv run ccc training log --complete-log     # print the current run's full log
uv run ccc training log --cleanup          # collapse Keras progress-bar spam in all log dirs
uv run ccc training log --delete crashed   # delete runs by status (see below)
```

`--list-all` shows each run with an ID (the `#` column), its start time, an outcome (`success` / `partial` / `crashed` / `stopped` / `running`), and the model variants trained (with `✓`/`✗` per variant). Pass that ID — or the run's timestamp — to `--show <ID>` to read the full log.

`--delete` removes whole run directories by status — one or more of `success`, `partial`, `crashed`, `stopped`, `unknown`, or `all` (e.g. `--delete crashed stopped`). A run directory contains the run's logs, wandb data, **and model checkpoints**, so deleting a run removes them all. It lists what will be removed with per-run sizes and asks to confirm (`-y`/`--yes` skips the prompt); the currently-running run is never deleted.

#### Uploading an offline run to W&B

`ccc training --offline` records the full run under `output/training/<timestamp>/wandb/offline-run-*`. Upload it later with wandb's own CLI — a 1:1 replay with metric curves, eval panels, and the model artifact:

```bash
uv run wandb sync --mark-synced output/training/<timestamp>/wandb/offline-run-*
```

(Needs the one-time `./wandb_login.sh` from [Setup](#setup); without it, export the key for this shell: `export WANDB_API_KEY=$(cat .wandb_key)`.)

### `uv run ccc classify`

If neither `--model-dir` nor `--model-id` is given, classify shows a `rich`-table of the trained models under `--models-root` (default: the `output/` run tree plus `models/`) and asks which to use by ID. Each row is parsed from the model's directory name — model, `col` (cell-outer-layer), learning rate (and adaptive vs. fixed), `te` (total epochs, so maximum number of epochs), and whether class-weighting (`CW`) or oversampling (`OS`) was used — plus its evaluation F1 scores (Macro / Micro / Weighted) read from the model's `model_info.json` (written at train time). Each classify run gets its own tmux session (`ccc-classify-<timestamp>`) and its own folder `output/classify/<timestamp>/` holding its log (`log/`), wandb files (`wandb/`), a copy of every JSON it labelled (`jsons/<images-dir-name>/`), and — with `--evaluate` — the report under `evaluation_results/`.

```bash
uv run ccc classify --images-dir <dir>                              # pick a model from the table
uv run ccc classify --images-dir <dir> --model-id 2                 # pick non-interactively by table ID
uv run ccc classify --images-dir <dir> --model-dir models/<run>     # use an exact folder
```

`--images-dir` accepts a glob, so several directories are classified in one run (the model is loaded once and reused). Quote it so the shell passes the pattern through:

```bash
uv run ccc classify --images-dir "data/2025-comb-images/cam-*"
```

Use the option `--reset` to blank every annotation's `label` back to `unlabeled` instead of running a model — useful to start over before re-classifying or hand-labelling. It can be used with pattern and it always skips the backup subfolder:

```bash
uv run ccc classify --images-dir "data/2025-comb-images/cam-*" --reset
```

## Honeybee pipeline commands (`hbcsp`)

The project depends on the [honeybee_cell_segmentation_pipeline](https://github.com/BioroboticsLab/honeybee_cell_segmentation_pipeline) packages. Their tools are grouped under `hbcsp`:

| Command | Purpose |
|---|---|
| `uv run hbcsp cell-finder` | Detect cells in comb images. (CPU only due to dependency problems with tensorflow) |
| `uv run hbcsp performance-validation` | Validate cell-finder output against ground truth. |
| `uv run hbcsp mask-writer` | Convert JSON annotations to PNG segmentation masks. |
| `uv run hbcsp frame-extractor` | Extract frames from video files. |
| `uv run hbcsp annotation-tool` | legacy tool for manually labeling comb images. |
| `uv run hbcsp background-generator` | Generate bee-free background images (Linux only). |

(Each tool also still ships its own console script, so `uv run cell-finder …` works too; `hbcsp` is just the grouped entry point.)

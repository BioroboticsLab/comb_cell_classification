# annotation-tool_v2

A desktop app, built on [napari](https://napari.org), for **annotating cells in a time-ordered image sequence**. You point it at a folder of images, it sorts them in time, shows them as a scrollable, corner-pin-stabilized stack (when corner-pin info is provided), and overlays the cell annotations from the per-image JSONs next to the frames. When on Linux you can then start automatic cell classification by choosing a model to use. You can also select cells (click, brush, island-grow, gap-fill), assign labels to them, run automatic spatial/temporal label corrections, and mark sequence breakpoints. All edits are written back to the per-image JSONs in the background.

It replaces the honeybee `annotation-tool` (still available as `hbcsp annotation-tool`, because not complete functionality is implemented) for the sequence-based labeling workflow.

## Launch

Part of the **ccc** repo — installed with the repo's `uv sync`, no separate environment.
From the repo root:

```bash
annotation-tool_v2 </path/to/data/annotated/cam-1>
```

The folder argument is required. `-v/--verbose` enables DEBUG logging.

## Quick start

1. Run `annotation-tool_v2 /path/to/image/folder`.
2. Scrub the slider (or **Left Arrow/Right Arrow**) through the sequence.
3. Paint over cells to select them (brush mode is already active), pick a label, press **j**.
4. Edits autosave to the per-image JSONs in the background.

## Expected data layout

Pass the flat image folder itself:

```text
parent_folder/
├── cam-1/            ← pass THIS folder
│   ├── background_cam-1_20250614T084456.555811.256Z.png
│   ├── background_cam-1_20250614T084456.555811.256Z.json
│   └── …
└── label_classes.json← optional label → colour mapping (here or inside the folder);
                      only used when the `label_classes_file` pref is unset
```

The `.png` files are sorted chronologically by the timestamp in their names:

```text
background_cam-1_20250614T084456.555811.256Z.png
                 └──────┬──────┘ └────┬────┘
                     timestamp    sub-second fraction
```

`YYYYMMDDTHHMMSS` is the primary sort key; the irregular trailing fraction is only a numeric tiebreaker. Files with no parseable timestamp are **skipped with a warning**.

Per image, a sibling `<stem>.json` carries that frame's data. The tool **reads**: `annotations` (the cells), an optional per-image `corner_pin` and `sequence_breakpoint`. It **writes back** the cell annotations and the `sequence_breakpoint` flag, preserving any unknown keys.

## Docks

| Dock | Contents |
| --- | --- |
| Controls | Display scale (Full / $\frac{1}{2}$ / $\frac{1}{4}$), Copy image path, Warp toggle |
| Annotations | Label dropdown, Apply label to selection (j), Lock labeled points, Save annotations now, Classify sequence… (Linux only with NVIDIA GPU), colour legend |
| Selection | Select all / none (this frame), Select Island from selection, max enclosed gap slider, Fill enclosed gaps |
| Post-process | Max island size, min border fraction, Apply Spatial / Temporal Rule to Selection (tabbed with Annotations) |
| Breakpoints | Toggle breakpoint @ current frame + status |
| Grid | Show grid, neighbor distance (0 = auto) |
| Log | Timestamped message log (left) |

## Keyboard shortcuts

### Navigation

| Key | Action |
| --- | --- |
| `Left Arrow` / `Right Arrow` | Previous / next image |
| `w` | Toggle stabilized (corner pins) ↔ raw frames |
| `g` | Toggle the grid overlay |

### Annotating

| Key | Action |
| --- | --- |
| `b` | Brush mode — paint over cells to select them (`Shift` add, `Cmd/Ctrl` + `Shift` remove); active on start |
| `c` | Points mode — click / drag-select cells directly |
| `j` | Apply the selected label to the selected points |
| `h` | Hide / show the annotation points |
| `Cmd/Ctrl` + `L` | Lock labeled points (only `unlabeled` selectable) |
| `Cmd/Ctrl` + `I` | Keep only the selected points (current frame) |
| `Alt` + scroll | Resize the selected points |

## Performance

Frames are **loaded lazily**: only the frame on the slider and its immediate neighbours are held in memory, with JPEG proxies generated at the chosen display scale, so the app stays responsive even for long sequences. The **Display scale** dropdown (Full / $\frac{1}{2}$ / $\frac{1}{4}$) downsamples the *shown* image only — clicks and annotation coordinates stay at full resolution. JSON writes run on a background worker.

## Project layout

`AnnotationApp` is assembled from one mixin per concern — each `gui/_*.py` owns its own state, its dock and its key bindings.

```text
src/napari_tools/annotation_tool_v2/
├── core/                   ### logic — no napari/Qt imports
│   ├── parsing.py          # timestamp parsing + folder discovery
│   ├── frames.py           # lazy, windowed frame loader (FrameStore)
│   └── postprocess.py      # spatial / temporal label-correction rules
├── gui/                    ### napari / Qt
│   ├── app.py              # AnnotationApp — state, folder loading, docks, Log
│   ├── _controls.py        # Controls dock, navigation keys, warp + display scale
│   ├── _display.py         # image layer, corner-pin warping (cv2), warp cache
│   ├── _annotations.py     # points + brush layers, selection, key bindings
│   ├── _labels.py          # label map + colours, Annotations dock, labeling
│   ├── _classify.py        # "Classify sequence…" model inference (Linux only)
│   ├── _persistence.py     # background JSON writes, autosave, close handlers
│   ├── _breakpoints.py     # Breakpoints dock
│   ├── _postprocess.py     # Selection + Post-process docks
│   ├── _grid.py            # hex-grid overlay
│   └── _prefs.py           # prefs JSON (window layout, label_classes_file)
├── default_prefs.json      # defaults, overridden per machine by .annotation_tool_v2_prefs.json in repo root
└── __main__.py             # CLI entry point
```

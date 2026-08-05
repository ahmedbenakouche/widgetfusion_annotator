# WidgetFusion Annotator

Desktop UI widget annotation from multiple sources: **hover diff**, **YOLO**, **accessibility**, then **fusion**.

| Source | Overlay |
|--------|---------|
| Hover | green |
| YOLO | orange |
| Accessibility | blue |
| Fusion | magenta |

---

## Installation

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
pip install .
```

**Linux**

```bash
source venv/bin/activate
pip install .
sudo apt install python3-gi gir1.2-atspi-2.0 at-spi2-core   # AT-SPI a11y
```
**For uv**
```bash
uv venv --system-site-packages
uv sync
```
Verify that system site-packages are available with 
```bash
cat .venv/pyvenv.cfg
```
should return  ```include-system-site-packages = true ```


---

## Launch

```bash
python widgetfusion_annotator.py
```

or :

```bash
widgetfusion-annotator
```

---

## Workflow

1. Choose detection methods (hover / YOLO / accessibility) and hover mode (manual or autoscan).
2. **Enter** advances phases: hover → YOLO → accessibility → review.
3. **Accessibility** phase: scan, then a live filter dialog (windows, types/roles + inclusion).
4. In **review**, **← / →** change the view (unless a manual selection is active):
   - green = hover · orange = YOLO · blue = a11y · all stacked · magenta = fusion
5. On the “all stacked” view, **Enter** opens fusion.
6. **S** opens the save dialog (sources, path, JSON preview).
7. Closing / canceling the methods dialog at startup exits the app.

A short status line at the bottom of the overlay reports phase changes, scans, fusion, manual mode, save, and similar events.

---

## Shortcuts

Shortcuts are captured globally and **suppressed for other apps** while the program is running.

| Key | Action |
|-----|--------|
| **Enter** | Next step / fusion |
| **M** | Manual mode |
| **S** | Save |
| **Q** | Quit |
| **← / →** | Cycle review views |
| **← ↑ → ↓** | In manual mode with a selection: move box(es) by 1 px |
| **A** | In manual mode with a multi-selection: align group to the primary box |
| **Ctrl+Z / Ctrl+Y** | In manual mode: undo / redo |

---

## Manual mode (**M**)

Available on a single-source view (not on “all stacked”).

| Action | Effect |
|--------|--------|
| Left-click a box | Select / move / resize |
| Left-drag on empty space | Multi-select boxes **fully enclosed** by the cyan dashed rect; otherwise create a new box |
| Click + drag a member of a multi-selection | Move the whole group |
| Arrow keys | Move selected box(es) rigidly by 1 px |
| **A** (multi-selection) | Align to the **primary** box (yellow): same `w`/`h`; vertical → same `x`, keep each **center y**; horizontal → same `y`, keep each **center x**; grid → snap row/column **centers** from the primary |
| Right-click the selection | Delete |
| Right-drag | Erase boxes **fully enclosed** by the red rect |
| Ctrl+Z / Ctrl+Y | Undo / redo |

Resize handles appear only when **exactly one** box is selected.

---

## Accessibility

| Platform | Backend | Live filter |
|----------|---------|-------------|
| **Windows** | UI Automation (UIA) | ✓ |
| **Linux** | AT-SPI2 (PyGObject) — **X11** session recommended | ✓ |
| **macOS** | Not available | — |

After the scan: live checkboxes for **visible windows**, types/roles, and **Inclusion (keep outer box, remove enclosed)**; the blue overlay updates live.

---

## Fusion

Match when **IoU ≥ threshold** **or** (optional) **strict 100% inclusion**.

- Default priority: **a11y → hover → YOLO** (editable).
- **Live** preview in the dialog (OK applies, Cancel restores).
- Geometry from the priority source; a11y metadata kept when present in the group.
- “Keep orphan boxes”: keep detections with no cross-source match.

---

## Save

Default folder: **Desktop/annotations**.

- Sources to export (hover / YOLO / a11y / fusion)
- Folder + JSON preview
- **One JSON + image per source** (checked by default when multiple sources)
- **Save** / **Don't save** (reopens session config) / **Cancel** (stay in session)


Example a11y entry:

```json
{
  "id": 0,
  "source": "a11y",
  "bbox": {"x": 10, "y": 20, "w": 80, "h": 24},
  "control_type": "Button",
  "class_name": "Button",
  "name": "OK",
  "window": "Notepad / Document - Notepad"
}
```

---

## Files

| File | Role |
|------|------|
| `widgetfusion_annotator.py` | Overlay, capture, phases, manual mode, shortcuts, export |
| `fusion_mode.py` | Matching, fusion, dialogs (session / fusion / a11y / save) |
| `accessibility_boxes.py` | UIA (Windows), AT-SPI (Linux), macOS stub |

---

## License

MIT — see `LICENSE`.

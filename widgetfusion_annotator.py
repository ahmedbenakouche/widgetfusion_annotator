import numpy as np
import sys
import cv2
import copy
import json
import os
import threading
import time
from datetime import datetime
from pynput import keyboard, mouse
from ultralytics import YOLO

from accessibility_boxes import (
    get_accessibility_boxes_raw,
    apply_a11y_filters,
    a11y_box_coords,
    make_manual_a11y_widget,
    is_a11y_available,
    warn_linux_a11y_session,
)
from fusion_mode import (
    CombinedModeConfig,
    SOURCE_LABELS,
    build_save_payload,
    cluster_union_rect,
    compute_auto_fused_boxes,
    combined_effective_layer,
    combined_hover_diff_enabled,
    combined_hover_diff_target_layer,
    combined_manual_editing_allowed,
    combined_overlay_layers,
    cycle_view,
    show_session_config_dialog,
    show_fusion_config_dialog,
    show_save_config_dialog,
    show_a11y_filter_dialog,
)

# Silence a noisy non-blocking Qt warning on Windows.
# Note: must be set BEFORE importing/initializing Qt.
_qt_rules = os.environ.get("QT_LOGGING_RULES", "")
if "qt.qpa.window=false" not in _qt_rules:
    os.environ["QT_LOGGING_RULES"] = (";".join([r for r in [_qt_rules, "qt.qpa.window=false"] if r])).lstrip(";")

from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QCursor, QFont
from PyQt6.QtWidgets import QApplication, QWidget

import mss
mss_local = threading.local()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Margin around overlay rects so screen-diff ignores anti-aliased edges.
OVERLAY_IGNORE_MARGIN = 0
DIFF_THRESHOLD = 0
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "annotations")

OVERLAY_COLOR = QColor(0, 255, 0, 220)
YOLO_DISPLAY_COLOR = QColor(255, 140, 0, 220)
A11Y_DISPLAY_COLOR = QColor(0, 120, 255, 220)
FUSED_DISPLAY_COLOR = QColor(255, 255, 255, 230)
OVERLAY_LINE_WIDTH = 0
OVERLAY_REFRESH_MS = 30
LOOP_SLEEP = 0.01

# After the session config dialog closes, wait before capturing the baseline.
# Linux compositors often animate / delay the redraw; Windows is usually faster.
BASELINE_CAPTURE_SETTLE_MS = 200 if sys.platform.startswith("linux") else 80

# Ignore transient UI (tooltips) that appear away from the cursor.
IGNORE_NEW_DIFF_AWAY_FROM_CURSOR = True
NEW_DIFF_MIN_AREA = 25
NEW_DIFF_DILATE = 2

# extract_box: morph close + reject concave merges via solidity.
EXTRACT_BOX_MORPH_CLOSE_K = 3
EXTRACT_BOX_MORPH_CLOSE_ITER = 1
EXTRACT_BOX_MIN_SOLIDITY = 0.92

# ─────────────────────────────────────────────
# YOLO (optional autoscan)
# ─────────────────────────────────────────────
YOLO_MODEL_PATH = "yolo26n-1280.pt"
YOLO_CONF = 0.4
YOLO_HOVER_POINTS = ("center",)
YOLO_ROW_BIN_PX = 60
YOLO_MOVE_DELAY_S = 0.5


# ─────────────────────────────────────────────
# STATE GLOBAL
# ─────────────────────────────────────────────
EXIT_PROGRAM = False
RUNNING = False
initial_img = None

YOLO_AUTOSCAN = False
A11Y_SCAN_PENDING = False
yolo_model = None

COMBINED_MODE = False
COMBINED_PHASE = ""
COMBINED_CONFIG: CombinedModeConfig | None = None
COMBINED_VIEW = "all"
COMBINED_AUTO_FUSED = False
COMBINED_PHASES_PENDING: list[str] = []
combined_hover_boxes: list = []
combined_yolo_boxes: list = []
combined_a11y_boxes: list = []
combined_a11y_raw_boxes: list = []
combined_fused_boxes: list = []

COMBINED_FUSION_HIGHLIGHT_IDX = -1
COMBINED_FUSION_HIGHLIGHT_CLUSTERS: list = []

MANUAL_MODE = False
manual_drag_start = None
manual_drag_kind = None  # "create" | "erase"
manual_preview_box = None
manual_selected_index = None  # primary (handles / single edit)
manual_selected_indices: list[int] = []  # multi-select (marquee / group nudge)
manual_edit_mode = None
manual_edit_anchor = None
manual_edit_origin_box = None
manual_edit_origin_boxes: dict[int, tuple[int, int, int, int]] | None = None
manual_edit_edges = None

MANUAL_HANDLE_RADIUS = 8
MANUAL_NUDGE_PX = 1
MANUAL_HISTORY_MAX = 50
manual_undo_stack: list = []
manual_redo_stack: list = []
_manual_nudge_batch = False
_ctrl_pressed = False
_manual_cursor_override_active = False
_manual_edit_undo_pushed = False

state_lock = threading.Lock()


def _manual_clear_selection() -> None:
    global manual_selected_index, manual_selected_indices
    global manual_edit_mode, manual_edit_anchor, manual_edit_origin_box
    global manual_edit_origin_boxes, manual_edit_edges
    manual_selected_index = None
    manual_selected_indices = []
    manual_edit_mode = None
    manual_edit_anchor = None
    manual_edit_origin_box = None
    manual_edit_origin_boxes = None
    manual_edit_edges = None


def _manual_set_selection(indices: list[int], primary: int | None = None) -> None:
    """Set multi-selection; primary keeps resize handles when exactly one is selected."""
    global manual_selected_index, manual_selected_indices
    uniq = sorted({int(i) for i in indices if i >= 0})
    manual_selected_indices = uniq
    if not uniq:
        manual_selected_index = None
    elif primary is not None and primary in uniq:
        manual_selected_index = primary
    else:
        manual_selected_index = uniq[0]


def _manual_selection_snap() -> dict:
    return {
        "selected": manual_selected_index,
        "selected_indices": list(manual_selected_indices),
    }

# Capture geometry in MSS coordinates (updated on first grab).
CAPTURE_LEFT = 0
CAPTURE_TOP = 0
CAPTURE_W, CAPTURE_H = 1, 1
app_bridge = None
overlay_window = None

mouse_controller = mouse.Controller()


def box_coords(item):
    """Return (x, y, w, h) from a tuple box or an accessibility widget dict."""
    return a11y_box_coords(item)


def hover_diff_boxes_list():
    """Mutable list receiving hover-diff detections. Caller must hold state_lock."""
    if not RUNNING:
        return combined_hover_boxes
    layer = combined_hover_diff_target_layer(COMBINED_PHASE, COMBINED_VIEW)
    if layer is not None:
        return _combined_list_for_layer(layer)
    return combined_hover_boxes


def _combined_list_for_layer(layer: str):
    if layer == "hover":
        return combined_hover_boxes
    if layer == "yolo":
        return combined_yolo_boxes
    if layer == "a11y":
        return combined_a11y_boxes
    if layer == "fused":
        return combined_fused_boxes
    raise ValueError(f"unknown combined layer: {layer}")


_COMBINED_LAYER_COLORS = {
    "hover": OVERLAY_COLOR,
    "yolo": YOLO_DISPLAY_COLOR,
    "a11y": A11Y_DISPLAY_COLOR,
    "fused": FUSED_DISPLAY_COLOR,
}


def active_boxes_list():
    """Mutable box list for the current session. Caller must hold state_lock."""
    if not RUNNING:
        return combined_hover_boxes
    layer = combined_effective_layer(COMBINED_PHASE, COMBINED_VIEW)
    if layer is None:
        return combined_hover_boxes
    return _combined_list_for_layer(layer)


def active_overlay_layers() -> list[tuple[list, QColor]]:
    """Return [(boxes, color), ...] to draw on the overlay."""
    layers = []
    for name in combined_overlay_layers(COMBINED_PHASE, COMBINED_VIEW):
        layers.append((_combined_list_for_layer(name)[:], _COMBINED_LAYER_COLORS[name]))
    return layers


def _clone_box_list(boxes) -> list:
    return copy.deepcopy(list(boxes))


def manual_history_clear() -> None:
    global _manual_nudge_batch, _manual_edit_undo_pushed
    manual_undo_stack.clear()
    manual_redo_stack.clear()
    _manual_nudge_batch = False
    _manual_edit_undo_pushed = False


def manual_history_push() -> None:
    """Snapshot active boxes before a manual mutation. Caller should hold state_lock."""
    global _manual_nudge_batch
    _manual_nudge_batch = False
    snap = {
        "boxes": _clone_box_list(active_boxes_list()),
        **_manual_selection_snap(),
    }
    manual_undo_stack.append(snap)
    if len(manual_undo_stack) > MANUAL_HISTORY_MAX:
        del manual_undo_stack[0 : len(manual_undo_stack) - MANUAL_HISTORY_MAX]
    manual_redo_stack.clear()


def manual_history_push_nudge() -> None:
    """Push at most once for a consecutive nudge sequence."""
    global _manual_nudge_batch
    if _manual_nudge_batch:
        return
    manual_history_push()
    _manual_nudge_batch = True


def manual_history_discard_last_push() -> None:
    if manual_undo_stack:
        manual_undo_stack.pop()


def _restore_manual_snapshot(snap: dict) -> None:
    global manual_edit_mode, manual_edit_anchor
    global manual_edit_origin_box, manual_edit_origin_boxes, manual_edit_edges
    boxes = active_boxes_list()
    boxes[:] = _clone_box_list(snap["boxes"])
    indices = [i for i in snap.get("selected_indices", []) if 0 <= i < len(boxes)]
    primary = snap.get("selected")
    if not indices and primary is not None and 0 <= primary < len(boxes):
        indices = [primary]
    _manual_set_selection(indices, primary=primary if primary in indices else None)
    manual_edit_mode = None
    manual_edit_anchor = None
    manual_edit_origin_box = None
    manual_edit_origin_boxes = None
    manual_edit_edges = None


def manual_undo() -> None:
    global _manual_nudge_batch, _manual_edit_undo_pushed
    with state_lock:
        if not MANUAL_MODE or not manual_undo_stack:
            return
        if COMBINED_MODE and not combined_manual_editing_allowed(COMBINED_PHASE, COMBINED_VIEW):
            return
        current = {
            "boxes": _clone_box_list(active_boxes_list()),
            **_manual_selection_snap(),
        }
        snap = manual_undo_stack.pop()
        manual_redo_stack.append(current)
        _restore_manual_snapshot(snap)
        _manual_nudge_batch = False
        _manual_edit_undo_pushed = False
        n = len(active_boxes_list())
    print(f"  [undo] — Total: {n}", flush=True)
    if overlay_window is not None:
        overlay_window.update()


def manual_redo() -> None:
    global _manual_nudge_batch, _manual_edit_undo_pushed
    with state_lock:
        if not MANUAL_MODE or not manual_redo_stack:
            return
        if COMBINED_MODE and not combined_manual_editing_allowed(COMBINED_PHASE, COMBINED_VIEW):
            return
        current = {
            "boxes": _clone_box_list(active_boxes_list()),
            **_manual_selection_snap(),
        }
        snap = manual_redo_stack.pop()
        manual_undo_stack.append(current)
        _restore_manual_snapshot(snap)
        _manual_nudge_batch = False
        _manual_edit_undo_pushed = False
        n = len(active_boxes_list())
    print(f"  [redo] — Total: {n}", flush=True)
    if overlay_window is not None:
        overlay_window.update()


# ─────────────────────────────────────────────
# OVERLAY
# ─────────────────────────────────────────────
class AppBridge(QObject):
    """Qt bridge to invoke UI actions from non-Qt threads."""
    quit_signal = pyqtSignal()
    manual_mode_signal = pyqtSignal(bool)
    session_config_signal = pyqtSignal()
    combined_fusion_signal = pyqtSignal()
    save_signal = pyqtSignal()
    a11y_filter_signal = pyqtSignal()


class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()

        self._base_flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        if sys.platform.startswith("linux"):
            self._base_flags |= Qt.WindowType.X11BypassWindowManagerHint

        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.set_click_through(True)

        screen = QApplication.primaryScreen()
        self.setGeometry(screen.geometry())

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(OVERLAY_REFRESH_MS)

    def set_click_through(self, enabled: bool):
        flags = self._base_flags
        if enabled:
            flags |= Qt.WindowType.WindowTransparentForInput
        self.setWindowFlags(flags)
        # setWindowFlags can reset attributes / geometry on some platforms.
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        screen = QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())
        self.show()
        self.raise_()
        # Cursor after show(): setWindowFlags/show can reset it on Windows.
        self.setCursor(
            QCursor(Qt.CursorShape.CrossCursor if not enabled else Qt.CursorShape.ArrowCursor)
        )
        self.update()

    def _overlay_to_capture(self, x_overlay: float, y_overlay: float):
        sx = self.width() / CAPTURE_W
        sy = self.height() / CAPTURE_H
        x = int(round(x_overlay / sx))
        y = int(round(y_overlay / sy))
        x = max(0, min(CAPTURE_W - 1, x))
        y = max(0, min(CAPTURE_H - 1, y))
        return x, y

    @staticmethod
    def _point_in_box(x, y, box):
        bx, by, bw, bh = box
        return bx <= x <= bx + bw and by <= y <= by + bh

    @staticmethod
    def _hit_test_box(x, y, box, r):
        bx, by, bw, bh = box
        x0, y0 = bx, by
        x1, y1 = bx + bw, by + bh

        # Edge hit needs proximity on one axis and alignment on the other.
        within_y = (y0 - r) <= y <= (y1 + r)
        within_x = (x0 - r) <= x <= (x1 + r)

        near_left = abs(x - x0) <= r and within_y
        near_right = abs(x - x1) <= r and within_y
        near_top = abs(y - y0) <= r and within_x
        near_bottom = abs(y - y1) <= r and within_x

        if near_left and near_top:
            return ("resize", {"left", "top"})
        if near_right and near_top:
            return ("resize", {"right", "top"})
        if near_left and near_bottom:
            return ("resize", {"left", "bottom"})
        if near_right and near_bottom:
            return ("resize", {"right", "bottom"})

        if near_left:
            return ("resize", {"left"})
        if near_right:
            return ("resize", {"right"})
        if near_top:
            return ("resize", {"top"})
        if near_bottom:
            return ("resize", {"bottom"})

        if OverlayWindow._point_in_box(x, y, box):
            return ("move", set())

        return (None, set())

    def mousePressEvent(self, event):
        global manual_drag_start, manual_drag_kind, manual_preview_box
        global manual_selected_index, manual_selected_indices
        global manual_edit_mode, manual_edit_anchor, manual_edit_origin_box
        global manual_edit_origin_boxes, manual_edit_edges, _manual_edit_undo_pushed

        with state_lock:
            if not MANUAL_MODE:
                return
            if COMBINED_MODE and not combined_manual_editing_allowed(COMBINED_PHASE, COMBINED_VIEW):
                return

        x, y = self._overlay_to_capture(event.position().x(), event.position().y())

        if event.button() == Qt.MouseButton.RightButton:
            with state_lock:
                manual_edit_mode = None
                manual_edit_edges = None
                manual_edit_anchor = None
                manual_edit_origin_box = None
                manual_edit_origin_boxes = None
            _manual_edit_undo_pushed = False
            manual_drag_kind = "erase"
            manual_drag_start = (x, y)
            manual_preview_box = (x, y, 1, 1)
            self.update()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        with state_lock:
            boxes = active_boxes_list()[:]
            selected_set = set(manual_selected_indices)

        hit_index = None
        hit_mode = None
        hit_edges = set()
        for i in range(len(boxes) - 1, -1, -1):
            mode, edges = self._hit_test_box(x, y, box_coords(boxes[i]), MANUAL_HANDLE_RADIUS)
            if mode is not None:
                hit_index = i
                hit_mode = mode
                hit_edges = edges
                break

        if hit_index is not None:
            with state_lock:
                boxes = active_boxes_list()
                manual_history_push()
                _manual_edit_undo_pushed = True
                # Clicking a member of a multi-selection starts a group move.
                group = (
                    hit_mode == "move"
                    and hit_index in selected_set
                    and len(selected_set) > 1
                )
                if group:
                    _manual_set_selection(list(selected_set), primary=hit_index)
                    manual_edit_mode = "move_group"
                    manual_edit_edges = set()
                    manual_edit_anchor = (x, y)
                    manual_edit_origin_box = None
                    manual_edit_origin_boxes = {
                        i: box_coords(boxes[i])
                        for i in manual_selected_indices
                        if 0 <= i < len(boxes)
                    }
                else:
                    _manual_set_selection([hit_index], primary=hit_index)
                    manual_edit_mode = hit_mode
                    manual_edit_edges = hit_edges
                    manual_edit_anchor = (x, y)
                    manual_edit_origin_box = box_coords(boxes[hit_index])
                    manual_edit_origin_boxes = None
            manual_drag_kind = None
            manual_drag_start = None
            manual_preview_box = None
            self.update()
            return

        with state_lock:
            _manual_clear_selection()
        _manual_edit_undo_pushed = False
        manual_drag_kind = "create"
        manual_drag_start = (x, y)
        manual_preview_box = (x, y, 1, 1)
        self.update()

    def mouseMoveEvent(self, event):
        global manual_preview_box
        global manual_selected_index, manual_edit_mode, manual_edit_anchor, manual_edit_origin_box
        global manual_edit_origin_boxes, manual_edit_edges
        with state_lock:
            if not MANUAL_MODE:
                return
            if COMBINED_MODE and not combined_manual_editing_allowed(COMBINED_PHASE, COMBINED_VIEW):
                return

            selected = manual_selected_index
            edit_mode = manual_edit_mode
            anchor = manual_edit_anchor
            origin = manual_edit_origin_box
            origin_boxes = manual_edit_origin_boxes
            drag_start = manual_drag_start
            edges = manual_edit_edges

        x1, y1 = self._overlay_to_capture(event.position().x(), event.position().y())

        if edit_mode == "move_group" and anchor is not None and origin_boxes:
            ax, ay = anchor
            dx = x1 - ax
            dy = y1 - ay
            origins = list(origin_boxes.values())
            lo_dx = max(-ox for ox, _, _, _ in origins)
            hi_dx = min(CAPTURE_W - ow - ox for ox, _, ow, _ in origins)
            lo_dy = max(-oy for _, oy, _, _ in origins)
            hi_dy = min(CAPTURE_H - oh - oy for _, oy, _, oh in origins)
            dx = int(max(lo_dx, min(hi_dx, dx)))
            dy = int(max(lo_dy, min(hi_dy, dy)))
            with state_lock:
                boxes = active_boxes_list()
                for i, (ox, oy, ow, oh) in origin_boxes.items():
                    if not (0 <= i < len(boxes)):
                        continue
                    nx, ny = ox + dx, oy + dy
                    current = boxes[i]
                    if isinstance(current, dict):
                        current["x"] = int(nx)
                        current["y"] = int(ny)
                    else:
                        boxes[i] = (int(nx), int(ny), int(ow), int(oh))
            self.update()
            return

        if selected is not None and edit_mode is not None and anchor is not None and origin is not None:
            ax, ay = anchor
            ox, oy, ow, oh = origin

            dx = x1 - ax
            dy = y1 - ay

            edges = edges or set()

            if edit_mode == "move":
                nx = ox + dx
                ny = oy + dy
                nx = max(0, min(CAPTURE_W - 1, nx))
                ny = max(0, min(CAPTURE_H - 1, ny))
                nx = min(nx, CAPTURE_W - ow)
                ny = min(ny, CAPTURE_H - oh)
                new_box = (int(nx), int(ny), int(ow), int(oh))
            else:
                left = ox
                top = oy
                right = ox + ow
                bottom = oy + oh

                if "left" in edges:
                    left = ox + dx
                if "right" in edges:
                    right = ox + ow + dx
                if "top" in edges:
                    top = oy + dy
                if "bottom" in edges:
                    bottom = oy + oh + dy

                x_min = int(max(0, min(left, right)))
                x_max = int(min(CAPTURE_W - 1, max(left, right)))
                y_min = int(max(0, min(top, bottom)))
                y_max = int(min(CAPTURE_H - 1, max(top, bottom)))

                w = max(1, x_max - x_min)
                h = max(1, y_max - y_min)
                new_box = (x_min, y_min, w, h)

            with state_lock:
                boxes = active_boxes_list()
                if manual_selected_index is not None and 0 <= manual_selected_index < len(boxes):
                    nx, ny, nw, nh = new_box
                    current = boxes[manual_selected_index]
                    if isinstance(current, dict):
                        current["x"] = nx
                        current["y"] = ny
                        current["w"] = nw
                        current["h"] = nh
                    else:
                        boxes[manual_selected_index] = new_box
            self.update()
            return

        if drag_start is None:
            return
        x0, y0 = drag_start
        x = min(x0, x1)
        y = min(y0, y1)
        w = max(1, abs(x1 - x0))
        h = max(1, abs(y1 - y0))
        manual_preview_box = (x, y, w, h)
        self.update()

    def mouseReleaseEvent(self, event):
        global manual_drag_start, manual_drag_kind, manual_preview_box
        global manual_selected_index, manual_edit_mode, manual_edit_anchor, manual_edit_origin_box
        global manual_edit_origin_boxes, manual_edit_edges, _manual_edit_undo_pushed
        with state_lock:
            if not MANUAL_MODE:
                return
            if COMBINED_MODE and not combined_manual_editing_allowed(COMBINED_PHASE, COMBINED_VIEW):
                return

        if event.button() == Qt.MouseButton.RightButton:
            box = manual_preview_box
            kind = manual_drag_kind
            x_click, y_click = self._overlay_to_capture(event.position().x(), event.position().y())
            manual_drag_start = None
            manual_drag_kind = None
            manual_preview_box = None

            if kind != "erase" or box is None:
                self.update()
                return

            x, y, w, h = box
            if w * h < NEW_DIFF_MIN_AREA:
                with state_lock:
                    boxes = active_boxes_list()
                    sel = manual_selected_index
                    if sel is not None and 0 <= sel < len(boxes):
                        bx, by, bw, bh = box_coords(boxes[sel])
                        if self._point_in_box(x_click, y_click, (bx, by, bw, bh)):
                            manual_history_push()
                            boxes.pop(sel)
                            _manual_clear_selection()
                            print(f"  [-] Removed selected bbox — Total: {len(boxes)}")
                self.update()
                return

            with state_lock:
                boxes = active_boxes_list()
                before = len(boxes)
                keep = [item for item in boxes if not box_contains(box, item, margin=0)]
                removed = before - len(keep)
                if removed:
                    manual_history_push()
                    boxes[:] = keep
                    _manual_clear_selection()
            if removed:
                print(f"  [-] Erased {removed} enclosed bbox(es) — Total: {before - removed}")
            self.update()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        with state_lock:
            if manual_edit_mode is not None:
                origin = manual_edit_origin_box
                origin_boxes = manual_edit_origin_boxes
                edit_mode = manual_edit_mode
                sel = manual_selected_index
                boxes = active_boxes_list()
                changed = True
                if _manual_edit_undo_pushed:
                    if edit_mode == "move_group" and origin_boxes:
                        changed = any(
                            0 <= i < len(boxes) and box_coords(boxes[i]) != origin_boxes[i]
                            for i in origin_boxes
                        )
                    elif (
                        origin is not None
                        and sel is not None
                        and 0 <= sel < len(boxes)
                    ):
                        changed = box_coords(boxes[sel]) != origin
                    else:
                        changed = False
                if _manual_edit_undo_pushed and not changed:
                    manual_history_discard_last_push()
                _manual_edit_undo_pushed = False
                manual_edit_mode = None
                manual_edit_anchor = None
                manual_edit_origin_box = None
                manual_edit_origin_boxes = None
                manual_edit_edges = None
                return

        if manual_drag_start is None:
            return

        box = manual_preview_box
        manual_drag_start = None
        manual_drag_kind = None
        manual_preview_box = None

        if box is None:
            return

        x, y, w, h = box
        if w * h < NEW_DIFF_MIN_AREA:
            return

        with state_lock:
            boxes = active_boxes_list()
            enclosed = [
                i for i, item in enumerate(boxes)
                if box_contains(box, item, margin=0)
            ]
            if enclosed:
                # Marquee: select enclosed boxes instead of creating a new one.
                _manual_set_selection(enclosed)
                print(f"  [=] Selected {len(enclosed)} bbox(es)")
            else:
                layer = combined_effective_layer(COMBINED_PHASE, COMBINED_VIEW)
                use_a11y_widget = layer == "a11y"
                if should_append_box(box, boxes):
                    manual_history_push()
                    if use_a11y_widget:
                        boxes.append(make_manual_a11y_widget(x, y, w, h))
                    else:
                        boxes.append(box)
                    print(f"  [+] Manual bbox ({x}, {y}) size {w}x{h} — Total: {len(boxes)}")
        self.update()

    def paintEvent(self, event):
        with state_lock:
            layers = active_overlay_layers()
            running = RUNNING
            manual = MANUAL_MODE
            preview = manual_preview_box
            preview_erase = manual_drag_kind == "erase"
            selected = manual_selected_index
            selected_set = set(manual_selected_indices)
            fusion_highlight_idx = COMBINED_FUSION_HIGHLIGHT_IDX
            fusion_clusters = (
                COMBINED_FUSION_HIGHLIGHT_CLUSTERS[:]
                if COMBINED_FUSION_HIGHLIGHT_IDX >= 0
                else []
            )

        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Manual mode: near-opaque full-screen fill so Windows hit-tests receive mouse events.
        if manual:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 1))

        has_highlight = fusion_highlight_idx >= 0 and fusion_clusters
        has_boxes = any(layer_boxes for layer_boxes, _ in layers)
        if not (running or manual or has_highlight) or (not has_boxes and preview is None):
            painter.end()
            return

        sx = self.width() / CAPTURE_W
        sy = self.height() / CAPTURE_H

        for layer_boxes, box_color in layers:
            pen = QPen(box_color)
            pen.setWidth(OVERLAY_LINE_WIDTH)
            painter.setPen(pen)

            for idx, item in enumerate(layer_boxes):
                x, y, w, h = box_coords(item)
                xd = round(x * sx)
                yd = round(y * sy)
                wd = round(w * sx)
                hd = round(h * sy)
                is_selected = (
                    manual
                    and len(layers) == 1
                    and (idx in selected_set or (selected is not None and idx == selected))
                )
                show_handles = is_selected and len(selected_set) <= 1
                if is_selected:
                    pen_sel = QPen(QColor(0, 200, 255, 240))
                    pen_sel.setWidth(max(1, OVERLAY_LINE_WIDTH + 1))
                    painter.setPen(pen_sel)
                else:
                    painter.setPen(pen)
                painter.drawRect(xd, yd, wd, hd)

                if show_handles:
                    handle_pen = QPen(QColor(0, 200, 255, 240))
                    handle_pen.setWidth(1)
                    painter.setPen(handle_pen)
                    handle_brush = QColor(0, 200, 255, 160)
                    hs = max(4, int(round(MANUAL_HANDLE_RADIUS * max(sx, sy))))
                    for hx, hy in [
                        (xd, yd),
                        (xd + wd, yd),
                        (xd, yd + hd),
                        (xd + wd, yd + hd),
                    ]:
                        painter.fillRect(int(hx - hs // 2), int(hy - hs // 2), hs, hs, handle_brush)

        if has_highlight and fusion_highlight_idx < len(fusion_clusters):
            cluster = fusion_clusters[fusion_highlight_idx]
            total = len(fusion_clusters)
            ux, uy, uw, uh = cluster_union_rect(cluster)
            hx, hy, hw, hh = round(ux * sx), round(uy * sy), round(uw * sx), round(uh * sy)

            painter.fillRect(hx, hy, hw, hh, QColor(255, 255, 0, 55))

            highlight = QPen(QColor(255, 255, 0, 255))
            highlight.setWidth(max(3, OVERLAY_LINE_WIDTH + 3))
            highlight.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(highlight)
            painter.drawRect(hx, hy, hw, hh)

            for src, _, box in cluster:
                x, y, w, h = box_coords(box)
                pick_pen = QPen(_COMBINED_LAYER_COLORS[src])
                pick_pen.setWidth(max(3, OVERLAY_LINE_WIDTH + 3))
                painter.setPen(pick_pen)
                painter.drawRect(round(x * sx), round(y * sy), round(w * sx), round(h * sy))

            banner_w = min(self.width() - 40, 920)
            painter.fillRect(20, 20, banner_w, 56, QColor(0, 0, 0, 200))
            font = QFont("Segoe UI", 12, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                32, 44,
                f"Widget {fusion_highlight_idx + 1} / {total} — choisissez la source dans la fenêtre",
            )

        if preview is not None:
            color = QColor(255, 60, 60, 230) if preview_erase else QColor(255, 255, 0, 220)
            pen_preview = QPen(color)
            pen_preview.setWidth(max(1, OVERLAY_LINE_WIDTH))
            if preview_erase:
                pen_preview.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen_preview)
            x, y, w, h = preview
            painter.drawRect(round(x * sx), round(y * sy), round(w * sx), round(h * sy))
            if preview_erase:
                painter.fillRect(
                    round(x * sx), round(y * sy), round(w * sx), round(h * sy),
                    QColor(255, 40, 40, 40),
                )

        painter.end()


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
def capture_screen():
    """Capture full screen as a BGR uint8 image."""
    global CAPTURE_LEFT, CAPTURE_TOP, CAPTURE_W, CAPTURE_H
    if not hasattr(mss_local, "sct"):
        mss_local.sct = mss.mss()

    monitor = mss_local.sct.monitors[1]
    shot = mss_local.sct.grab(monitor)
    # Keep capture geometry in sync with MSS (physical pixels).
    CAPTURE_LEFT = int(monitor.get("left", 0))
    CAPTURE_TOP = int(monitor.get("top", 0))
    CAPTURE_W = int(shot.width)
    CAPTURE_H = int(shot.height)
    return np.array(shot)[:, :, :3]


def get_cursor_capture_coords():
    """
    Return cursor position in the MSS capture coordinate space.
    This compensates for DPI scaling mismatches between PyAutoGUI and MSS.
    """
    cx, cy = mouse_controller.position
    x = int(round(cx - CAPTURE_LEFT))
    y = int(round(cy - CAPTURE_TOP))

    x = max(0, min(CAPTURE_W - 1, x))
    y = max(0, min(CAPTURE_H - 1, y))
    return x, y


def capture_to_screen_coords(x, y):
    """Convert capture coords to absolute screen coords (MSS monitor space)."""
    return int(CAPTURE_LEFT + x), int(CAPTURE_TOP + y)


def move_cursor_system(x_screen: int, y_screen: int):
    """Move system cursor; on Windows use SendInput (taskbar hovers ignore pynput)."""
    if sys.platform.startswith("win"):
        import ctypes
        user32 = ctypes.windll.user32
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        if screen_w <= 1 or screen_h <= 1:
            mouse_controller.position = (x_screen, y_screen)
            return

        # SendInput absolute coords are in 0..65535.
        abs_x = int(x_screen * 65535 / (screen_w - 1))
        abs_y = int(y_screen * 65535 / (screen_h - 1))

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]

        inp = INPUT(
            type=0,
            mi=MOUSEINPUT(
                dx=abs_x,
                dy=abs_y,
                mouseData=0,
                dwFlags=0x8000 | 0x0001,  # MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE
                time=0,
                dwExtraInfo=None,
            ),
        )
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
        return

    mouse_controller.position = (x_screen, y_screen)


def load_yolo_model():
    """Load YOLO model lazily (Ultralytics)."""
    global yolo_model
    if yolo_model is not None:
        return yolo_model

    model_path = YOLO_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"[WARN] YOLO model not found: {model_path}")
        return None

    try:
        yolo_model = YOLO(model_path)
        print(f"YOLO loaded: {model_path}")
        return yolo_model
    except Exception as e:
        print(f"[WARN] Failed to load YOLO model: {e}")
        return None


def yolo_infer_boxes_bgr(image_bgr):
    """Run YOLO on a BGR image and return list of (x, y, w, h) in capture coords."""
    model = load_yolo_model()
    if model is None:
        return []

    try:
        results = model.predict(image_bgr, conf=YOLO_CONF, verbose=False, end2end=True)
    except Exception as e:
        print(f"[WARN] YOLO inference failed: {e}")
        return []

    if not results:
        return []

    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    try:
        xyxy = r0.boxes.xyxy.cpu().numpy()
    except Exception:
        xyxy = np.array(r0.boxes.xyxy)

    boxes = []
    for x1, y1, x2, y2 in xyxy:
        x1 = int(round(max(0, min(CAPTURE_W - 1, x1))))
        y1 = int(round(max(0, min(CAPTURE_H - 1, y1))))
        x2 = int(round(max(0, min(CAPTURE_W - 1, x2))))
        y2 = int(round(max(0, min(CAPTURE_H - 1, y2))))

        x = min(x1, x2)
        y = min(y1, y2)
        w = max(1, abs(x2 - x1))
        h = max(1, abs(y2 - y1))
        boxes.append((x, y, w, h))

    # Row bins + right-to-left so slight y jitter doesn't scramble scan order.
    def sort_key(b):
        x, y, w, h = b
        y_center = y + h / 2.0
        row = int(round(y_center / max(1, YOLO_ROW_BIN_PX)))
        return (row, -x)

    boxes.sort(key=sort_key)
    return boxes


def yolo_hover_points_for_box(box):
    """Return hover points (x, y) inside the box in capture coordinates."""
    x, y, w, h = box
    pts = []
    for name in YOLO_HOVER_POINTS:
        if name == "top_right":
            px = x + w - 2
            py = y + 1
        elif name == "bottom_left":
            px = x + 1
            py = y + h - 2
        else:  # center
            px = x + w // 2
            py = y + h // 2

        px = max(0, min(CAPTURE_W - 1, int(px)))
        py = max(0, min(CAPTURE_H - 1, int(py)))
        pts.append((px, py))
    return pts

def compute_diff(img1, img2):
    """Return a binary mask of changed pixels."""
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    return mask


def cursor_inside_contour(contour, cx, cy):
    """True if the cursor is inside the contour."""
    return cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0


def extract_box(mask):
    """Pick the contour under the mouse and return its bounding box.
    Applies light morphological closing on the mask to reduce 1px noise, then rejects
    low-solidity contours (merged hover+tooltip / concave blobs).
    """
    k = max(3, int(EXTRACT_BOX_MORPH_CLOSE_K) | 1)  # force odd >= 3
    kernel = np.ones((k, k), np.uint8)
    smoothed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=max(0, int(EXTRACT_BOX_MORPH_CLOSE_ITER)),
    )

    contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cx, cy = get_cursor_capture_coords()
    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity < EXTRACT_BOX_MIN_SOLIDITY:
            continue

        x, y, w, h = cv2.boundingRect(c)

        inside_dist = cv2.pointPolygonTest(c, (float(cx), float(cy)), True)
        if inside_dist >= 0:
            candidates.append((inside_dist, (int(x), int(y), int(w), int(h))))

    if not candidates:
        return None

    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][1]


def boxes_close(b1, b2, tol=3):
    """Loose equality for bbox stability."""
    x1, y1, w1, h1 = box_coords(b1)
    x2, y2, w2, h2 = box_coords(b2)
    return (
        abs(x1 - x2) <= tol and
        abs(y1 - y2) <= tol and
        abs(w1 - w2) <= tol and
        abs(h1 - h2) <= tol
    )


def box_contains(outer, inner, margin=2):
    """True if outer contains inner (with margin)."""
    xo, yo, wo, ho = box_coords(outer)
    xi, yi, wi, hi = box_coords(inner)
    return (
        xo <= xi + margin and
        yo <= yi + margin and
        xo + wo >= xi + wi - margin and
        yo + ho >= yi + hi - margin
    )


def should_append_box(new_box, existing_boxes):
    """Reject duplicates and nested boxes."""
    for old_box in existing_boxes:
        if boxes_close(new_box, old_box, tol=3):
            return False

        if box_contains(new_box, old_box, margin=2):
            return False

        if box_contains(old_box, new_box, margin=2):
            return False

    return True


def build_overlay_mask(shape):
    """Mask overlay rectangles so the diff ignores them."""
    mask = np.zeros(shape[:2], dtype=np.uint8)

    with state_lock:
        running = RUNNING
        items = (
            combined_hover_boxes
            + combined_yolo_boxes
            + combined_a11y_boxes
            + combined_fused_boxes
        )

    if not running or not items:
        return mask

    thickness = OVERLAY_LINE_WIDTH + 2 * OVERLAY_IGNORE_MARGIN

    for item in items:
        x, y, w, h = box_coords(item)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=thickness)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def remove_overlay_from_current(base_img, current_img):
    """Replace overlay pixels with base pixels."""
    overlay_mask = build_overlay_mask(current_img.shape)

    if not np.any(overlay_mask):
        return current_img

    cleaned = current_img.copy()
    cleaned[overlay_mask > 0] = base_img[overlay_mask > 0]
    return cleaned


# ─────────────────────────────────────────────
# SAUVEGARDE
# ─────────────────────────────────────────────
# BGR colors matching overlay QColor (RGB).
_SAVE_BGR_COLORS = {
    "hover": (0, 255, 0),
    "yolo": (0, 140, 255),
    "a11y": (255, 120, 0),
    "fused": (255, 255, 255),
}


def save_results(selected: dict, base_img, output_dir: str = OUTPUT_DIR, separate_files: bool = False):
    """Save annotated PNG + JSON for the chosen sources (combined or per-source)."""
    flat = [box for boxes in selected.values() for box in boxes]
    if not flat:
        print("[!] Aucune détection à enregistrer.")
        return None, None

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_h, img_w = base_img.shape[:2]
    base_img_path = os.path.join(output_dir, f"base_{timestamp}.png")
    cv2.imwrite(base_img_path, base_img)
    print(f"[DONE] Base image: {base_img_path}")

    def _write_one(source_key: str | None, subset: dict) -> tuple[str, str]:
        payload = build_save_payload(subset, img_w, img_h, timestamp=timestamp)
        annotated = base_img.copy()
        thickness = max(2, OVERLAY_LINE_WIDTH or 2)
        for entry in payload["annotations"]:
            x = entry["bbox"]["x"]
            y = entry["bbox"]["y"]
            w = entry["bbox"]["w"]
            h = entry["bbox"]["h"]
            color = _SAVE_BGR_COLORS.get(entry.get("source", ""), (0, 255, 0))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        suffix = f"_{source_key}" if source_key else ""
        img_path = os.path.join(output_dir, f"annotated{suffix}_{timestamp}.png")
        json_path = os.path.join(output_dir, f"annotations{suffix}_{timestamp}.json")
        cv2.imwrite(img_path, annotated)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(
            f"[DONE] {payload['total_widgets']} widgets ({', '.join(payload['sources'])}) — {img_path}"
        )
        print(f"[DONE] JSON: {json_path}")
        return img_path, json_path

    if separate_files and len(selected) > 1:
        last = None
        for source, boxes in selected.items():
            last = _write_one(source, {source: boxes})
        return last

    return _write_one(None, selected)


def _end_session_after_save() -> None:
    """Clear session state after a successful save, then reopen config."""
    global RUNNING, YOLO_AUTOSCAN, A11Y_SCAN_PENDING
    global COMBINED_MODE, COMBINED_PHASE, COMBINED_CONFIG, COMBINED_VIEW, COMBINED_AUTO_FUSED
    global combined_hover_boxes, combined_yolo_boxes, combined_a11y_boxes, combined_a11y_raw_boxes, combined_fused_boxes
    global COMBINED_FUSION_HIGHLIGHT_IDX, COMBINED_FUSION_HIGHLIGHT_CLUSTERS

    with state_lock:
        RUNNING = False
        YOLO_AUTOSCAN = False
        A11Y_SCAN_PENDING = False
        COMBINED_MODE = False
        COMBINED_PHASE = ""
        COMBINED_CONFIG = None
        COMBINED_AUTO_FUSED = False
        COMBINED_FUSION_HIGHLIGHT_IDX = -1
        COMBINED_FUSION_HIGHLIGHT_CLUSTERS = []
        combined_hover_boxes = []
        combined_yolo_boxes = []
        combined_a11y_boxes = []
        combined_a11y_raw_boxes = []
        combined_fused_boxes = []

    _fusion_wizard_highlight(-1, [])
    if app_bridge is not None:
        app_bridge.manual_mode_signal.emit(False)
    request_new_session()


def on_save_request() -> None:
    """Qt-thread save flow: dialog (sources + path + JSON preview) then write files."""
    with state_lock:
        if not RUNNING:
            request_new_session()
            return
        available = {
            "hover": combined_hover_boxes[:],
            "yolo": combined_yolo_boxes[:],
            "a11y": combined_a11y_boxes[:],
            "fused": combined_fused_boxes[:],
        }
        base_img = initial_img.copy() if initial_img is not None else None

    if base_img is None:
        return

    available = {k: v for k, v in available.items() if v}
    if not available:
        print("[!] Aucune détection à enregistrer.")
        return

    img_h, img_w = base_img.shape[:2]
    result = show_save_config_dialog(
        available=available,
        image_width=img_w,
        image_height=img_h,
        default_dir=OUTPUT_DIR,
        parent=overlay_window,
    )
    if result.action == "cancel":
        _sync_manual_cursor()
        return
    if result.action == "discard":
        _end_session_after_save()
        return

    selection = result.selection
    if selection is None:
        _sync_manual_cursor()
        return
    save_results(
        result.selected_boxes,
        base_img,
        output_dir=selection.output_dir,
        separate_files=selection.separate_files,
    )
    _end_session_after_save()


def request_save() -> None:
    """Thread-safe save request (called from keyboard listener)."""
    if app_bridge is not None:
        app_bridge.save_signal.emit()


# ─────────────────────────────────────────────
# THREAD DE SURVEILLANCE
# ─────────────────────────────────────────────
def screen_watcher():
    """Background loop: capture → diff → bbox under cursor."""
    global EXIT_PROGRAM, RUNNING, initial_img

    prev_mask_raw = None
    ignored_transient_mask = None  # uint8 mask of ignored regions.

    while not EXIT_PROGRAM:
        with state_lock:
            running = RUNNING
            base_img = initial_img
            manual = MANUAL_MODE
            combined = COMBINED_MODE
            combined_phase = COMBINED_PHASE
            combined_view = COMBINED_VIEW
            fusion_picking = COMBINED_FUSION_HIGHLIGHT_IDX >= 0

        hover_diff_ok = (
            (not combined or combined_hover_diff_enabled(combined_phase, combined_view))
            and not fusion_picking
        )
        if manual or not running or base_img is None or not hover_diff_ok:
            time.sleep(0.05)
            continue

        current = capture_screen()
        current_clean = remove_overlay_from_current(base_img, current)
        mask_raw = compute_diff(base_img, current_clean)

        if IGNORE_NEW_DIFF_AWAY_FROM_CURSOR:
            if ignored_transient_mask is None or ignored_transient_mask.shape != mask_raw.shape:
                ignored_transient_mask = np.zeros(mask_raw.shape, dtype=np.uint8)

            if prev_mask_raw is not None:
                new_pixels = cv2.bitwise_and(mask_raw, cv2.bitwise_not(prev_mask_raw))
                if np.any(new_pixels):
                    cx, cy = get_cursor_capture_coords()
                    contours, _ = cv2.findContours(new_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        if cv2.contourArea(c) < NEW_DIFF_MIN_AREA:
                            continue
                        if not cursor_inside_contour(c, cx, cy):
                            cv2.drawContours(ignored_transient_mask, [c], -1, 255, thickness=-1)

            ignored_transient_mask = cv2.bitwise_and(ignored_transient_mask, mask_raw)

            if NEW_DIFF_DILATE > 0 and np.any(ignored_transient_mask):
                k = cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (NEW_DIFF_DILATE * 2 + 1, NEW_DIFF_DILATE * 2 + 1)
                )
                ignored_transient_mask = cv2.dilate(ignored_transient_mask, k, iterations=1)

            mask = mask_raw.copy()
            if np.any(ignored_transient_mask):
                mask[ignored_transient_mask > 0] = 0
        else:
            mask = mask_raw

        box = extract_box(mask)

        if box is not None:
            with state_lock:
                if RUNNING:
                    target = hover_diff_boxes_list()
                    if should_append_box(box, target):
                        target.append(box)
                        print(f"  [+] Widget ({box[0]}, {box[1]}) "
                              f"size {box[2]}x{box[3]} — Total: {len(target)}")

        prev_mask_raw = mask_raw
        time.sleep(LOOP_SLEEP)


def yolo_sweeper():
    """Background loop that moves the cursor over YOLO detections to trigger hover changes."""
    global YOLO_AUTOSCAN

    last_session_id = None

    while not EXIT_PROGRAM:
        with state_lock:
            running = RUNNING
            manual = MANUAL_MODE
            autoscan = YOLO_AUTOSCAN
            base_img = initial_img.copy() if initial_img is not None else None

        if not running or manual or not autoscan or base_img is None:
            time.sleep(0.05)
            continue

        # Session identity: restart sweep when base_img changes.
        session_id = id(base_img)
        if session_id == last_session_id:
            time.sleep(0.05)
            continue

        last_session_id = session_id

        boxes = yolo_infer_boxes_bgr(base_img)
        if not boxes:
            print("YOLO autoscan: no detections.")
            with state_lock:
                YOLO_AUTOSCAN = False
            continue

        print(f"YOLO autoscan: {len(boxes)} boxes.")

        for box in boxes:
            with state_lock:
                if EXIT_PROGRAM or not RUNNING or MANUAL_MODE or not YOLO_AUTOSCAN:
                    break

            for (px, py) in yolo_hover_points_for_box(box):
                with state_lock:
                    if EXIT_PROGRAM or not RUNNING or MANUAL_MODE or not YOLO_AUTOSCAN:
                        break

                sx, sy = capture_to_screen_coords(px, py)
                try:
                    move_cursor_system(sx, sy)
                except Exception:
                    pass
                time.sleep(YOLO_MOVE_DELAY_S)

        with state_lock:
            YOLO_AUTOSCAN = False


def _a11y_scan_worker(capture_left, capture_top, capture_width, capture_height):
    """Run UI Automation scan off the main keyboard thread."""
    global combined_a11y_boxes, combined_a11y_raw_boxes, A11Y_SCAN_PENDING

    try:
        raw = get_accessibility_boxes_raw(
            capture_left,
            capture_top,
            capture_width,
            capture_height,
        )
    except Exception as exc:
        print(f"[WARN] Accessibility scan failed: {exc}", flush=True)
        raw = []

    filtered = apply_a11y_filters(raw)
    open_filter_ui = False
    with state_lock:
        if RUNNING and COMBINED_PHASE == "a11y":
            combined_a11y_raw_boxes = raw
            combined_a11y_boxes = filtered
            print(
                f"Accessibilité: {len(filtered)} box(es) "
                f"(filtre défaut, {len(raw)} brute(s)).",
                flush=True,
            )
            open_filter_ui = True
        A11Y_SCAN_PENDING = False

    if open_filter_ui and app_bridge is not None:
        app_bridge.a11y_filter_signal.emit()


def _preview_a11y_filters(filtered: list) -> None:
    global combined_a11y_boxes
    with state_lock:
        combined_a11y_boxes = list(filtered)
    if overlay_window is not None:
        overlay_window.update()


def on_a11y_filter_request() -> None:
    """Qt-thread: open live a11y filter dialog after the raw scan."""
    with state_lock:
        if not (RUNNING and COMBINED_MODE and COMBINED_PHASE == "a11y"):
            return
        raw = combined_a11y_raw_boxes[:]

    if overlay_window is not None:
        overlay_window.set_click_through(True)

    try:
        filtered = show_a11y_filter_dialog(
            raw_boxes=raw,
            on_preview=_preview_a11y_filters,
            parent=overlay_window,
        )
        _preview_a11y_filters(filtered)
        print(f"Accessibilité: {len(filtered)} box(es) après filtres.", flush=True)
    finally:
        if overlay_window is not None:
            overlay_window.set_click_through(not MANUAL_MODE)
        _sync_manual_cursor()
        if overlay_window is not None:
            overlay_window.raise_()
            overlay_window.update()


def _print_combined_phase_help(phase: str) -> None:
    labels = {
        "hover": "HOVER",
        "yolo": "YOLO",
        "a11y": "ACCESSIBILITÉ",
        "review": "REVUE",
    }
    if phase in labels:
        print(f"→ Phase {labels[phase]}", flush=True)


def _begin_combined_phase(phase: str) -> None:
    global COMBINED_PHASE, YOLO_AUTOSCAN, A11Y_SCAN_PENDING, COMBINED_VIEW
    global combined_yolo_boxes

    COMBINED_PHASE = phase
    YOLO_AUTOSCAN = False
    manual_history_clear()

    if phase == "hover":
        if COMBINED_CONFIG and COMBINED_CONFIG.hover_autoscan:
            YOLO_AUTOSCAN = True
    elif phase == "yolo":
        with state_lock:
            base = initial_img
        if base is not None:
            combined_yolo_boxes.clear()
            combined_yolo_boxes.extend(yolo_infer_boxes_bgr(base))
            print(f"YOLO: {len(combined_yolo_boxes)} box(es).", flush=True)
    elif phase == "a11y":
        A11Y_SCAN_PENDING = True
        threading.Thread(
            target=_a11y_scan_worker,
            args=(CAPTURE_LEFT, CAPTURE_TOP, CAPTURE_W, CAPTURE_H),
            daemon=True,
        ).start()
    elif phase == "review":
        COMBINED_VIEW = "all"
        YOLO_AUTOSCAN = False
        if MANUAL_MODE and app_bridge is not None:
            app_bridge.manual_mode_signal.emit(False)

    _print_combined_phase_help(phase)


def advance_combined_phase() -> None:
    global COMBINED_PHASE, COMBINED_PHASES_PENDING, YOLO_AUTOSCAN

    with state_lock:
        if not COMBINED_MODE:
            return
        if COMBINED_PHASE == "a11y" and A11Y_SCAN_PENDING:
            print("Scan accessibilité en cours…", flush=True)
            return
        if not COMBINED_PHASES_PENDING:
            if COMBINED_PHASE != "review":
                _begin_combined_phase("review")
            return
        YOLO_AUTOSCAN = False
        next_phase = COMBINED_PHASES_PENDING.pop(0)
    _begin_combined_phase(next_phase)


def _priority_label(priority: tuple) -> str:
    return " > ".join(SOURCE_LABELS.get(s, s) for s in priority)


def _fusion_wizard_highlight(index: int, clusters: list) -> None:
    global COMBINED_FUSION_HIGHLIGHT_IDX, COMBINED_FUSION_HIGHLIGHT_CLUSTERS

    with state_lock:
        COMBINED_FUSION_HIGHLIGHT_IDX = index
        COMBINED_FUSION_HIGHLIGHT_CLUSTERS = clusters if index >= 0 else []
    if overlay_window is not None:
        overlay_window.update()


def _preview_fusion(fused: list) -> None:
    """Live fusion preview while the config dialog is open."""
    global COMBINED_VIEW
    with state_lock:
        combined_fused_boxes.clear()
        combined_fused_boxes.extend(fused)
        COMBINED_VIEW = "fused"
    if overlay_window is not None:
        overlay_window.update()


def _apply_fusion_result(fused: list, priority: tuple | None = None) -> None:
    global COMBINED_VIEW, COMBINED_AUTO_FUSED

    with state_lock:
        combined_fused_boxes.clear()
        combined_fused_boxes.extend(fused)
        COMBINED_VIEW = "fused"
        COMBINED_AUTO_FUSED = True
        count = len(combined_fused_boxes)

    if priority is not None:
        print(f"Fusion: {count} box(es) ({_priority_label(priority)}).", flush=True)
    else:
        print(f"Fusion: {count} box(es).", flush=True)


def on_combined_fusion_request() -> None:
    global COMBINED_VIEW

    if overlay_window is None:
        return

    with state_lock:
        if not (COMBINED_MODE and COMBINED_PHASE == "review"):
            return
        if COMBINED_AUTO_FUSED:
            return
        if COMBINED_VIEW != "all":
            return
        hover = combined_hover_boxes[:]
        yolo = combined_yolo_boxes[:]
        a11y = combined_a11y_boxes[:]
        prev_fused = combined_fused_boxes[:]
        prev_view = COMBINED_VIEW

    try:
        config = show_fusion_config_dialog(
            hover_boxes=hover,
            yolo_boxes=yolo,
            a11y_boxes=a11y,
            on_preview=_preview_fusion,
            parent=overlay_window,
        )
        if config is None:
            with state_lock:
                combined_fused_boxes.clear()
                combined_fused_boxes.extend(prev_fused)
                COMBINED_VIEW = prev_view
            if overlay_window is not None:
                overlay_window.update()
            return

        fused = compute_auto_fused_boxes(hover, yolo, a11y, config)
        if not fused:
            print(
                f"Aucun widget à fusionner "
                f"(IoU ≥ {config.min_iou:.0%}"
                f"{' ou inclusion 100%' if config.strict_inclusion else ''}).",
                flush=True,
            )
            with state_lock:
                combined_fused_boxes.clear()
                combined_fused_boxes.extend(prev_fused)
                COMBINED_VIEW = prev_view
            if overlay_window is not None:
                overlay_window.update()
            return

        _apply_fusion_result(fused, priority=config.source_priority)
    finally:
        # Modal dialogs often reset the cursor / input flags on Windows.
        if overlay_window is not None:
            overlay_window.set_click_through(not MANUAL_MODE)
        _sync_manual_cursor()
        if overlay_window is not None:
            overlay_window.raise_()
            overlay_window.update()


def request_combined_fusion() -> None:
    if app_bridge is None:
        return
    app_bridge.combined_fusion_signal.emit()


def run_combined_auto_fusion() -> None:
    """Keyboard thread entry: open fusion dialog on Qt thread."""
    with state_lock:
        if not (COMBINED_MODE and COMBINED_PHASE == "review"):
            return
    request_combined_fusion()


def combined_cycle_view(backward: bool = False) -> None:
    global COMBINED_VIEW

    disable_manual = False
    with state_lock:
        if not (COMBINED_MODE and COMBINED_PHASE == "review"):
            return
        COMBINED_VIEW = cycle_view(COMBINED_VIEW, backward=backward)
        view = COMBINED_VIEW
        manual_history_clear()
        if view == "all" and MANUAL_MODE:
            disable_manual = True

    if disable_manual and app_bridge is not None:
        app_bridge.manual_mode_signal.emit(False)


def start_session(config: CombinedModeConfig) -> None:
    global RUNNING, initial_img, YOLO_AUTOSCAN
    global COMBINED_MODE, COMBINED_CONFIG, COMBINED_PHASES_PENDING, COMBINED_VIEW, COMBINED_AUTO_FUSED
    global combined_hover_boxes, combined_yolo_boxes, combined_a11y_boxes, combined_a11y_raw_boxes, combined_fused_boxes
    global COMBINED_FUSION_HIGHLIGHT_IDX, COMBINED_FUSION_HIGHLIGHT_CLUSTERS

    with state_lock:
        if RUNNING:
            return

        combined_hover_boxes = []
        combined_yolo_boxes = []
        combined_a11y_boxes = []
        combined_a11y_raw_boxes = []
        combined_fused_boxes = []
        COMBINED_FUSION_HIGHLIGHT_IDX = -1
        COMBINED_FUSION_HIGHLIGHT_CLUSTERS = []
        # Flush any pending hide/paint so the closed dialog is not in the baseline.
        QApplication.processEvents()
        initial_img = capture_screen()
        RUNNING = True
        COMBINED_MODE = True
        COMBINED_CONFIG = config
        COMBINED_PHASES_PENDING = config.phase_order()[1:]
        COMBINED_VIEW = "all"
        COMBINED_AUTO_FUSED = False
        first_phase = config.phase_order()[0] if config.phase_order() else "review"

    if app_bridge is not None:
        app_bridge.manual_mode_signal.emit(False)

    if first_phase == "review":
        _begin_combined_phase("review")
    else:
        _begin_combined_phase(first_phase)


def prompt_session_start() -> None:
    if overlay_window is None:
        return
    overlay_window.raise_()
    overlay_window.activateWindow()
    config = show_session_config_dialog(a11y_available=is_a11y_available(), parent=overlay_window)
    if config is not None:
        # Defer baseline capture: mss grabs pixels immediately, while the desktop
        # compositor (esp. Linux) may still show the closing dialog for a short time.
        QApplication.processEvents()
        QTimer.singleShot(BASELINE_CAPTURE_SETTLE_MS, lambda c=config: start_session(c))
    else:
        quit_program()


def request_new_session() -> None:
    """Open the session config dialog on the Qt thread (safe from keyboard listener)."""
    if app_bridge is not None:
        app_bridge.session_config_signal.emit()


def on_advance_or_fusion() -> None:
    with state_lock:
        if not RUNNING:
            request_new_session()
            return
        phase = COMBINED_PHASE
    if phase != "review":
        advance_combined_phase()
        return
    run_combined_auto_fusion()


def request_toggle_manual_mode() -> None:
    """Thread-safe manual mode toggle (called from keyboard listener)."""
    if app_bridge is None:
        return
    with state_lock:
        if not RUNNING:
            return
        next_state = not MANUAL_MODE
        if next_state and COMBINED_FUSION_HIGHLIGHT_IDX >= 0:
            return
        if next_state and not combined_manual_editing_allowed(COMBINED_PHASE, COMBINED_VIEW):
            return
    app_bridge.manual_mode_signal.emit(next_state)


# ─────────────────────────────────────────────
# CONTROLES
# ─────────────────────────────────────────────
def _sync_manual_cursor() -> None:
    """Keep CrossCursor while MANUAL_MODE is on (survives modal fusion dialogs)."""
    global _manual_cursor_override_active

    if MANUAL_MODE:
        cross = QCursor(Qt.CursorShape.CrossCursor)
        if overlay_window is not None:
            overlay_window.setCursor(cross)
        if QApplication.overrideCursor() is None:
            QApplication.setOverrideCursor(cross)
            _manual_cursor_override_active = True
        else:
            QApplication.changeOverrideCursor(cross)
            _manual_cursor_override_active = True
    else:
        if _manual_cursor_override_active:
            while QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            _manual_cursor_override_active = False
        if overlay_window is not None:
            overlay_window.setCursor(QCursor(Qt.CursorShape.ArrowCursor))


def set_manual_mode(enabled: bool):
    """Enable/disable manual mode. Must be called from the Qt thread."""
    global MANUAL_MODE, manual_drag_start, manual_drag_kind, manual_preview_box
    with state_lock:
        if enabled and COMBINED_FUSION_HIGHLIGHT_IDX >= 0:
            enabled = False
        if enabled and COMBINED_MODE and not combined_manual_editing_allowed(COMBINED_PHASE, COMBINED_VIEW):
            enabled = False
        MANUAL_MODE = bool(enabled)
        manual_drag_start = None
        manual_drag_kind = None
        manual_preview_box = None
        _manual_clear_selection()
        manual_history_clear()

    if overlay_window is not None:
        overlay_window.set_click_through(not MANUAL_MODE)
    _sync_manual_cursor()


def quit_program():
    """Stop everything and close Qt."""
    global EXIT_PROGRAM, RUNNING, A11Y_SCAN_PENDING, MANUAL_MODE
    global COMBINED_MODE, COMBINED_PHASE, COMBINED_AUTO_FUSED
    global COMBINED_FUSION_HIGHLIGHT_IDX, COMBINED_FUSION_HIGHLIGHT_CLUSTERS

    with state_lock:
        RUNNING = False
        A11Y_SCAN_PENDING = False
        COMBINED_MODE = False
        COMBINED_PHASE = ""
        COMBINED_AUTO_FUSED = False
        COMBINED_FUSION_HIGHLIGHT_IDX = -1
        COMBINED_FUSION_HIGHLIGHT_CLUSTERS = []
        EXIT_PROGRAM = True
        MANUAL_MODE = False

    _sync_manual_cursor()

    if app_bridge is not None:
        app_bridge.quit_signal.emit()

    return False


def _is_enter_key(key) -> bool:
    if key == keyboard.Key.enter:
        return True
    return getattr(key, "name", None) in ("enter", "numpad_enter")


def nudge_selected_box(dx: int, dy: int) -> None:
    """Move selected manual bbox(es) by (dx, dy), rigidly clamped to capture bounds."""
    with state_lock:
        if not MANUAL_MODE:
            return
        if COMBINED_MODE and not combined_manual_editing_allowed(COMBINED_PHASE, COMBINED_VIEW):
            return
        indices = list(manual_selected_indices)
        if not indices and manual_selected_index is not None:
            indices = [manual_selected_index]
        if not indices:
            return
        boxes = active_boxes_list()
        origins = []
        for i in indices:
            if 0 <= i < len(boxes):
                origins.append((i, box_coords(boxes[i])))
        if not origins:
            return
        lo_dx = max(-ox for _, (ox, _, _, _) in origins)
        hi_dx = min(CAPTURE_W - ow - ox for _, (ox, _, ow, _) in origins)
        lo_dy = max(-oy for _, (_, oy, _, _) in origins)
        hi_dy = min(CAPTURE_H - oh - oy for _, (_, oy, _, oh) in origins)
        adx = int(max(lo_dx, min(hi_dx, dx)))
        ady = int(max(lo_dy, min(hi_dy, dy)))
        if adx == 0 and ady == 0:
            return
        manual_history_push_nudge()
        for i, (ox, oy, ow, oh) in origins:
            nx, ny = ox + adx, oy + ady
            current = boxes[i]
            if isinstance(current, dict):
                current["x"] = int(nx)
                current["y"] = int(ny)
            else:
                boxes[i] = (int(nx), int(ny), int(ow), int(oh))

    if overlay_window is not None:
        overlay_window.update()


def _is_ctrl_key(key) -> bool:
    return key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)


def _ctrl_letter(key) -> str | None:
    """Return 'z' / 'y' for Ctrl+letter, including Windows control codes."""
    ch = getattr(key, "char", None)
    if ch == "\x1a":
        return "z"
    if ch == "\x19":
        return "y"
    if isinstance(ch, str) and len(ch) == 1 and ch.isalpha():
        return ch.lower()
    vk = getattr(key, "vk", None)
    if vk in (90, 0x5A):  # Z
        return "z"
    if vk in (89, 0x59):  # Y
        return "y"
    return None


def on_key_press(key):
    """Keyboard handler: Entrée / M / S / Q / flèches / Ctrl+Z / Ctrl+Y."""
    global _ctrl_pressed
    try:
        if _is_ctrl_key(key):
            _ctrl_pressed = True
            return

        if _ctrl_pressed:
            letter = _ctrl_letter(key)
            if letter == "z":
                with state_lock:
                    in_manual = MANUAL_MODE
                if in_manual:
                    manual_undo()
                return
            if letter == "y":
                with state_lock:
                    in_manual = MANUAL_MODE
                if in_manual:
                    manual_redo()
                return

        if _is_enter_key(key):
            on_advance_or_fusion()
            return

        arrow_delta = {
            keyboard.Key.left: (-MANUAL_NUDGE_PX, 0),
            keyboard.Key.right: (MANUAL_NUDGE_PX, 0),
            keyboard.Key.up: (0, -MANUAL_NUDGE_PX),
            keyboard.Key.down: (0, MANUAL_NUDGE_PX),
        }
        if key in arrow_delta:
            with state_lock:
                can_nudge = (
                    MANUAL_MODE
                    and (manual_selected_indices or manual_selected_index is not None)
                    and (
                        not COMBINED_MODE
                        or combined_manual_editing_allowed(COMBINED_PHASE, COMBINED_VIEW)
                    )
                )
            if can_nudge:
                dx, dy = arrow_delta[key]
                nudge_selected_box(dx, dy)
                return
            if key == keyboard.Key.left:
                combined_cycle_view(backward=True)
            elif key == keyboard.Key.right:
                combined_cycle_view(backward=False)
            return

        if hasattr(key, "char") and key.char is not None:
            k = key.char.lower()
            if k == "m":
                request_toggle_manual_mode()
                return
            if k == "s":
                request_save()
                return
            if k == "q":
                quit_program()
                return
    except Exception as e:
        print(f"[WARN] Keyboard error: {e}", flush=True)


def on_key_release(key):
    global _ctrl_pressed
    try:
        if _is_ctrl_key(key):
            _ctrl_pressed = False
    except Exception:
        pass


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main():
    global app_bridge, overlay_window

    app = QApplication([])

    try:
        _ = capture_screen()
    except Exception as e:
        print(f"[WARN] Initial screen capture failed: {e}")

    overlay = OverlayWindow()
    overlay.show()
    overlay_window = overlay

    app_bridge = AppBridge()
    app_bridge.quit_signal.connect(app.quit)
    app_bridge.manual_mode_signal.connect(set_manual_mode)
    app_bridge.session_config_signal.connect(prompt_session_start)
    app_bridge.combined_fusion_signal.connect(on_combined_fusion_request)
    app_bridge.save_signal.connect(on_save_request)
    app_bridge.a11y_filter_signal.connect(on_a11y_filter_request)

    print("=" * 23)
    print("  WidgetFusion Annotator")
    print("=" * 23)
    if sys.platform.startswith("linux"):
        warn_linux_a11y_session()
        if not is_a11y_available():
            print(
                "Linux : a11y indisponible — installez "
                "python3-gi gir1.2-atspi-2.0 at-spi2-core "
                "(session X11 recommandée).\n",
                flush=True,
            )
    elif sys.platform == "darwin":
        print("macOS : support expérimental (pas d’a11y).\n", flush=True)
    print("Entrée · étape/fusion   M · manuel   S · enregistrer   Q · quitter   ←/→ · vues   Ctrl+Z/Y · undo/redo\n", flush=True)

    t = threading.Thread(target=screen_watcher, daemon=True)
    t.start()

    y = threading.Thread(target=yolo_sweeper, daemon=True)
    y.start()

    kb_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    kb_listener.start()

    QTimer.singleShot(0, prompt_session_start)

    app.exec()

    kb_listener.stop()
    t.join(timeout=1)
    y.join(timeout=1)


if __name__ == "__main__":
    main()

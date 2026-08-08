"""Platform-specific UI accessibility bounding box extraction."""

from __future__ import annotations

import os
import sys
from typing import Any, List, Sequence, Tuple, TypedDict

Box = Tuple[int, int, int, int]

# Compositor helpers rarely useful as annotation targets.
_LINUX_SKIP_APP_NAMES = frozenset({
    "kwin_x11",
    "kwin_wayland",
    "xfwm4",
    "mutter",
    "muffin",
})

# Desktop shell / panel processes (dock, status bar). Hit-testing differs from normal apps.
_LINUX_SHELL_APP_NAMES = frozenset({
    "gnome-shell",
    "plasmashell",
    "cinnamon",
    "mate-panel",
    "xfce4-panel",
    "lxqt-panel",
    "lxpanel",
    "budgie-panel",
    "wingpanel",
    "pantheon-panel",
    "cairo-dock",
    "plank",
    "latte-dock",
})

# Processes that often host the desktop icons / wallpaper layer.
_LINUX_DESKTOP_APP_NAMES = frozenset({
    "gjs",
    "nautilus-desktop",
    "nemo-desktop",
    "caja-desktop",
    "pcmanfm",
    "pcmanfm-qt",
    "xfdesktop",
})


class A11yWidget(TypedDict, total=False):
    x: int
    y: int
    w: int
    h: int
    control_type: str
    control_type_id: int
    class_name: str
    name: str
    window: str  # top-level window label (scope root / X11 fallback)


# Windows UIA control types (default “clickable” filter on win32).
CLICKABLE_UIA_TYPES = frozenset({
    "Button",
    "CheckBox",
    "ComboBox",
    "Custom",
    "DataItem",
    "Edit",
    "HeaderItem",
    "Hyperlink",
    "Image",
    "ListItem",
    "MenuItem",
    "RadioButton",
    "ScrollBar",
    "SemanticZoom",
    "Slider",
    "Spinner",
    "SplitButton",
    "TabItem",
    "Text",
    "Thumb",
    "TreeItem",
})

# Linux AT-SPI role names (native — NOT mapped to UIA).
CLICKABLE_ATSPI_ROLES = frozenset({
    "push button",
    "toggle button",
    "check box",
    "radio button",
    "combo box",
    "entry",
    "password text",
    "link",
    "icon",
    "image",
    "label",  # clock, desktop captions, etc.
    "menu",   # panel / status menus (network, sound, battery, …)
    "menu item",
    "check menu item",
    "radio menu item",
    "slider",
    "spin button",
    "page tab",
    "scroll bar",
    "list item",
    "tree item",
    "table cell",
    "x11 window",  # fallback for sandboxed apps missing from AT-SPI
})

# Roles kept for desktop shell chrome: dock / panel (even if normally layout).
_LINUX_SHELL_CHROME_ROLES = frozenset({
    "push button",
    "toggle button",
    "label",
    "menu",
    "link",
    "image",
    "icon",
    "slider",
})


def default_clickable_control_types() -> frozenset[str]:
    """Platform-native default type set for the a11y filter dialog."""
    if sys.platform.startswith("linux"):
        return CLICKABLE_ATSPI_ROLES
    return CLICKABLE_UIA_TYPES


# Alias kept for older call sites (Windows UIA names).
CLICKABLE_CONTROL_TYPES = CLICKABLE_UIA_TYPES


# Common UIA ControlTypeIds → names (UIAutomationClient.h)
UIA_CONTROL_TYPE_NAMES: dict[int, str] = {
    50000: "Button",
    50001: "Calendar",
    50002: "CheckBox",
    50003: "ComboBox",
    50004: "Edit",
    50005: "Hyperlink",
    50006: "Image",
    50007: "ListItem",
    50008: "List",
    50009: "Menu",
    50010: "MenuBar",
    50011: "MenuItem",
    50012: "ProgressBar",
    50013: "RadioButton",
    50014: "ScrollBar",
    50015: "Slider",
    50016: "Spinner",
    50017: "StatusBar",
    50018: "Tab",
    50019: "TabItem",
    50020: "Text",
    50021: "ToolBar",
    50022: "ToolTip",
    50023: "Tree",
    50024: "TreeItem",
    50025: "Custom",
    50026: "Group",
    50027: "Thumb",
    50028: "DataGrid",
    50029: "DataItem",
    50030: "Document",
    50031: "SplitButton",
    50032: "Window",
    50033: "Pane",
    50034: "Header",
    50035: "HeaderItem",
    50036: "Table",
    50037: "TitleBar",
    50038: "Separator",
    50039: "SemanticZoom",
    50040: "AppBar",
}


# ─────────────────────────────────────────────
# Shared geometry helpers
# ─────────────────────────────────────────────


def a11y_box_coords(item: Box | A11yWidget) -> Box:
    if isinstance(item, dict):
        return (int(item["x"]), int(item["y"]), int(item["w"]), int(item["h"]))
    return item


def make_manual_a11y_widget(x: int, y: int, w: int, h: int) -> A11yWidget:
    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "control_type": "Manual",
        "control_type_id": 0,
        "class_name": "",
        "name": "",
        "window": "(manual)",
    }


UNKNOWN_WINDOW_LABEL = "(unknown window)"


def a11y_window_label(item: A11yWidget | dict[str, Any]) -> str:
    """Stable window key used by the live filter dialog."""
    raw = item.get("window")
    if not isinstance(raw, str):
        return UNKNOWN_WINDOW_LABEL
    label = raw.strip()
    return label or UNKNOWN_WINDOW_LABEL


def _bootstrap_system_gi() -> None:
    """Allow venv interpreters to import distro-packaged PyGObject (python3-gi)."""
    if "gi" in sys.modules:
        return
    candidates = (
        "/usr/lib/python3/dist-packages",
        f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages",
        "/usr/local/lib/python3/dist-packages",
    )
    for path in candidates:
        if path in sys.path:
            continue
        if os.path.isdir(os.path.join(path, "gi")):
            sys.path.append(path)


def _import_atspi():
    """Return the Atspi GI module, or None if unavailable."""
    try:
        _bootstrap_system_gi()
        import gi
        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi
        return Atspi
    except Exception:
        return None


_LINUX_SESSION_WARNED = False


def linux_display_server() -> str:
    """Return 'x11', 'wayland', or 'unknown' for the current session."""
    session = (os.environ.get("XDG_SESSION_TYPE") or "").strip().lower()
    if session in {"x11", "wayland"}:
        return session
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "unknown"


def warn_linux_a11y_session() -> None:
    """
    A11y Linux is supported as one pipeline: AT-SPI + X11 stacking.
    Wayland sessions are not a supported target (results would not be reliable).
    """
    global _LINUX_SESSION_WARNED
    if _LINUX_SESSION_WARNED or not sys.platform.startswith("linux"):
        return
    _LINUX_SESSION_WARNED = True
    mode = linux_display_server()
    if mode == "x11":
        print("[INFO] Session X11 — a11y Linux OK (AT-SPI + stacking X11).", flush=True)
        return
    if mode == "wayland":
        print(
            "[WARN] Wayland session detected. "
            "WidgetFusion a11y targets **X11** (same pipeline everywhere). "
            "Under Wayland results are not reliable. "
            "Restart in an 'Ubuntu on Xorg' / X11 session.",
            flush=True,
        )
        return
    print(
        "[WARN] Unknown display session type — a11y expects X11 + AT-SPI.",
        flush=True,
    )


def is_a11y_available() -> bool:
    """True when this platform can run an accessibility scan."""
    if sys.platform == "win32":
        try:
            import comtypes  # noqa: F401
            return True
        except ImportError:
            return False
    if sys.platform.startswith("linux"):
        return _import_atspi() is not None
    return False


def get_accessibility_boxes(
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> List[A11yWidget]:
    """Return accessibility widgets after default filters (all types, no parent inclusion)."""
    raw = get_accessibility_boxes_raw(
        capture_left, capture_top, capture_width, capture_height
    )
    return apply_a11y_filters(raw)


def get_accessibility_boxes_raw(
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> List[A11yWidget]:
    """Return unfiltered accessibility widgets inside the capture region."""
    if sys.platform == "win32":
        return _get_boxes_windows(
            capture_left, capture_top, capture_width, capture_height
        )
    if sys.platform.startswith("linux"):
        warn_linux_a11y_session()
        return _get_boxes_linux(
            capture_left, capture_top, capture_width, capture_height
        )
    if sys.platform == "darwin":
        return _get_boxes_darwin(
            capture_left, capture_top, capture_width, capture_height
        )

    print(f"[WARN] Accessibility boxes are not supported on platform: {sys.platform}")
    return []


def _parse_bounding_rect(rect) -> tuple[int, int, int, int] | None:
    if rect is None:
        return None

    left = int(getattr(rect, "left", 0))
    top = int(getattr(rect, "top", 0))
    right = getattr(rect, "right", None)
    bottom = getattr(rect, "bottom", None)

    if right is not None and bottom is not None:
        width = int(right) - left
        height = int(bottom) - top
    else:
        width = int(getattr(rect, "width", 0))
        height = int(getattr(rect, "height", 0))

    if width <= 0 or height <= 0:
        return None
    return left, top, width, height


def _screen_rect_intersects_capture(
    left: int,
    top: int,
    width: int,
    height: int,
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> bool:
    right = left + width
    bottom = top + height
    cap_right = capture_left + capture_width
    cap_bottom = capture_top + capture_height
    return not (
        right <= capture_left
        or left >= cap_right
        or bottom <= capture_top
        or top >= cap_bottom
    )


def _runtime_id_key(element) -> tuple:
    try:
        runtime_id = element.GetRuntimeId()
        if runtime_id is not None:
            return tuple(runtime_id)
    except Exception:
        pass
    return (id(element),)


# ─────────────────────────────────────────────
# Widget bbox filtering
# ─────────────────────────────────────────────


def _box_contains(outer: Box | A11yWidget, inner: Box | A11yWidget, margin: int = 0) -> bool:
    ox, oy, ow, oh = a11y_box_coords(outer)
    ix, iy, iw, ih = a11y_box_coords(inner)
    return (
        ox <= ix + margin
        and oy <= iy + margin
        and ox + ow >= ix + iw - margin
        and oy + oh >= iy + ih - margin
        and outer != inner
        and a11y_box_coords(outer) != a11y_box_coords(inner)
    )


def _suppress_contained_children(boxes: List[A11yWidget]) -> List[A11yWidget]:
    """Drop widgets fully contained inside another kept widget."""
    drop: set[int] = set()
    for i, parent in enumerate(boxes):
        if i in drop:
            continue
        for j, child in enumerate(boxes):
            if i == j or j in drop:
                continue
            if not _box_contains(parent, child):
                continue
            parent_name = str(parent.get("name") or "").strip()
            child_name = str(child.get("name") or "").strip()
            parent_role = str(parent.get("control_type") or "")
            # Empty shell menus that wrap a named clock/label → keep the label.
            if (
                parent_role in {"menu", "panel", "Pane"}
                and not parent_name
                and child_name
            ):
                drop.add(i)
                break
            drop.add(j)
    return [box for i, box in enumerate(boxes) if i not in drop]


def _a11y_near_duplicate(a: A11yWidget, b: A11yWidget) -> bool:
    """
    True when two boxes are the same control reported twice with a small offset.

    Same type only for near matches: otherwise unchecking TreeItem can leave a
    sibling Tree/DataItem that was silently merged (box looks “re-aligned”).
    Exact identical rects may still collapse across types.
    """
    ab = a11y_box_coords(a)
    bb = a11y_box_coords(b)
    if ab == bb:
        return True
    ra = str(a.get("control_type") or "")
    rb = str(b.get("control_type") or "")
    if ra != rb:
        return False
    iou = _box_iou_xywh(ab, bb)
    if iou >= 0.88:
        return True
    na = str(a.get("name") or "").strip()
    nb = str(b.get("name") or "").strip()
    if not na or na != nb:
        return False
    ax, ay, aw, ah = ab
    bx, by, bw, bh = bb
    if abs(aw - bw) > 10 or abs(ah - bh) > 10:
        return False
    if abs(ax - bx) > 16 or abs(ay - by) > 16:
        return False
    return iou >= 0.55


def _merge_near_duplicate_a11y_boxes(boxes: List[A11yWidget]) -> List[A11yWidget]:
    """Collapse slightly-offset duplicate accessibility hits (keep best)."""
    if len(boxes) < 2:
        return list(boxes)
    drop: set[int] = set()
    for i, a in enumerate(boxes):
        if i in drop:
            continue
        ax, ay, _, _ = a11y_box_coords(a)
        for j in range(i + 1, len(boxes)):
            if j in drop:
                continue
            b = boxes[j]
            if not _a11y_near_duplicate(a, b):
                continue
            qa = _a11y_widget_quality(a)
            qb = _a11y_widget_quality(b)
            if qb > qa:
                drop.add(i)
                break
            if qa > qb:
                drop.add(j)
                continue
            bx, by, _, _ = a11y_box_coords(b)
            # Same quality: keep the top-left hit (usually the real cell).
            if (bx, by) < (ax, ay):
                drop.add(i)
                break
            drop.add(j)
    return [box for i, box in enumerate(boxes) if i not in drop]


def apply_a11y_filters(
    boxes: Sequence[A11yWidget] | List[A11yWidget],
    enabled_types: set[str] | frozenset[str] | None = None,
    enabled_windows: set[str] | frozenset[str] | None = None,
    parent_inclusion: bool = False,
) -> List[A11yWidget]:
    """Filter raw a11y widgets by window, control type, and optional parent inclusion."""
    if enabled_types is None:
        filtered = list(boxes)
    else:
        types = set(enabled_types)
        filtered = [b for b in boxes if str(b.get("control_type") or "") in types]
    if enabled_windows is not None:
        filtered = [b for b in filtered if a11y_window_label(b) in enabled_windows]
    if parent_inclusion:
        filtered = _suppress_contained_children(filtered)
    filtered = _merge_near_duplicate_a11y_boxes(filtered)
    return filtered


def _screen_rect_to_capture_box(
    left: int,
    top: int,
    width: int,
    height: int,
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> Box | None:
    if width <= 0 or height <= 0:
        return None

    x = left - capture_left
    y = top - capture_top
    x2 = min(capture_width, x + width)
    y2 = min(capture_height, y + height)
    x = max(0, x)
    y = max(0, y)
    width = x2 - x
    height = y2 - y

    if width <= 0 or height <= 0:
        return None

    return (int(x), int(y), int(width), int(height))


def _window_label(element) -> str:
    parts = []
    for getter in ("CurrentName", "CurrentClassName", "CurrentAutomationId"):
        try:
            value = getattr(element, getter)
            if callable(value):
                value = value()
            if value:
                parts.append(str(value))
        except Exception:
            pass
    return " / ".join(parts) if parts else "(unknown window)"


def _control_type_name(control_type_id: int) -> str:
    return UIA_CONTROL_TYPE_NAMES.get(control_type_id, f"ControlType({control_type_id})")


def _element_metadata(element) -> dict[str, Any]:
    control_type_id = 0
    try:
        control_type_id = int(element.CurrentControlType)
    except Exception:
        pass

    class_name = ""
    name = ""
    try:
        class_name = str(element.CurrentClassName or "")
    except Exception:
        pass
    try:
        name = str(element.CurrentName or "")
    except Exception:
        pass

    control_type = _control_type_name(control_type_id)
    return {
        "control_type": control_type,
        "control_type_id": control_type_id,
        "class_name": class_name,
        "name": name,
    }


def _element_hwnd(element) -> int:
    try:
        hwnd = int(element.CurrentNativeWindowHandle)
        return hwnd if hwnd else 0
    except Exception:
        return 0


def _hwnd_is_visible_on_screen(hwnd: int) -> bool:
    """Reject minimized / invisible / cloaked (virtual desktop) windows."""
    if not hwnd:
        return False

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    if not user32.IsWindow(hwnd):
        return False
    if not user32.IsWindowVisible(hwnd):
        return False
    if user32.IsIconic(hwnd):
        return False

    # DWMWA_CLOAKED = 14 — window on another virtual desktop / UWP cloaked.
    try:
        dwmapi = ctypes.windll.dwmapi
        cloaked = wintypes.DWORD(0)
        if dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            14,
            ctypes.byref(cloaked),
            ctypes.sizeof(cloaked),
        ) == 0 and cloaked.value:
            return False
    except Exception:
        pass

    return True


def _hwnd_top_level(hwnd: int) -> int:
    import ctypes
    GA_ROOT = 2
    try:
        root = ctypes.windll.user32.GetAncestor(hwnd, GA_ROOT)
        return int(root) if root else hwnd
    except Exception:
        return hwnd


def _hwnd_is_visually_exposed(
    hwnd: int,
    left: int,
    top: int,
    width: int,
    height: int,
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> bool:
    """
    True if at least one sample point inside (window ∩ capture) hits this window
    (or one of its descendants). Filters apps that only "intersect" geometrically
    but are fully covered by another window.
    """
    import ctypes

    user32 = ctypes.windll.user32

    x0 = max(left, capture_left)
    y0 = max(top, capture_top)
    x1 = min(left + width, capture_left + capture_width)
    y1 = min(top + height, capture_top + capture_height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False

    samples = [
        (x0 + (x1 - x0) // 2, y0 + (y1 - y0) // 2),
        (x0 + (x1 - x0) // 4, y0 + (y1 - y0) // 4),
        (x0 + 3 * (x1 - x0) // 4, y0 + (y1 - y0) // 4),
        (x0 + (x1 - x0) // 4, y0 + 3 * (y1 - y0) // 4),
        (x0 + 3 * (x1 - x0) // 4, y0 + 3 * (y1 - y0) // 4),
        (x0 + (x1 - x0) // 2, y0 + max(1, (y1 - y0) // 8)),
        (x0 + (x1 - x0) // 2, y1 - max(1, (y1 - y0) // 8)),
    ]

    root = _hwnd_top_level(hwnd)
    for sx, sy in samples:
        hit = user32.WindowFromPoint(ctypes.wintypes.POINT(int(sx), int(sy)))
        if not hit:
            continue
        if _hwnd_top_level(int(hit)) == root:
            return True
    return False


# ─────────────────────────────────────────────
# Window scope
# ─────────────────────────────────────────────


def _top_level_window_is_visible_in_capture(
    element,
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> bool:
    """Window filter: intersects capture, not minimized/cloaked, visually exposed."""
    parsed = _parse_bounding_rect(element.CurrentBoundingRectangle)
    if parsed is None:
        return False
    if not _screen_rect_intersects_capture(
        *parsed,
        capture_left,
        capture_top,
        capture_width,
        capture_height,
    ):
        return False

    try:
        if element.CurrentIsOffscreen:
            return False
    except Exception:
        pass

    hwnd = _element_hwnd(element)
    if not hwnd:
        return False
    if not _hwnd_is_visible_on_screen(hwnd):
        return False
    return _hwnd_is_visually_exposed(
        hwnd,
        *parsed,
        capture_left,
        capture_top,
        capture_width,
        capture_height,
    )


def find_visible_window_roots(
    automation,
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> list[tuple[object, str]]:
    """Top-level desktop windows that pass the window visibility filter."""
    roots: list[tuple[object, str]] = []
    seen: set[tuple] = set()

    try:
        desktop = automation.GetRootElement()
        walker = automation.ControlViewWalker
        child = walker.GetFirstChildElement(desktop)
        while child is not None:
            try:
                if not _top_level_window_is_visible_in_capture(
                    child,
                    capture_left,
                    capture_top,
                    capture_width,
                    capture_height,
                ):
                    raise RuntimeError("skip")

                key = _runtime_id_key(child)
                if key not in seen:
                    seen.add(key)
                    roots.append((child, _window_label(child)))
            except RuntimeError:
                pass
            except Exception:
                pass
            try:
                child = walker.GetNextSiblingElement(child)
            except Exception:
                break
    except Exception as exc:
        print(f"[WARN] Failed to list top-level accessibility roots: {exc}", flush=True)

    return roots


# ─────────────────────────────────────────────
# Widget collection inside visible windows
# ─────────────────────────────────────────────


def _collect_uia_boxes(
    automation,
    root_element,
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
    seen: set[Box] | None = None,
    window_label: str = "",
) -> List[A11yWidget]:
    boxes: List[A11yWidget] = []
    if seen is None:
        seen = set()
    win = (window_label or "").strip() or UNKNOWN_WINDOW_LABEL

    def walk(element, depth: int = 0) -> None:
        if element is None or depth > 64:
            return

        skip_box = False
        try:
            if element.CurrentIsOffscreen:
                skip_box = True
        except Exception:
            pass

        if not skip_box:
            try:
                metadata = _element_metadata(element)
                parsed = _parse_bounding_rect(element.CurrentBoundingRectangle)
                if parsed is not None:
                    box = _screen_rect_to_capture_box(
                        *parsed,
                        capture_left=capture_left,
                        capture_top=capture_top,
                        capture_width=capture_width,
                        capture_height=capture_height,
                    )
                    if box is not None and box not in seen:
                        seen.add(box)
                        x, y, w, h = box
                        boxes.append({
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                            **metadata,
                            "window": win,
                        })
            except Exception:
                pass

        try:
            walker = automation.ControlViewWalker
            child = walker.GetFirstChildElement(element)
            while child is not None:
                walk(child, depth + 1)
                child = walker.GetNextSiblingElement(child)
        except Exception:
            pass

    walk(root_element)
    return boxes


def _get_boxes_windows(
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> List[A11yWidget]:
    """Enumerate UI Automation widgets for visible top-level windows (Windows UIA)."""
    try:
        import comtypes.client
    except ImportError:
        print("[WARN] comtypes is required on Windows for accessibility boxes.")
        print("[WARN] Install with: pip install comtypes")
        return []

    try:
        comtypes.client.GetModule("UIAutomationCore.dll")
        from comtypes.gen import UIAutomationClient as UIA
    except Exception as exc:
        print(f"[WARN] Failed to load UI Automation: {exc}")
        return []

    try:
        automation = comtypes.client.CreateObject(
            UIA.CUIAutomation,
            interface=UIA.IUIAutomation,
        )
    except Exception as exc:
        print(f"[WARN] UI Automation init failed: {exc}")
        return []

    scope_roots = find_visible_window_roots(
        automation,
        capture_left,
        capture_top,
        capture_width,
        capture_height,
    )
    if not scope_roots:
        print("[WARN] Could not determine target window for accessibility scan.")
        return []

    print(
        f"[INFO] Visible windows (scope): {len(scope_roots)} top-level root(s)",
        flush=True,
    )
    for _, label in scope_roots:
        print(f"  - {label}", flush=True)

    seen: set[Box] = set()
    boxes: List[A11yWidget] = []
    for root_element, label in scope_roots:
        boxes.extend(
            _collect_uia_boxes(
                automation,
                root_element,
                capture_left,
                capture_top,
                capture_width,
                capture_height,
                seen=seen,
                window_label=label,
            )
        )

    print(f"[INFO] Widgets collected (raw): {len(boxes)} bbox", flush=True)
    return boxes


# Layout / chrome containers — walked but not emitted (cuts false positives).
_LINUX_LAYOUT_ROLES = frozenset({
    "panel",
    "filler",
    "section",
    "layered pane",
    "scroll pane",
    "viewport",
    "root pane",
    "glass pane",
    "split pane",
    "html container",
    "application",
    "frame",
    "window",
    "dialog",
    "desktop frame",
    "paragraph",
    "article",
    "document frame",
    "document web",
    "page",
    "form",
    "tool bar",
    "menu",
    "menu bar",
    "page tab list",
    "separator",
    "status bar",
})

# Non-interactive text noise (very common in Electron/IDEs).
_LINUX_NOISE_ROLES = frozenset({
    "static",
    "text",
    "heading",
    "paragraph",
    "tool tip",
    "canvas",
    "unknown",
    "accelerator label",
    "landmark",
})

_LINUX_WINDOW_ROLES = frozenset({
    "frame",
    "window",
    "dialog",
    "alert",
    "file chooser",
})

# Giant overlays that must not be kept as widgets.
_LINUX_SKIP_CONTAINER_ROLES = frozenset({
    "panel",
    "filler",
    "layered pane",
    "scroll pane",
    "viewport",
    "root pane",
    "glass pane",
    "split pane",
    "frame",
    "window",
    "dialog",
    "section",
})
_LINUX_MAX_CONTAINER_CAPTURE_FRAC = 0.35

# Roles kept when scanning the desktop icons layer.
_LINUX_DESKTOP_ROLES = frozenset({
    "icon",
    "label",
    "push button",
    "toggle button",
    "image",
    "link",
})

_LINUX_MIN_BOX_SIDE = 6


def _normalize_atspi_role(role_name: str) -> str:
    return (role_name or "").strip().lower()


def _linux_accessible_showing(Atspi, accessible) -> bool:
    try:
        states = accessible.get_state_set()
        if states is None:
            return True
        if states.contains(Atspi.StateType.DEFUNCT):
            return False
        if states.contains(Atspi.StateType.ICONIFIED):
            return False
        if states.contains(Atspi.StateType.SHOWING):
            return True
        return bool(states.contains(Atspi.StateType.VISIBLE))
    except Exception:
        return True


def _linux_content_origin(
    x11_geom: tuple[int, int, int, int],
    atspi_size: tuple[int, int],
    frame_extents: tuple[int, int, int, int] | None = None,
) -> tuple[int, int]:
    """
    Map AT-SPI WINDOW coords into screen space.

    Prefer `_GTK_FRAME_EXTENTS` / `_NET_FRAME_EXTENTS` (left, right, top, bottom)
    so CSD shadows are peeled off the X11 frame. Fallback: center AT-SPI size
    inside the X11 geometry.
    """
    xx, yy, xw, xh = x11_geom
    if frame_extents is not None:
        left, right, top, bottom = frame_extents
        if (
            left >= 0
            and right >= 0
            and top >= 0
            and bottom >= 0
            and left + right < xw
            and top + bottom < xh
        ):
            return xx + left, yy + top
    aw, ah = atspi_size
    if aw <= 0 or ah <= 0:
        return xx, yy
    pad_x = max(0, xw - aw) // 2
    pad_y = max(0, xh - ah) // 2
    return xx + pad_x, yy + pad_y


def _linux_extents_screen(
    Atspi,
    accessible,
    window_origin: tuple[int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    """
    Screen-space bbox.

    When a window content origin is calibrated (X11 frame − CSD pad), always
    prefer WINDOW + origin. Mixing SCREEN for some nodes and WINDOW for others
    produces near-duplicate boxes shifted by a few pixels (GTK/Nautilus).
    """
    try:
        component = accessible.get_component_iface()
        if component is None:
            return None

        if window_origin is not None:
            try:
                win = component.get_extents(Atspi.CoordType.WINDOW)
                wx, wy = int(win.x), int(win.y)
                ww, wh = int(win.width), int(win.height)
                if ww > 0 and wh > 0:
                    return (
                        int(window_origin[0] + wx),
                        int(window_origin[1] + wy),
                        ww,
                        wh,
                    )
            except Exception:
                pass

        screen = component.get_extents(Atspi.CoordType.SCREEN)
        left, top = int(screen.x), int(screen.y)
        width, height = int(screen.width), int(screen.height)
        if width <= 0 or height <= 0:
            return None
        if abs(left) > 100_000 or abs(top) > 100_000:
            return None
        return left, top, width, height
    except Exception:
        return None


def _linux_role_name(Atspi, accessible) -> tuple[str, int]:
    """Return (role_name, role_id) — cheap; avoids get_attributes."""
    control_type_id = 0
    role = None
    try:
        role = accessible.get_role()
        control_type_id = int(role)
    except Exception:
        pass
    role_name = ""
    try:
        role_name = str(accessible.get_role_name() or "")
    except Exception:
        if role is not None:
            try:
                role_name = str(Atspi.role_get_name(role) or "")
            except Exception:
                role_name = ""
    return role_name, control_type_id


def _linux_metadata(Atspi, accessible, role_name: str = "", control_type_id: int = 0) -> dict[str, Any]:
    if not role_name and not control_type_id:
        role_name, control_type_id = _linux_role_name(Atspi, accessible)

    name = ""
    try:
        name = str(accessible.get_name() or "")
    except Exception:
        pass

    # Skip get_attributes() — expensive D-Bus round-trip per node; role is enough.
    class_name = role_name

    return {
        "control_type": _normalize_atspi_role(role_name) or "unknown",
        "control_type_id": control_type_id,
        "class_name": class_name,
        "name": name,
    }


def _linux_window_label(app_name: str, accessible) -> str:
    parts = [app_name] if app_name else []
    try:
        win_name = accessible.get_name()
        if win_name:
            parts.append(str(win_name))
    except Exception:
        pass
    try:
        role = accessible.get_role_name()
        if role:
            parts.append(str(role))
    except Exception:
        pass
    return " / ".join(parts) if parts else "(unknown window)"


def _linux_window_attrs(accessible) -> dict[str, str]:
    try:
        attrs = accessible.get_attributes() or {}
        if hasattr(attrs, "keys"):
            return {str(k): str(attrs.get(k)) for k in attrs.keys()}
    except Exception:
        pass
    return {}


# ─────────────────────────────────────────────
# X11 visibility (mapped + stacking) — Linux
# ─────────────────────────────────────────────


class _X11Stack:
    """Cached _NET_CLIENT_LIST_STACKING + geometries (bottom → top)."""

    def __init__(self) -> None:
        self.ok = False
        self.screen_w = 0
        self.screen_h = 0
        self.windows: list[tuple[int, int, int, int, int]] = []  # xid,x,y,w,h
        self.titles: dict[int, str] = {}
        self.pids: dict[int, int] = {}
        # xid → (left, right, top, bottom) CSD / frame shadow extents
        self.frame_extents: dict[int, tuple[int, int, int, int]] = {}

        try:
            import ctypes
            from ctypes import POINTER, Structure, byref, c_char_p, c_int, c_long, c_ulong
        except Exception:
            return

        class XWindowAttributes(Structure):
            _fields_ = [
                ("x", c_int), ("y", c_int), ("width", c_int), ("height", c_int),
                ("border_width", c_int), ("depth", c_int), ("visual", c_ulong),
                ("root", c_ulong), ("class", c_int), ("bit_gravity", c_int),
                ("win_gravity", c_int), ("backing_store", c_int),
                ("backing_planes", c_ulong), ("backing_pixel", c_ulong),
                ("save_under", c_int), ("colormap", c_ulong), ("map_installed", c_int),
                ("map_state", c_int), ("all_event_masks", c_ulong),
                ("your_event_mask", c_ulong), ("do_not_propagate_mask", c_ulong),
                ("override_redirect", c_int), ("screen", c_ulong),
            ]

        try:
            xlib = ctypes.cdll.LoadLibrary("libX11.so.6")
        except OSError:
            return

        xlib.XOpenDisplay.restype = c_ulong
        xlib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        xlib.XDefaultRootWindow.restype = c_ulong
        xlib.XDefaultRootWindow.argtypes = [c_ulong]
        xlib.XInternAtom.restype = c_ulong
        xlib.XInternAtom.argtypes = [c_ulong, ctypes.c_char_p, c_int]
        xlib.XGetWindowAttributes.argtypes = [c_ulong, c_ulong, POINTER(XWindowAttributes)]
        xlib.XTranslateCoordinates.argtypes = [
            c_ulong, c_ulong, c_ulong, c_int, c_int,
            POINTER(c_int), POINTER(c_int), POINTER(c_ulong),
        ]
        xlib.XGetWindowProperty.argtypes = [
            c_ulong, c_ulong, c_ulong, c_long, c_long, c_int, c_ulong,
            POINTER(c_ulong), POINTER(c_int), POINTER(c_ulong), POINTER(c_ulong),
            POINTER(POINTER(c_ulong)),
        ]
        xlib.XFetchName.argtypes = [c_ulong, c_ulong, POINTER(c_char_p)]
        xlib.XFree.argtypes = [ctypes.c_void_p]
        xlib.XCloseDisplay.argtypes = [c_ulong]

        display = xlib.XOpenDisplay(None)
        if not display:
            return
        try:
            root = xlib.XDefaultRootWindow(display)
            root_attrs = XWindowAttributes()
            if xlib.XGetWindowAttributes(display, root, byref(root_attrs)):
                self.screen_w = int(root_attrs.width)
                self.screen_h = int(root_attrs.height)

            atom = xlib.XInternAtom(display, b"_NET_CLIENT_LIST_STACKING", 0)
            net_name = xlib.XInternAtom(display, b"_NET_WM_NAME", 0)
            utf8 = xlib.XInternAtom(display, b"UTF8_STRING", 0)
            net_pid = xlib.XInternAtom(display, b"_NET_WM_PID", 0)
            net_state = xlib.XInternAtom(display, b"_NET_WM_STATE", 0)
            state_hidden = xlib.XInternAtom(display, b"_NET_WM_STATE_HIDDEN", 0)
            gtk_frame = xlib.XInternAtom(display, b"_GTK_FRAME_EXTENTS", 0)
            net_frame = xlib.XInternAtom(display, b"_NET_FRAME_EXTENTS", 0)
            actual_type = c_ulong()
            actual_format = c_int()
            nitems = c_ulong()
            bytes_after = c_ulong()
            prop = POINTER(c_ulong)()
            status = xlib.XGetWindowProperty(
                display, root, atom, 0, 4096, 0, 0,
                byref(actual_type), byref(actual_format), byref(nitems),
                byref(bytes_after), byref(prop),
            )
            if status != 0 or not nitems.value:
                return

            IsViewable = 2
            for i in range(int(nitems.value)):
                xid = int(prop[i])
                attrs = XWindowAttributes()
                if not xlib.XGetWindowAttributes(display, xid, byref(attrs)):
                    continue
                if int(attrs.map_state) != IsViewable:
                    continue
                # Minimized / "show desktop" windows often stay mapped but HIDDEN.
                try:
                    s_type = c_ulong()
                    s_format = c_int()
                    s_nitems = c_ulong()
                    s_bytes = c_ulong()
                    s_prop = POINTER(c_ulong)()
                    st = xlib.XGetWindowProperty(
                        display, xid, net_state, 0, 64, 0, 0,
                        byref(s_type), byref(s_format), byref(s_nitems),
                        byref(s_bytes), byref(s_prop),
                    )
                    hidden = False
                    if st == 0 and s_nitems.value and s_prop:
                        for si in range(int(s_nitems.value)):
                            if int(s_prop[si]) == int(state_hidden):
                                hidden = True
                                break
                        xlib.XFree(s_prop)
                    if hidden:
                        continue
                except Exception:
                    pass
                rx = c_int()
                ry = c_int()
                child = c_ulong()
                if not xlib.XTranslateCoordinates(
                    display, xid, root, 0, 0, byref(rx), byref(ry), byref(child)
                ):
                    continue
                w, h = int(attrs.width), int(attrs.height)
                if w <= 1 or h <= 1:
                    continue
                self.windows.append((xid, int(rx.value), int(ry.value), w, h))

                # Title
                title = ""
                try:
                    t_type = c_ulong()
                    t_format = c_int()
                    t_nitems = c_ulong()
                    t_bytes = c_ulong()
                    t_prop = POINTER(c_ulong)()
                    st = xlib.XGetWindowProperty(
                        display, xid, net_name, 0, 1024, 0, utf8,
                        byref(t_type), byref(t_format), byref(t_nitems),
                        byref(t_bytes), byref(t_prop),
                    )
                    if st == 0 and t_nitems.value and t_prop:
                        title = ctypes.string_at(t_prop, int(t_nitems.value)).decode(
                            "utf-8", errors="replace"
                        ).strip()
                        xlib.XFree(t_prop)
                except Exception:
                    title = ""
                if not title:
                    try:
                        name_p = c_char_p()
                        if xlib.XFetchName(display, xid, byref(name_p)) and name_p.value:
                            title = name_p.value.decode("utf-8", errors="replace").strip()
                            xlib.XFree(name_p)
                    except Exception:
                        pass
                if title:
                    self.titles[xid] = title

                # Process id (for X11 / AT-SPI window matching).
                try:
                    p_type = c_ulong()
                    p_format = c_int()
                    p_nitems = c_ulong()
                    p_bytes = c_ulong()
                    p_prop = POINTER(c_ulong)()
                    st = xlib.XGetWindowProperty(
                        display, xid, net_pid, 0, 1, 0, 0,
                        byref(p_type), byref(p_format), byref(p_nitems),
                        byref(p_bytes), byref(p_prop),
                    )
                    if st == 0 and p_nitems.value and p_prop:
                        self.pids[xid] = int(p_prop[0])
                        xlib.XFree(p_prop)
                except Exception:
                    pass

                # CSD shadow / frame extents (left, right, top, bottom).
                for fe_atom in (gtk_frame, net_frame):
                    try:
                        f_type = c_ulong()
                        f_format = c_int()
                        f_nitems = c_ulong()
                        f_bytes = c_ulong()
                        f_prop = POINTER(c_ulong)()
                        st = xlib.XGetWindowProperty(
                            display, xid, fe_atom, 0, 4, 0, 0,
                            byref(f_type), byref(f_format), byref(f_nitems),
                            byref(f_bytes), byref(f_prop),
                        )
                        if st == 0 and int(f_nitems.value) >= 4 and f_prop:
                            extents = (
                                int(f_prop[0]),
                                int(f_prop[1]),
                                int(f_prop[2]),
                                int(f_prop[3]),
                            )
                            xlib.XFree(f_prop)
                            if any(v > 0 for v in extents):
                                self.frame_extents[xid] = extents
                                break
                        elif f_prop:
                            xlib.XFree(f_prop)
                    except Exception:
                        pass

            if prop:
                xlib.XFree(prop)
            self.ok = bool(self.windows)
        finally:
            xlib.XCloseDisplay(display)

    def top_at(self, x: int, y: int) -> tuple[int, int, int, int] | None:
        """Return (x,y,w,h) of topmost mapped client containing the point."""
        hit = self.top_xid_at(x, y)
        if hit is None:
            return None
        _xid, wx, wy, ww, wh = hit
        return wx, wy, ww, wh

    def top_xid_at(self, x: int, y: int) -> tuple[int, int, int, int, int] | None:
        """Return (xid,x,y,w,h) of topmost mapped client containing the point."""
        for xid, wx, wy, ww, wh in reversed(self.windows):
            if wx <= x < wx + ww and wy <= y < wy + wh:
                return xid, wx, wy, ww, wh
        return None


def _match_x11_xid(
    stack: _X11Stack,
    extents: tuple[int, int, int, int],
    atspi_name: str | None = None,
) -> int | None:
    """Best X11 client id for an AT-SPI window (title first, then geometry IoU)."""
    if not stack.ok:
        return None

    # 1) Title / WM name match — robust when AT-SPI vs X11 extents disagree (GTK/CSD).
    # Prefer the topmost matching client (stack is bottom → top).
    needle = (atspi_name or "").strip().lower()
    if needle:
        for xid, x, y, w, h in reversed(stack.windows):
            if _linux_is_desktop_stage((x, y, w, h), stack.screen_w, stack.screen_h):
                continue
            title = (stack.titles.get(xid) or "").strip().lower()
            if not title:
                continue
            if needle == title or needle in title or title in needle:
                return xid
        # Named AT-SPI frame with no mapped X11 title → minimized / withdrawn.
        # Do NOT fall back to geometry: maximized apps share nearly identical
        # frames, so IoU would steal Cursor/Chrome and keep scanning Nautilus.
        return None

    # 2) Geometry IoU only for nameless frames (looser — CSD vs client area).
    best_xid = None
    best_iou = 0.0
    second_iou = 0.0
    for xid, x, y, w, h in reversed(stack.windows):
        if _linux_is_desktop_stage((x, y, w, h), stack.screen_w, stack.screen_h):
            continue
        iou = _box_iou_xywh(extents, (x, y, w, h))
        if iou > best_iou:
            second_iou = best_iou
            best_iou = iou
            best_xid = xid
        elif iou > second_iou:
            second_iou = iou
    # Ambiguous: several nearly-maximized windows look the same.
    if best_iou >= 0.30 and best_iou - second_iou < 0.05 and second_iou >= 0.30:
        return None
    return best_xid if best_iou >= 0.30 else None


def _box_iou_xywh(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _linux_sample_points(
    left: int,
    top: int,
    width: int,
    height: int,
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> list[tuple[int, int]]:
    x0 = max(left, capture_left)
    y0 = max(top, capture_top)
    x1 = min(left + width, capture_left + capture_width)
    y1 = min(top + height, capture_top + capture_height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return []
    mx = x0 + (x1 - x0) // 2
    my = y0 + (y1 - y0) // 2
    # Include edge midpoints so a mostly-covered window (Cursor under a smaller
    # app) still counts as exposed if any strip remains visible.
    return [
        (mx, my),
        (x0 + (x1 - x0) // 4, y0 + (y1 - y0) // 4),
        (x0 + 3 * (x1 - x0) // 4, y0 + (y1 - y0) // 4),
        (x0 + (x1 - x0) // 4, y0 + 3 * (y1 - y0) // 4),
        (x0 + 3 * (x1 - x0) // 4, y0 + 3 * (y1 - y0) // 4),
        (x0 + 4, my),
        (x1 - 5, my),
        (mx, y0 + 4),
        (mx, y1 - 5),
    ]


def _linux_app_window_exposed(
    stack: _X11Stack,
    extents: tuple[int, int, int, int],
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
    xid: int | None = None,
) -> bool:
    """True if at least one sample point hits THIS window's X11 id (not a look-alike)."""
    if not stack.ok:
        return True  # can't verify — keep
    # Caller must resolve xid (title match). Never re-match by geometry alone:
    # maximized apps share frames, so a hidden Nautilus would steal Cursor.
    if xid is None:
        return False
    # Sample on X11 frame geometry — AT-SPI extents often disagree (GTK/CSD).
    sample_ext = extents
    for id_, x, y, w, h in stack.windows:
        if id_ == xid:
            sample_ext = (x, y, w, h)
            break
    samples = _linux_sample_points(
        *sample_ext, capture_left, capture_top, capture_width, capture_height
    )
    if not samples:
        return False
    for sx, sy in samples:
        hit = stack.top_xid_at(sx, sy)
        if hit is not None and hit[0] == xid:
            return True
    return False


def _linux_is_desktop_stage(
    geom: tuple[int, int, int, int],
    screen_w: int,
    screen_h: int,
) -> bool:
    """True for the fullscreen desktop/shell stage — not a nearly-maximized app."""
    x, y, w, h = geom
    if screen_w <= 0 or screen_h <= 0:
        return False
    return (
        x <= 2
        and y <= 2
        and w >= int(screen_w * 0.98)
        and h >= int(screen_h * 0.98)
    )


def _linux_is_shell_app(app_name: str) -> bool:
    """True for desktop-shell / panel processes across common Linux DEs."""
    key = (app_name or "").strip().lower()
    if not key:
        return False
    if key in _LINUX_SHELL_APP_NAMES:
        return True
    if key.endswith("-panel") or key.endswith("_panel"):
        return True
    if key.endswith("-dock") or key.endswith("_dock"):
        return True
    # e.g. "gnome-shell" — not terminals
    if key.endswith("-shell") and "terminal" not in key:
        return True
    return False


def _linux_shell_persistent_chrome(
    extents: tuple[int, int, int, int],
    screen_w: int = 0,
    screen_h: int = 0,
) -> bool:
    """
    Keep only typical always-visible shell chrome (dock / panels).
    Closed panel menus often stay in AT-SPI with real coords — drop those.
    """
    left, top, w, h = extents
    # Left / right dock strip
    if left + w <= 100:
        return True
    if screen_w > 0 and left >= screen_w - 100 and w <= 100:
        return True
    # Top panel / status bar
    if top <= 8 and h <= 48 and top + h <= 56:
        return True
    # Bottom panel
    if screen_h > 0 and top + h >= screen_h - 8 and h <= 72 and top >= screen_h - 80:
        return True
    return False


def _enrich_desktop_icon_names(boxes: List[A11yWidget]) -> None:
    """Copy nearby label text onto nameless desktop icons (in-place)."""
    icons = [b for b in boxes if b.get("control_type") == "icon"]
    labels = [
        b for b in boxes
        if b.get("control_type") == "label" and str(b.get("name") or "").strip()
    ]
    for icon in icons:
        if str(icon.get("name") or "").strip():
            continue
        ix, iy, iw, ih = int(icon["x"]), int(icon["y"]), int(icon["w"]), int(icon["h"])
        icx = ix + iw / 2
        best = None
        best_dist = 1e9
        for lab in labels:
            lx, ly, lw, lh = int(lab["x"]), int(lab["y"]), int(lab["w"]), int(lab["h"])
            if ly < iy or ly > iy + ih + 100:
                continue
            lcx = lx + lw / 2
            if abs(lcx - icx) > max(iw, lw) + 20:
                continue
            dist = abs(ly - (iy + ih)) + abs(lcx - icx)
            if dist < best_dist:
                best_dist = dist
                best = lab
        if best is not None:
            icon["name"] = str(best["name"])


def _collect_x11_orphan_windows(
    stack: _X11Stack,
    atspi_geoms: list[tuple[int, int, int, int]],
    atspi_titles: set[str],
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
    boxes: List[A11yWidget],
    seen: dict[Box, int],
) -> int:
    """
    Add one synthetic widget per visible X11 window that has no AT-SPI tree
    (typical of Snap Firefox). Inner controls are still unavailable.
    """
    if not stack.ok:
        return 0
    titles_l = {t.strip().lower() for t in atspi_titles if t and t.strip()}
    added = 0
    for xid, x, y, w, h in stack.windows:
        if _linux_is_desktop_stage((x, y, w, h), stack.screen_w, stack.screen_h):
            continue
        geom = (x, y, w, h)
        title = str(stack.titles.get(xid, "") or "")
        title_l = title.strip().lower()
        if title_l and any(title_l == t or title_l in t or t in title_l for t in titles_l):
            continue
        if any(_box_iou_xywh(geom, g) >= 0.45 for g in atspi_geoms):
            continue
        if not _linux_app_window_exposed(
            stack, geom, capture_left, capture_top, capture_width, capture_height, xid=xid
        ):
            continue
        box = _screen_rect_to_capture_box(
            x, y, w, h,
            capture_left=capture_left,
            capture_top=capture_top,
            capture_width=capture_width,
            capture_height=capture_height,
        )
        if box is None:
            continue
        win_label = title or f"X11:0x{xid:x}"
        meta = {
            "control_type": "x11 window",
            "control_type_id": 0,
            "class_name": "x11",
            "name": win_label,
            "window": win_label,
        }
        prev = seen.get(box)
        _remember_a11y_box(boxes, seen, box, meta)
        if seen.get(box) != prev or prev is None:
            added += 1
            print(
                f"[WARN] Window without AT-SPI (X11 fallback): {win_label!r} "
                f"— internal widgets unavailable (Snap/Flatpak?).",
                flush=True,
            )
    return added


def _warn_missing_snap_browsers(Atspi) -> None:
    """Snap/Flatpak apps often never appear on the host AT-SPI bus."""
    try:
        import subprocess
        chunks: list[str] = []
        for pat in ("firefox", "snap-store", "software-boutique"):
            try:
                chunks.append(
                    subprocess.check_output(
                        ["pgrep", "-a", "-f", pat],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                )
            except subprocess.CalledProcessError:
                pass
        out = "\n".join(chunks)
    except Exception:
        return
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        return
    try:
        desktop = Atspi.get_desktop(0)
        names = set()
        for i in range(int(desktop.get_child_count())):
            app = desktop.get_child_at_index(i)
            if app is None:
                continue
            names.add((app.get_name() or "").lower())
    except Exception:
        return
    joined = "\n".join(lines).lower()
    if "firefox" in joined and ("/snap/firefox" in joined or "firefox_firefox" in joined):
        if not any("firefox" in n or "moz" in n for n in names):
            print(
                "[WARN] Firefox Snap: no host AT-SPI tree "
                "(confinement). X11 fallback = 1 window bbox. "
                "Fix: Mozilla Firefox .deb, or Flatpak with a11y, "
                "or try Terminal / Calculator / Files.",
                flush=True,
            )
    if "snap-store" in joined or "/snap/snap-store" in joined:
        if not any("snap-store" in n or "software" in n for n in names):
            print(
                "[WARN] Software Center (Snap): no host AT-SPI tree. "
                "Internal widgets unavailable via a11y.",
                flush=True,
            )


def _linux_shell_chrome_exposed(
    stack: _X11Stack,
    extents: tuple[int, int, int, int],
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> bool:
    """
    Shell / desktop widget is visible if at least one sample is not under a
    normal app window. Nearly-maximized apps must NOT be confused with the
    fullscreen desktop stage.
    """
    if not stack.ok:
        return True
    samples = _linux_sample_points(
        *extents, capture_left, capture_top, capture_width, capture_height
    )
    if not samples:
        return False
    screen_w = stack.screen_w or capture_width
    screen_h = stack.screen_h or capture_height
    for sx, sy in samples:
        top = stack.top_at(sx, sy)
        if top is None:
            return True
        if _linux_is_desktop_stage(top, screen_w, screen_h):
            return True
        # A regular app window covers this sample — try others.
        continue
    return False


def _linux_shell_walk_region(
    extents: tuple[int, int, int, int],
    capture_width: int,
    capture_height: int,
) -> bool:
    """True if this rect may contain dock / top-bar chrome (keep walking)."""
    left, top, w, h = extents
    if left + w <= 140:
        return True
    if top <= 56 and h <= 100:
        return True
    # Fullscreen / large shell roots that contain the chrome.
    if left <= 4 and top <= 4 and w >= capture_width * 0.5 and h >= capture_height * 0.5:
        return True
    return False


def _linux_should_skip_huge_container(
    role_name: str,
    box: Box,
    capture_width: int,
    capture_height: int,
) -> bool:
    """Drop giant panels/frames that hide real controls under parent-inclusion."""
    role = (role_name or "").strip().lower()
    if role not in _LINUX_SKIP_CONTAINER_ROLES:
        return False
    _x, _y, w, h = box
    capture_area = max(1, capture_width * capture_height)
    return (w * h) >= capture_area * _LINUX_MAX_CONTAINER_CAPTURE_FRAC


def _linux_is_desktop_window(
    accessible,
    app_name: str,
    attrs: dict[str, str],
) -> bool:
    """True for the desktop icons / wallpaper layer (DE-agnostic checks)."""
    if (attrs.get("window-type") or "").lower() == "desktop":
        return True
    app_key = (app_name or "").strip().lower()
    if app_key in _LINUX_DESKTOP_APP_NAMES:
        return True
    if app_key.endswith("-desktop") or app_key.endswith("_desktop"):
        return True
    try:
        name = (accessible.get_name() or "").lower()
    except Exception:
        name = ""
    # Localized / generic desktop-icon window titles
    if "desktop icon" in name or "icônes du bureau" in name or "icones du bureau" in name:
        return True
    if name in {"desktop", "bureau"}:
        return True
    return False


def _linux_should_skip_window(
    Atspi,
    accessible,
    app_name: str,
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
    x11_stack: _X11Stack | None = None,
) -> bool:
    """True if this top-level window should not be scanned."""
    app_key = app_name.lower()
    if app_key in _LINUX_SKIP_APP_NAMES:
        return True

    try:
        role_name = (accessible.get_role_name() or "").lower()
    except Exception:
        role_name = ""
    if role_name not in _LINUX_WINDOW_ROLES:
        return True

    if not _linux_accessible_showing(Atspi, accessible):
        return True

    parsed = _linux_extents_screen(Atspi, accessible)
    if parsed is None:
        return True

    attrs = _linux_window_attrs(accessible)
    is_desktop = _linux_is_desktop_window(accessible, app_name, attrs)
    is_shell = _linux_is_shell_app(app_name)

    # Desktop icons layer: always keep the root; per-icon exposure while walking.
    if is_desktop:
        return False

    # Prefer X11 frame geometry: GTK often reports bogus AT-SPI SCREEN (0,0,…).
    hit_test = parsed
    xid: int | None = None
    if x11_stack is not None and x11_stack.ok and not is_shell:
        try:
            win_name = str(accessible.get_name() or "")
        except Exception:
            win_name = ""
        xid = _match_x11_xid(x11_stack, parsed, win_name)
        if xid is None:
            # Not in mapped/non-hidden X11 clients → minimized or withdrawn.
            return True
        for id_, x, y, w, h in x11_stack.windows:
            if id_ == xid:
                hit_test = (x, y, w, h)
                break
        # Keep every window that still has some exposed pixels (Cursor behind a
        # small Calculator must stay in scope).
        if not _linux_app_window_exposed(
            x11_stack,
            hit_test,
            capture_left,
            capture_top,
            capture_width,
            capture_height,
            xid=xid,
        ):
            return True

    if not _screen_rect_intersects_capture(
        *hit_test,
        capture_left,
        capture_top,
        capture_width,
        capture_height,
    ):
        return True

    return False


def _a11y_widget_quality(meta: dict[str, Any]) -> tuple[int, int, int]:
    """Higher is better when two widgets share the same bbox (prefer real controls)."""
    control = str(meta.get("control_type") or "")
    clickable = default_clickable_control_types()
    interactive = 1 if control in clickable else 0
    named = 1 if str(meta.get("name") or "").strip() else 0
    containerish = control in _LINUX_LAYOUT_ROLES or control in {
        "Pane", "Window", "Group", "ToolBar", "Tab", "panel", "filler", "section",
    }
    not_container = 0 if containerish else 1
    return (interactive, named, not_container)


def _remember_a11y_box(
    boxes: List[A11yWidget],
    seen: dict[Box, int],
    box: Box,
    meta: dict[str, Any],
) -> None:
    """Insert or replace a widget for this bbox, keeping the higher-quality one."""
    x, y, w, h = box
    if w < _LINUX_MIN_BOX_SIDE or h < _LINUX_MIN_BOX_SIDE:
        return
    entry: A11yWidget = {"x": x, "y": y, "w": w, "h": h, **meta}
    if box not in seen:
        seen[box] = len(boxes)
        boxes.append(entry)
        return
    idx = seen[box]
    if _a11y_widget_quality(meta) > _a11y_widget_quality(boxes[idx]):
        boxes[idx] = entry


def _collect_atspi_boxes(
    Atspi,
    root_element,
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
    boxes: List[A11yWidget],
    seen: dict[Box, int],
    *,
    is_shell: bool = False,
    is_desktop: bool = False,
    x11_stack: _X11Stack | None = None,
    window_label: str = "",
) -> None:
    window_origin: tuple[int, int] | None = None
    own_xid: int | None = None
    win_label = (window_label or "").strip() or UNKNOWN_WINDOW_LABEL
    if x11_stack is not None and x11_stack.ok and not is_shell and not is_desktop:
        try:
            root_name = str(root_element.get_name() or "")
        except Exception:
            root_name = ""
        # Prefer WINDOW size of the frame (logical content) over broken SCREEN.
        atspi_wh: tuple[int, int] | None = None
        try:
            comp = root_element.get_component_iface()
            if comp is not None:
                win_ext = comp.get_extents(Atspi.CoordType.WINDOW)
                if int(win_ext.width) > 0 and int(win_ext.height) > 0:
                    atspi_wh = (int(win_ext.width), int(win_ext.height))
        except Exception:
            atspi_wh = None
        if atspi_wh is None:
            root_ext = _linux_extents_screen(Atspi, root_element)
            if root_ext is not None:
                atspi_wh = (root_ext[2], root_ext[3])
        root_ext = _linux_extents_screen(Atspi, root_element)
        own_xid = _match_x11_xid(
            x11_stack,
            root_ext or (0, 0, 1, 1),
            root_name,
        )
        if own_xid is not None and atspi_wh is not None:
            for id_, x, y, w, h in x11_stack.windows:
                if id_ == own_xid:
                    window_origin = _linux_content_origin(
                        (x, y, w, h),
                        atspi_wh,
                        x11_stack.frame_extents.get(own_xid),
                    )
                    break

    def walk(element, depth: int = 0, parent_role: str = "") -> None:
        if element is None or depth > 64:
            return

        # Fast state gate — skip dead subtrees entirely (big win on Electron).
        try:
            states = element.get_state_set()
            if states is not None and states.contains(Atspi.StateType.DEFUNCT):
                return
            showing = True
            if states is not None:
                if states.contains(Atspi.StateType.ICONIFIED):
                    showing = False
                elif states.contains(Atspi.StateType.SHOWING):
                    showing = True
                else:
                    showing = bool(states.contains(Atspi.StateType.VISIBLE))
        except Exception:
            showing = True

        skip_box = not showing
        prune_children = False
        role = ""

        if not skip_box:
            try:
                role_name, role_id = _linux_role_name(Atspi, element)
                role = _normalize_atspi_role(role_name)
                # Nautilus icon grid: outer table-cell hosts nested cell/image/label
                # (padding ghosts). Keep only the outer cell.
                nested_icon_chrome = parent_role == "table cell" and role in {
                    "table cell",
                    "image",
                    "label",
                    "icon",
                }
                parsed = _linux_extents_screen(
                    Atspi, element, window_origin=window_origin
                )
                if parsed is not None and not nested_icon_chrome:
                    # Shell: prune closed menus / popovers outside dock+topbar.
                    if is_shell:
                        if not _linux_shell_walk_region(
                            parsed, capture_width, capture_height
                        ):
                            return
                    # Desktop icons under an app window → drop.
                    elif is_desktop and x11_stack is not None and x11_stack.ok:
                        if not _linux_shell_chrome_exposed(
                            x11_stack,
                            parsed,
                            capture_left,
                            capture_top,
                            capture_width,
                            capture_height,
                        ):
                            parsed = None

                    if parsed is not None and is_shell:
                        if not _linux_shell_persistent_chrome(
                            parsed,
                            screen_w=capture_width,
                            screen_h=capture_height,
                        ):
                            parsed = None

                    if parsed is not None:
                        box = _screen_rect_to_capture_box(
                            *parsed,
                            capture_left=capture_left,
                            capture_top=capture_top,
                            capture_width=capture_width,
                            capture_height=capture_height,
                        )
                        if box is not None:
                            _x, _y, bw, bh = box
                            dock_rail = (
                                is_shell
                                and role in _LINUX_SKIP_CONTAINER_ROLES
                                and bw <= 120
                                and bh >= capture_height * 0.45
                            )
                            emit = True
                            if dock_rail:
                                emit = False
                            elif is_shell:
                                emit = role in _LINUX_SHELL_CHROME_ROLES
                            elif is_desktop:
                                emit = role in _LINUX_DESKTOP_ROLES
                            elif role in _LINUX_LAYOUT_ROLES or role in _LINUX_NOISE_ROLES:
                                emit = False
                            elif _linux_should_skip_huge_container(
                                role, box, capture_width, capture_height
                            ):
                                emit = False
                            # Drop widgets covered by another window.
                            if emit and x11_stack is not None and x11_stack.ok:
                                if is_shell:
                                    if not _linux_shell_chrome_exposed(
                                        x11_stack,
                                        parsed,
                                        capture_left,
                                        capture_top,
                                        capture_width,
                                        capture_height,
                                    ):
                                        emit = False
                                elif not is_desktop:
                                    # No X11 id → cannot prove visibility (minimized
                                    # / stolen match). Never emit app widgets then.
                                    if own_xid is None or not _linux_app_window_exposed(
                                        x11_stack,
                                        parsed,
                                        capture_left,
                                        capture_top,
                                        capture_width,
                                        capture_height,
                                        xid=own_xid,
                                    ):
                                        emit = False
                            if emit:
                                metadata = _linux_metadata(
                                    Atspi, element, role_name, role_id
                                )
                                metadata["window"] = win_label
                                _remember_a11y_box(boxes, seen, box, metadata)
            except Exception:
                pass
        elif not is_shell and not is_desktop:
            # Hidden app subtrees: don't pay D-Bus to walk them.
            prune_children = True

        if prune_children:
            return

        # Shell trees are huge — hard-cap depth after region pruning.
        if is_shell and depth >= 14:
            return
        # No geometry on shell node: don't spelunk forever.
        if is_shell and skip_box and depth >= 4:
            return
        if is_shell and not skip_box:
            # If we had extents but they were outside interest, we already returned.
            pass

        try:
            child_count = int(element.get_child_count())
        except Exception:
            return
        for idx in range(child_count):
            try:
                child = element.get_child_at_index(idx)
            except Exception:
                continue
            walk(child, depth + 1, parent_role=role or parent_role)

    walk(root_element)


def find_visible_window_roots_linux(
    Atspi,
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
    x11_stack: _X11Stack | None = None,
) -> list[tuple[object, str, bool, bool]]:
    """Top-level AT-SPI frames/windows that are actually visible in the capture.

    Each root is (element, label, is_shell, is_desktop).
    """
    roots: list[tuple[object, str, bool, bool]] = []
    try:
        desktop = Atspi.get_desktop(0)
    except Exception as exc:
        print(f"[WARN] AT-SPI get_desktop failed: {exc}", flush=True)
        return []

    try:
        app_count = int(desktop.get_child_count())
    except Exception as exc:
        print(f"[WARN] AT-SPI desktop child_count failed: {exc}", flush=True)
        return []

    for i in range(app_count):
        try:
            app = desktop.get_child_at_index(i)
        except Exception:
            continue
        if app is None:
            continue
        try:
            app_name = str(app.get_name() or "")
        except Exception:
            app_name = ""
        app_is_shell = _linux_is_shell_app(app_name)

        try:
            win_count = int(app.get_child_count())
        except Exception:
            continue
        for j in range(win_count):
            try:
                win = app.get_child_at_index(j)
            except Exception:
                continue
            if win is None:
                continue
            if _linux_should_skip_window(
                Atspi,
                win,
                app_name,
                capture_left,
                capture_top,
                capture_width,
                capture_height,
                x11_stack=x11_stack,
            ):
                continue
            attrs = _linux_window_attrs(win)
            is_desktop = _linux_is_desktop_window(win, app_name, attrs)
            # Desktop-icons layer is not shell chrome even if the process is.
            is_shell = app_is_shell and not is_desktop
            roots.append(
                (win, _linux_window_label(app_name, win), is_shell, is_desktop)
            )

    return roots


def _get_boxes_linux(
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> List[A11yWidget]:
    """Enumerate AT-SPI2 widgets for visible top-level windows (Linux)."""
    Atspi = _import_atspi()
    if Atspi is None:
        print("[WARN] AT-SPI2 / PyGObject unavailable.")
        print("[WARN] Install: sudo apt install python3-gi gir1.2-atspi-2.0 at-spi2-core")
        return []

    try:
        # 0 = success; safe to call more than once.
        Atspi.init()
    except Exception as exc:
        print(f"[WARN] AT-SPI init failed: {exc}")
        return []

    x11_stack = _X11Stack()
    if not x11_stack.ok:
        print(
            "[WARN] X11 stacking unavailable — hidden-window filtering is limited.",
            flush=True,
        )

    _warn_missing_snap_browsers(Atspi)

    scope_roots = find_visible_window_roots_linux(
        Atspi,
        capture_left,
        capture_top,
        capture_width,
        capture_height,
        x11_stack=x11_stack,
    )

    seen: dict[Box, int] = {}
    boxes: List[A11yWidget] = []
    atspi_geoms: list[tuple[int, int, int, int]] = []
    atspi_titles: set[str] = set()

    if scope_roots:
        # Shell last: apps/desktop first, then dock/top bar (region-pruned walk).
        scope_roots = sorted(scope_roots, key=lambda r: 1 if r[2] else 0)
        print(
            f"[INFO] Visible windows (scope): {len(scope_roots)} top-level root(s)",
            flush=True,
        )
        for root_element, label, is_shell, is_desktop in scope_roots:
            tag = " [shell]" if is_shell else (" [desktop]" if is_desktop else "")
            print(f"  - {label}{tag}", flush=True)
            try:
                win_name = str(root_element.get_name() or "")
            except Exception:
                win_name = ""
            if win_name:
                atspi_titles.add(win_name)
            ext = _linux_extents_screen(Atspi, root_element)
            # Prefer X11 frame for orphan matching — AT-SPI SCREEN can be (0,0,…).
            geom_for_orphan = ext
            if (
                x11_stack.ok
                and not is_desktop
                and not is_shell
                and ext is not None
            ):
                xid = _match_x11_xid(x11_stack, ext, win_name)
                if xid is not None:
                    for id_, x, y, w, h in x11_stack.windows:
                        if id_ == xid:
                            geom_for_orphan = (x, y, w, h)
                            break
            # Don't use the fullscreen desktop layer for orphan matching (IoU≈1 with everything).
            if geom_for_orphan is not None and not is_desktop and not is_shell:
                atspi_geoms.append(geom_for_orphan)
            _collect_atspi_boxes(
                Atspi,
                root_element,
                capture_left,
                capture_top,
                capture_width,
                capture_height,
                boxes,
                seen,
                is_shell=is_shell,
                is_desktop=is_desktop,
                x11_stack=x11_stack,
                window_label=label,
            )
        _enrich_desktop_icon_names(boxes)
    else:
        print("[WARN] No visible AT-SPI root.", flush=True)

    orphan_n = _collect_x11_orphan_windows(
        x11_stack,
        atspi_geoms,
        atspi_titles,
        capture_left,
        capture_top,
        capture_width,
        capture_height,
        boxes,
        seen,
    )
    if orphan_n:
        print(f"[INFO] X11 fallback: {orphan_n} window(s) without AT-SPI tree.", flush=True)

    if not boxes:
        print("[WARN] Could not determine target window for accessibility scan.")
        return []

    print(f"[INFO] Widgets collected (raw): {len(boxes)} bbox", flush=True)
    return boxes


def _get_boxes_darwin(
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> List[A11yWidget]:
    # TODO: macOS Accessibility API (AXUIElementCopyAttributeValue).
    print("[WARN] Accessibility boxes are not implemented on macOS yet.")
    return []

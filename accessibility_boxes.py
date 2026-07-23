"""Platform-specific UI accessibility bounding box extraction."""

from __future__ import annotations

import sys
from typing import Any, List, Sequence, Tuple, TypedDict

Box = Tuple[int, int, int, int]


class A11yWidget(TypedDict):
    x: int
    y: int
    w: int
    h: int
    control_type: str
    control_type_id: int
    class_name: str
    name: str


# Optional allowlist of UIA control_type names; None = keep all.
A11Y_CONTROL_TYPE_ALLOWLIST: set[str] | None = None

CLICKABLE_CONTROL_TYPES = frozenset({
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
    }


def get_accessibility_boxes(
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> List[A11yWidget]:
    """Return accessibility widgets after default filters (clickable + parent inclusion)."""
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
        for j, child in enumerate(boxes):
            if i == j or j in drop:
                continue
            if _box_contains(parent, child):
                drop.add(j)
    return [box for i, box in enumerate(boxes) if i not in drop]


def apply_a11y_filters(
    boxes: Sequence[A11yWidget] | List[A11yWidget],
    enabled_types: set[str] | frozenset[str] | None = None,
    parent_inclusion: bool = True,
) -> List[A11yWidget]:
    """Filter raw UIA widgets by control type and optional parent-child inclusion."""
    types = set(CLICKABLE_CONTROL_TYPES) if enabled_types is None else set(enabled_types)
    if A11Y_CONTROL_TYPE_ALLOWLIST is not None:
        types &= A11Y_CONTROL_TYPE_ALLOWLIST

    filtered = [b for b in boxes if str(b.get("control_type") or "") in types]
    if parent_inclusion:
        filtered = _suppress_contained_children(filtered)
    return filtered


def filter_widget_boxes(boxes: List[A11yWidget]) -> tuple[List[A11yWidget], str]:
    """Apply default parent-priority dedup (legacy helper)."""
    filtered = _suppress_contained_children(boxes)
    summary = f"{len(boxes)} -> {len(filtered)} (parent-priority)"
    return filtered, summary


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
) -> List[A11yWidget]:
    boxes: List[A11yWidget] = []
    if seen is None:
        seen = set()

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
        f"[INFO] Fenêtres visibles (scope): {len(scope_roots)} top-level root(s)",
        flush=True,
    )
    for _, label in scope_roots:
        print(f"  - {label}", flush=True)

    seen: set[Box] = set()
    boxes: List[A11yWidget] = []
    for root_element, _label in scope_roots:
        boxes.extend(
            _collect_uia_boxes(
                automation,
                root_element,
                capture_left,
                capture_top,
                capture_width,
                capture_height,
                seen=seen,
            )
        )

    print(f"[INFO] Widgets collectés (brut): {len(boxes)} bbox", flush=True)
    return boxes


def _get_boxes_linux(
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> List[A11yWidget]:
    # TODO: AT-SPI2 / a11y bus (Linux).
    print("[WARN] Accessibility boxes are not implemented on Linux yet.")
    return []


def _get_boxes_darwin(
    capture_left: int,
    capture_top: int,
    capture_width: int,
    capture_height: int,
) -> List[A11yWidget]:
    # TODO: macOS Accessibility API (AXUIElementCopyAttributeValue).
    print("[WARN] Accessibility boxes are not implemented on macOS yet.")
    return []

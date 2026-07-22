"""WidgetFusion session: hover + YOLO + accessibility + fusion workflow."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Literal, Sequence, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

Box = Tuple[int, int, int, int]
Source = Literal["hover", "yolo", "a11y"]
Layer = Literal["hover", "yolo", "a11y", "fused"]
View = Literal["hover", "yolo", "a11y", "all", "fused"]

SOURCE_PRIORITY: tuple[Source, ...] = ("a11y", "hover", "yolo")

SOURCE_LABELS: dict[Source, str] = {
    "hover": "Hover (vert)",
    "yolo": "YOLO (orange)",
    "a11y": "Accessibilité (bleu)",
}

Cluster = list[tuple[Source, int, Any]]

DEFAULT_FUSION_MIN_IOU = 0.5
DEFAULT_INCLUSION_COVERAGE = 0.9


@dataclass
class CombinedModeConfig:
    enable_hover: bool = True
    hover_autoscan: bool = False
    enable_yolo: bool = True
    enable_a11y: bool = True

    def phase_order(self) -> list[str]:
        phases: list[str] = []
        if self.enable_hover:
            phases.append("hover")
        if self.enable_yolo:
            phases.append("yolo")
        if self.enable_a11y:
            phases.append("a11y")
        return phases


def box_coords(item: Any) -> Box:
    if isinstance(item, dict):
        return (int(item["x"]), int(item["y"]), int(item["w"]), int(item["h"]))
    return item


def box_area(item: Any) -> int:
    _, _, w, h = box_coords(item)
    return max(0, w) * max(0, h)


def intersection_area(a: Any, b: Any) -> int:
    ax, ay, aw, ah = box_coords(a)
    bx, by, bw, bh = box_coords(b)
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0
    return (ix1 - ix0) * (iy1 - iy0)


def box_iou(a: Any, b: Any) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def soft_inclusion(a: Any, b: Any, min_coverage: float = DEFAULT_INCLUSION_COVERAGE) -> bool:
    """
    True if most of the smaller box lies inside the other (allows slight overflow).
    coverage = intersection / area(smaller) ≥ min_coverage.
    """
    inter = intersection_area(a, b)
    if inter <= 0:
        return False
    smaller = min(box_area(a), box_area(b))
    if smaller <= 0:
        return False
    return (inter / smaller) >= min_coverage


def boxes_match(
    a: Any,
    b: Any,
    min_iou: float,
    inclusion_coverage: float = DEFAULT_INCLUSION_COVERAGE,
) -> bool:
    """Same widget if IoU ≥ min_iou, or soft inclusion (coverage of smaller box)."""
    if box_iou(a, b) >= min_iou:
        return True
    return soft_inclusion(a, b, min_coverage=inclusion_coverage)


def _index_boxes(source: Source, boxes: Sequence[Any]) -> list[tuple[Source, int, Any]]:
    return [(source, i, box) for i, box in enumerate(boxes)]


def unique_sources(hits: Iterable[tuple[Source, int, Any]]) -> set[Source]:
    return {source for source, _, _ in hits}


@dataclass
class FusionConfig:
    mode: Literal["auto", "manual"]
    source_priority: tuple[Source, ...] = SOURCE_PRIORITY
    min_sources: int = 2
    min_iou: float = DEFAULT_FUSION_MIN_IOU
    inclusion_coverage: float = DEFAULT_INCLUSION_COVERAGE
    include_orphans: bool = True


def _index_all_boxes(
    hover_boxes: Sequence[Any],
    yolo_boxes: Sequence[Any],
    a11y_boxes: Sequence[Any],
    source_priority: Sequence[Source] | None = None,
) -> list[tuple[Source, int, Any]]:
    """Index boxes with anchors ordered by source priority (highest first)."""
    by_source: dict[Source, Sequence[Any]] = {
        "hover": hover_boxes,
        "yolo": yolo_boxes,
        "a11y": a11y_boxes,
    }
    priority = tuple(source_priority or SOURCE_PRIORITY)
    indexed: list[tuple[Source, int, Any]] = []
    for source in priority:
        indexed.extend(_index_boxes(source, by_source.get(source, ())))
    for source, boxes in by_source.items():
        if source not in priority:
            indexed.extend(_index_boxes(source, boxes))
    return indexed


def build_fusion_groups(
    hover_boxes: Sequence[Any],
    yolo_boxes: Sequence[Any],
    a11y_boxes: Sequence[Any],
    min_iou: float,
    min_sources: int = 2,
    inclusion_coverage: float = DEFAULT_INCLUSION_COVERAGE,
    source_priority: Sequence[Source] | None = None,
) -> list[Cluster]:
    """
    Match widgets without transitive clustering.
    Anchors are visited in source_priority order so the preferred source
    claims matches first.
    """
    indexed = _index_all_boxes(
        hover_boxes, yolo_boxes, a11y_boxes, source_priority=source_priority
    )
    if not indexed:
        return []

    used: set[tuple[Source, int]] = set()
    groups: list[Cluster] = []

    for i, entry_i in enumerate(indexed):
        src_i, idx_i, box_i = entry_i
        if (src_i, idx_i) in used:
            continue

        group: Cluster = [entry_i]
        for j, entry_j in enumerate(indexed):
            if i == j:
                continue
            src_j, idx_j, box_j = entry_j
            if src_j == src_i or (src_j, idx_j) in used:
                continue
            if boxes_match(
                box_i,
                box_j,
                min_iou,
                inclusion_coverage=inclusion_coverage,
            ):
                group.append(entry_j)

        if len(unique_sources(group)) >= min_sources:
            groups.append(group)
            for src, idx, _ in group:
                used.add((src, idx))

    return groups


def collect_orphan_boxes(
    hover_boxes: Sequence[Any],
    yolo_boxes: Sequence[Any],
    a11y_boxes: Sequence[Any],
    groups: Sequence[Cluster],
) -> list[Any]:
    """Boxes from any source that did not match another source in a fusion group."""
    used: set[tuple[Source, int]] = set()
    for group in groups:
        for src, idx, _ in group:
            used.add((src, idx))

    orphans: list[Any] = []
    for src, boxes in (("hover", hover_boxes), ("yolo", yolo_boxes), ("a11y", a11y_boxes)):
        for idx, box in enumerate(boxes):
            if (src, idx) not in used:
                orphans.append(box)
    return orphans


def fuse_groups_auto(
    groups: Sequence[Cluster],
    source_priority: Sequence[Source],
) -> list[Any]:
    priority = tuple(source_priority)
    fused: list[Any] = []
    for group in groups:
        sources = unique_sources(group)
        best_source = min(sources, key=lambda s: priority.index(s))
        fused.append(resolve_fused_box(group, best_source))
    return fused


def cluster_a11y_metadata(cluster: Cluster) -> dict[str, Any] | None:
    """Return a11y UIA fields from the cluster if an a11y box is present."""
    for src, _, box in cluster:
        if src != "a11y" or not isinstance(box, dict):
            continue
        return {
            "control_type": box.get("control_type", ""),
            "class_name": box.get("class_name", ""),
            "name": box.get("name", ""),
        }
    return None


def resolve_fused_box(cluster: Cluster, chosen_source: Source) -> Any:
    """
    Keep geometry from chosen_source; attach a11y metadata if the cluster has any.
    """
    geom = cluster_box_for_source(cluster, chosen_source)
    if geom is None:
        geom = cluster[0][2]
    x, y, w, h = box_coords(geom)
    meta = cluster_a11y_metadata(cluster)
    if meta is None:
        return geom
    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        **meta,
    }


def cluster_union_rect(cluster: Cluster) -> Box:
    xs, ys, x2s, y2s = [], [], [], []
    for _, _, box in cluster:
        x, y, w, h = box_coords(box)
        xs.append(x)
        ys.append(y)
        x2s.append(x + w)
        y2s.append(y + h)
    x0, y0 = min(xs), min(ys)
    return (x0, y0, max(x2s) - x0, max(y2s) - y0)


def combined_effective_layer(phase: str, view: str) -> Layer | None:
    """
    Single layer for M / undo / hover target.
    None = vue « all » (affichage seul, pas d'édition).
    """
    if phase != "review":
        if phase in ("hover", "yolo", "a11y"):
            return phase  # type: ignore[return-value]
        return None
    if view == "all":
        return None
    if view in ("hover", "yolo", "a11y", "fused"):
        return view  # type: ignore[return-value]
    return None


def combined_manual_editing_allowed(phase: str, view: str) -> bool:
    """M works like S/L/G on one list — never on the read-only « all » view."""
    return combined_effective_layer(phase, view) is not None


def combined_hover_diff_enabled(phase: str, view: str) -> bool:
    """Hover diff in hover phase; in review only on single-layer views (not « all »)."""
    if phase == "hover":
        return True
    if phase == "review":
        return view != "all"
    return False


def combined_hover_diff_target_layer(phase: str, view: str) -> Layer | None:
    """Which list receives hover-diff detections in combined mode."""
    if not combined_hover_diff_enabled(phase, view):
        return None
    if phase == "hover" or view == "hover":
        return "hover"
    if phase == "review" and view == "fused":
        return "fused"
    if phase == "review" and view in ("yolo", "a11y"):
        return "hover"
    return "hover"


def combined_overlay_layers(phase: str, view: str) -> list[Layer]:
    """Which layers to draw (order matters for hit-testing in single-layer views)."""
    if phase != "review":
        if phase in ("hover", "yolo", "a11y"):
            return [phase]  # type: ignore[list-item]
        return []
    if view == "all":
        return ["hover", "yolo", "a11y"]
    if view == "fused":
        return ["fused"]
    if view in ("hover", "yolo", "a11y"):
        return [view]  # type: ignore[list-item]
    return []


class CombinedConfigDialog(QDialog):
    """Choose which detection methods to run in the combined session."""

    def __init__(self, a11y_available: bool = True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WidgetFusion — méthodes de détection")
        self.setModal(True)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Choisissez les méthodes à exécuter (dans l'ordre : hover → YOLO → accessibilité).\n"
            "Appuyez sur Entrée entre chaque étape."
        ))

        self.hover_cb = QCheckBox("Hover diff")
        self.hover_cb.setChecked(True)
        layout.addWidget(self.hover_cb)

        hover_group = QGroupBox("Hover")
        hover_layout = QVBoxLayout(hover_group)
        self.hover_manual_rb = QRadioButton("Survol manuel")
        self.hover_autoscan_rb = QRadioButton("Autoscan (comme Y)")
        self.hover_manual_rb.setChecked(True)
        hover_mode_group = QButtonGroup(self)
        hover_mode_group.addButton(self.hover_manual_rb)
        hover_mode_group.addButton(self.hover_autoscan_rb)
        hover_layout.addWidget(self.hover_manual_rb)
        hover_layout.addWidget(self.hover_autoscan_rb)
        layout.addWidget(hover_group)

        self.yolo_cb = QCheckBox("YOLO")
        self.yolo_cb.setChecked(True)
        layout.addWidget(self.yolo_cb)

        self.a11y_cb = QCheckBox("Accessibilité (Windows UIA)")
        self.a11y_cb.setChecked(a11y_available)
        self.a11y_cb.setEnabled(a11y_available)
        if not a11y_available:
            self.a11y_cb.setToolTip("Disponible uniquement sur Windows pour le moment.")
        layout.addWidget(self.a11y_cb)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._result: CombinedModeConfig | None = None

    def _accept(self) -> None:
        if not (self.hover_cb.isChecked() or self.yolo_cb.isChecked() or self.a11y_cb.isChecked()):
            return
        self._result = CombinedModeConfig(
            enable_hover=self.hover_cb.isChecked(),
            hover_autoscan=self.hover_autoscan_rb.isChecked(),
            enable_yolo=self.yolo_cb.isChecked(),
            enable_a11y=self.a11y_cb.isChecked(),
        )
        self.accept()

    def config(self) -> CombinedModeConfig | None:
        return self._result


def _help_button(parent: QWidget, title: str, text: str) -> QPushButton:
    """Small « ? » button that shows a short explanation."""
    btn = QPushButton("?")
    btn.setFixedWidth(24)
    btn.setToolTip(title)
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    btn.clicked.connect(
        lambda: QMessageBox.information(parent, title, text)
    )
    return btn


def _label_with_help(parent: QWidget, label: str, help_title: str, help_text: str) -> QWidget:
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(QLabel(label))
    layout.addWidget(_help_button(parent, help_title, help_text))
    layout.addStretch()
    return row


class FusionConfigDialog(QDialog):
    """Choose fusion thresholds, priority, and auto vs manual picking."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fusion")
        self.setModal(True)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        threshold_group = QGroupBox("Matching")
        threshold_form = QFormLayout(threshold_group)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(1, 100)
        self.iou_spin.setDecimals(0)
        self.iou_spin.setSuffix(" %")
        self.iou_spin.setValue(DEFAULT_FUSION_MIN_IOU * 100)
        threshold_form.addRow(
            _label_with_help(
                self,
                "IoU min :",
                "IoU minimum",
                "Recouvrement entre deux bbox (intersection / union).\n"
                "Si IoU ≥ ce seuil → même widget.",
            ),
            self.iou_spin,
        )

        self.inclusion_spin = QDoubleSpinBox()
        self.inclusion_spin.setRange(1, 100)
        self.inclusion_spin.setDecimals(0)
        self.inclusion_spin.setSuffix(" %")
        self.inclusion_spin.setValue(DEFAULT_INCLUSION_COVERAGE * 100)
        threshold_form.addRow(
            _label_with_help(
                self,
                "Inclusion min :",
                "Inclusion souple",
                "Part minimale de la plus petite bbox qui doit être dans "
                "l’intersection avec l’autre.\n"
                "Exemple 90 % : un léger débordement est OK.\n"
                "100 % = inclusion stricte (fragile au moindre pixel).",
            ),
            self.inclusion_spin,
        )
        layout.addWidget(threshold_group)

        orphans_row = QHBoxLayout()
        self.orphans_cb = QCheckBox("Garder les bbox isolées")
        self.orphans_cb.setChecked(True)
        orphans_row.addWidget(self.orphans_cb)
        orphans_row.addWidget(
            _help_button(
                self,
                "Bbox isolées",
                "Bbox détectées par une seule méthode, sans correspondance "
                "dans une autre source. Si coché, elles sont ajoutées à la "
                "vue fusion ; sinon elles sont ignorées.",
            )
        )
        orphans_row.addStretch()
        layout.addLayout(orphans_row)

        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.auto_rb = QRadioButton("Automatique (priorité globale)")
        self.manual_rb = QRadioButton("Manuel (choix widget par widget)")
        self.auto_rb.setChecked(True)
        mode_btn_group = QButtonGroup(self)
        mode_btn_group.addButton(self.auto_rb)
        mode_btn_group.addButton(self.manual_rb)
        auto_row = QHBoxLayout()
        auto_row.addWidget(self.auto_rb)
        auto_row.addWidget(
            _help_button(
                self,
                "Mode automatique",
                "Pour chaque conflit, la source la plus haute dans "
                "l’ordre de priorité est retenue partout.",
            )
        )
        auto_row.addStretch()
        manual_row = QHBoxLayout()
        manual_row.addWidget(self.manual_rb)
        manual_row.addWidget(
            _help_button(
                self,
                "Mode manuel",
                "Une fenêtre s’ouvre pour chaque widget en conflit : "
                "vous choisissez hover, YOLO ou accessibilité.\n"
                "L’ordre de priorité sert de présélection par défaut.",
            )
        )
        manual_row.addStretch()
        mode_layout.addLayout(auto_row)
        mode_layout.addLayout(manual_row)
        layout.addWidget(mode_group)

        priority_group = QGroupBox("Priorité")
        priority_layout = QVBoxLayout(priority_group)
        priority_header = QHBoxLayout()
        priority_header.addWidget(QLabel("Ordre (haut = prioritaire)"))
        priority_header.addWidget(
            _help_button(
                self,
                "Ordre de priorité",
                "Automatique : la source en haut gagne sur chaque conflit.\n"
                "Manuel : cet ordre pré-coche la source proposée, "
                "modifiable à chaque widget.",
            )
        )
        priority_header.addStretch()
        priority_layout.addLayout(priority_header)

        self.priority_list = QListWidget()
        for source in SOURCE_PRIORITY:
            item = QListWidgetItem(SOURCE_LABELS[source])
            item.setData(Qt.ItemDataRole.UserRole, source)
            self.priority_list.addItem(item)
        self.priority_list.setMaximumHeight(100)
        priority_layout.addWidget(self.priority_list)

        btn_row = QHBoxLayout()
        up_btn = QPushButton("Monter")
        down_btn = QPushButton("Descendre")
        up_btn.clicked.connect(self._move_priority_up)
        down_btn.clicked.connect(self._move_priority_down)
        btn_row.addWidget(up_btn)
        btn_row.addWidget(down_btn)
        btn_row.addStretch()
        priority_layout.addLayout(btn_row)
        layout.addWidget(priority_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._result: FusionConfig | None = None

    def _move_priority_up(self) -> None:
        row = self.priority_list.currentRow()
        if row <= 0:
            return
        item = self.priority_list.takeItem(row)
        self.priority_list.insertItem(row - 1, item)
        self.priority_list.setCurrentRow(row - 1)

    def _move_priority_down(self) -> None:
        row = self.priority_list.currentRow()
        if row < 0 or row >= self.priority_list.count() - 1:
            return
        item = self.priority_list.takeItem(row)
        self.priority_list.insertItem(row + 1, item)
        self.priority_list.setCurrentRow(row + 1)

    def _priority_tuple(self) -> tuple[Source, ...]:
        out: list[Source] = []
        for i in range(self.priority_list.count()):
            item = self.priority_list.item(i)
            source = item.data(Qt.ItemDataRole.UserRole)
            out.append(source)
        return tuple(out)

    def _accept(self) -> None:
        mode: Literal["auto", "manual"] = "auto" if self.auto_rb.isChecked() else "manual"
        self._result = FusionConfig(
            mode=mode,
            source_priority=self._priority_tuple(),
            min_iou=self.iou_spin.value() / 100.0,
            inclusion_coverage=self.inclusion_spin.value() / 100.0,
            include_orphans=self.orphans_cb.isChecked(),
        )
        self.accept()

    def config(self) -> FusionConfig | None:
        return self._result


def show_fusion_config_dialog(parent=None) -> FusionConfig | None:
    dialog = FusionConfigDialog(parent=parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.config()


def default_source_for_cluster(cluster: Cluster, priority: Sequence[Source]) -> Source:
    present = unique_sources(cluster)
    for source in priority:
        if source in present:
            return source
    return next(iter(present))


def cluster_box_for_source(cluster: Cluster, source: Source) -> Any | None:
    for src, _, box in cluster:
        if src == source:
            return box
    return None


class FusionWidgetPickDialog(QDialog):
    """Pick which detection source to keep for one conflicting widget."""

    def __init__(
        self,
        index: int,
        total: int,
        cluster: Cluster,
        default_source: Source,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Fusion — widget {index} / {total}")
        self.setModal(True)
        self.setMinimumWidth(460)

        self._cluster = cluster
        self._selected_source: Source | None = None
        self._source_order: list[Source] = []

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel(f"<b>Widget {index} / {total}</b> — source à garder ?"))
        header.addWidget(
            _help_button(
                self,
                "Choix de source",
                "Plusieurs méthodes ont détecté le même élément.\n"
                "Choisissez quelle bbox conserver pour la fusion.",
            )
        )
        header.addStretch()
        layout.addLayout(header)

        sources = [s for s in SOURCE_PRIORITY if s in unique_sources(cluster)]
        group = QGroupBox("Source à retenir")
        group_layout = QVBoxLayout(group)
        self._btn_group = QButtonGroup(self)

        for src in sources:
            box = cluster_box_for_source(cluster, src)
            if box is None:
                continue
            x, y, w, h = box_coords(box)
            rb = QRadioButton(f"{SOURCE_LABELS[src]}  —  {w}×{h} px  @ ({x}, {y})")
            self._source_order.append(src)
            self._btn_group.addButton(rb, len(self._source_order) - 1)
            group_layout.addWidget(rb)
            if src == default_source:
                rb.setChecked(True)

        layout.addWidget(group)

        buttons = QDialogButtonBox()
        next_label = "Terminer" if index == total else "Suivant"
        next_btn = buttons.addButton(next_label, QDialogButtonBox.ButtonRole.AcceptRole)
        cancel_btn = buttons.addButton("Annuler tout", QDialogButtonBox.ButtonRole.RejectRole)
        next_btn.clicked.connect(self._accept)
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        checked = self._btn_group.checkedButton()
        if checked is None:
            return
        idx = self._btn_group.id(checked)
        if idx < 0 or idx >= len(self._source_order):
            return
        self._selected_source = self._source_order[idx]
        self.accept()

    def selected_box(self) -> Any:
        assert self._selected_source is not None
        return resolve_fused_box(self._cluster, self._selected_source)

    def selected_source(self) -> Source:
        assert self._selected_source is not None
        return self._selected_source


def run_manual_fusion_wizard(
    clusters: Sequence[Cluster],
    source_priority: Sequence[Source],
    parent=None,
    on_widget_change=None,
) -> list[Any] | None:
    """Ask source per widget via dialog. Returns None if cancelled."""
    resolved: list[Any] = []
    total = len(clusters)
    priority = tuple(source_priority)

    for i, cluster in enumerate(clusters):
        if on_widget_change is not None:
            on_widget_change(i, clusters)
        default = default_source_for_cluster(cluster, priority)
        dialog = FusionWidgetPickDialog(i + 1, total, cluster, default, parent=parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            if on_widget_change is not None:
                on_widget_change(-1, [])
            return None
        resolved.append(dialog.selected_box())
        print(
            f"[INFO] Widget {i + 1}/{total} : {SOURCE_LABELS[dialog.selected_source()]} retenu.",
            flush=True,
        )

    if on_widget_change is not None:
        on_widget_change(-1, [])
    return resolved


def show_session_config_dialog(a11y_available: bool = True, parent=None) -> CombinedModeConfig | None:
    dialog = CombinedConfigDialog(a11y_available=a11y_available, parent=parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.config()


# ─────────────────────────────────────────────
# SAVE DIALOG
# ─────────────────────────────────────────────
SAVE_LAYER_LABELS: dict[str, str] = {
    "hover": "Hover (vert)",
    "yolo": "YOLO (orange)",
    "a11y": "Accessibilité (bleu)",
    "fused": "Fusion (blanc)",
}


@dataclass
class SaveSelection:
    sources: list[str] = field(default_factory=list)
    output_dir: str = "annotations"
    separate_files: bool = True


@dataclass
class SaveDialogResult:
    """action: save | discard | cancel (cancel keeps the current session)."""
    action: Literal["save", "discard", "cancel"] = "cancel"
    selection: SaveSelection | None = None
    selected_boxes: dict[str, list[Any]] = field(default_factory=dict)


def annotation_entry(item: Any, annotation_id: int, source: str) -> dict[str, Any]:
    """Build one JSON annotation; a11y dicts keep UIA metadata (no control_type_id)."""
    x, y, w, h = box_coords(item)
    entry: dict[str, Any] = {
        "id": annotation_id,
        "source": source,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
    }
    if isinstance(item, dict):
        entry["control_type"] = item.get("control_type", "")
        entry["class_name"] = item.get("class_name", "")
        entry["name"] = item.get("name", "")
    return entry


def build_save_payload(
    selected: dict[str, Sequence[Any]],
    image_width: int,
    image_height: int,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build the annotations JSON for the selected sources (order preserved)."""
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    annotations: list[dict[str, Any]] = []
    sources_used: list[str] = []
    for source, boxes in selected.items():
        if not boxes:
            continue
        sources_used.append(source)
        for box in boxes:
            annotations.append(annotation_entry(box, len(annotations), source))
    return {
        "timestamp": stamp,
        "sources": sources_used,
        "total_widgets": len(annotations),
        "image_width": image_width,
        "image_height": image_height,
        "annotations": annotations,
    }


class SaveConfigDialog(QDialog):
    """Choose what to save, where, and preview the JSON."""

    def __init__(
        self,
        available: dict[str, Sequence[Any]],
        image_width: int,
        image_height: int,
        default_dir: str = "annotations",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Enregistrer les annotations")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setMinimumHeight(540)

        self._available = {k: list(v) for k, v in available.items() if v}
        self._image_width = image_width
        self._image_height = image_height
        self._result = SaveDialogResult(action="cancel")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Sources disponibles à enregistrer :"))

        self._checks: dict[str, QCheckBox] = {}
        sources_box = QGroupBox("Contenu")
        sources_layout = QVBoxLayout(sources_box)
        for key in ("hover", "yolo", "a11y", "fused"):
            boxes = self._available.get(key)
            if not boxes:
                continue
            label = SAVE_LAYER_LABELS.get(key, key)
            extra = ""
            if key == "a11y":
                extra = " — JSON enrichi (control_type, class_name, name)"
            elif key in ("hover", "yolo"):
                extra = " — bbox uniquement"
            elif key == "fused":
                extra = " — résultat fusion (métadonnées a11y si conservées)"
            cb = QCheckBox(f"{label}  ({len(boxes)}){extra}")
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_preview)
            self._checks[key] = cb
            sources_layout.addWidget(cb)
        layout.addWidget(sources_box)

        if not self._checks:
            layout.addWidget(QLabel("Aucune détection à enregistrer."))

        self._separate_cb = QCheckBox(
            "Fichiers séparés par source (un JSON + image annotée par case cochée)"
        )
        self._separate_cb.setChecked(True)
        self._separate_cb.setEnabled(len(self._checks) > 1)
        self._separate_cb.stateChanged.connect(self._refresh_preview)
        layout.addWidget(self._separate_cb)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Dossier :"))
        self._path_edit = QLineEdit(os.path.abspath(default_dir))
        path_row.addWidget(self._path_edit, stretch=1)
        browse = QPushButton("Parcourir…")
        browse.clicked.connect(self._browse)
        path_row.addWidget(browse)
        layout.addLayout(path_row)

        layout.addWidget(QLabel("Aperçu JSON :"))
        self._preview = QPlainTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._preview, stretch=1)

        buttons = QDialogButtonBox()
        save_btn = buttons.addButton("Enregistrer", QDialogButtonBox.ButtonRole.AcceptRole)
        discard_btn = buttons.addButton(
            "Ne pas sauvegarder", QDialogButtonBox.ButtonRole.DestructiveRole
        )
        cancel_btn = buttons.addButton("Annuler", QDialogButtonBox.ButtonRole.RejectRole)
        save_btn.clicked.connect(self._accept_save)
        discard_btn.clicked.connect(self._accept_discard)
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

        self._refresh_preview()

    def _selected_map(self) -> dict[str, list[Any]]:
        return {
            key: self._available[key]
            for key, cb in self._checks.items()
            if cb.isChecked() and key in self._available
        }

    def _preview_text(self, selected: dict[str, list[Any]]) -> str:
        separate = self._separate_cb.isChecked() and len(selected) > 1
        if separate:
            parts: list[str] = [
                "// Mode fichiers séparés — un JSON par source :",
                "",
            ]
            for source, boxes in selected.items():
                payload = build_save_payload(
                    {source: boxes},
                    self._image_width,
                    self._image_height,
                    timestamp="<timestamp>",
                )
                payload["_file"] = f"annotations_{source}_<timestamp>.json"
                if len(payload["annotations"]) > 4:
                    short = dict(payload)
                    short["annotations"] = payload["annotations"][:4]
                    short["_preview_note"] = (
                        f"… +{len(payload['annotations']) - 4} annotation(s)"
                    )
                    payload = short
                parts.append(json.dumps(payload, indent=2, ensure_ascii=False))
                parts.append("")
            return "\n".join(parts).strip()

        payload = build_save_payload(
            selected, self._image_width, self._image_height, timestamp="<timestamp>"
        )
        payload["_file"] = "annotations_<timestamp>.json"
        if len(payload["annotations"]) > 8:
            short = dict(payload)
            short["annotations"] = payload["annotations"][:8]
            short["_preview_note"] = (
                f"… +{len(payload['annotations']) - 8} annotation(s) non affichée(s)"
            )
            payload = short
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def _refresh_preview(self) -> None:
        self._preview.setPlainText(self._preview_text(self._selected_map()))

    def _browse(self) -> None:
        start = self._path_edit.text().strip() or os.getcwd()
        chosen = QFileDialog.getExistingDirectory(self, "Dossier de sauvegarde", start)
        if chosen:
            self._path_edit.setText(chosen)

    def _accept_save(self) -> None:
        selected = self._selected_map()
        if not selected:
            return
        out = self._path_edit.text().strip() or "annotations"
        self._result = SaveDialogResult(
            action="save",
            selection=SaveSelection(
                sources=list(selected.keys()),
                output_dir=out,
                separate_files=self._separate_cb.isChecked() and len(selected) > 1,
            ),
            selected_boxes=selected,
        )
        self.accept()

    def _accept_discard(self) -> None:
        self._result = SaveDialogResult(action="discard")
        self.accept()

    def result_data(self) -> SaveDialogResult:
        return self._result


def show_save_config_dialog(
    available: dict[str, Sequence[Any]],
    image_width: int,
    image_height: int,
    default_dir: str = "annotations",
    parent=None,
) -> SaveDialogResult:
    dialog = SaveConfigDialog(
        available=available,
        image_width=image_width,
        image_height=image_height,
        default_dir=default_dir,
        parent=parent,
    )
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return SaveDialogResult(action="cancel")
    return dialog.result_data()


def cycle_view(view: View, backward: bool = False) -> View:
    """Review-phase views only (arrows)."""
    order: tuple[View, ...] = ("hover", "yolo", "a11y", "all", "fused")
    try:
        idx = order.index(view)
    except ValueError:
        idx = 0
    idx = (idx - 1) if backward else (idx + 1)
    return order[idx % len(order)]

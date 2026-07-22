# WidgetFusion Annotator

Application desktop d'annotation de widgets UI : hover diff, YOLO, accessibilité (Windows), fusion multi-sources.

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install .
```

## Lancement

```bash
python widgetfusion_annotator.py
```

ou :

```bash
widgetfusion-annotator
```

## Workflow

1. Choisir les méthodes au démarrage (dialogue de config)
2. Enchaîner les phases : hover → YOLO → accessibilité → revue
3. **← / →** pour changer de vue en revue
4. Raccourcis clavier :
   - **Entrée** : étape suivante / fusion
   - **M** : mode manuel
   - **S** : enregistrer
   - **Q** : quitter

## Fichiers

| Fichier | Description |
|---------|-------------|
| `widgetfusion_annotator.py` | Application (overlay Qt, export) |
| `fusion_mode.py` | Phases, fusion, dialogues |
| `accessibility_boxes.py` | UI Automation (Windows) |

## Sorties

Dossier de sauvegarde par défaut : **Bureau/annotations** (modifiable dans le dialogue d’enregistrement).

## Plateformes

- **Windows** : support complet
- **Linux / macOS** : expérimental

## Licence

MIT — voir `LICENSE`.

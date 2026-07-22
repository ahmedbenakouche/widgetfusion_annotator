# WidgetFusion Annotator

Application desktop d’annotation de widgets UI : hover diff, YOLO, accessibilité (Windows), fusion multi-sources.

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install .
```

Le modèle YOLO attendu par défaut : `yolo26n-1280.pt` (à placer à la racine du projet).

## Lancement

```bash
python widgetfusion_annotator.py
```

ou, après `pip install .` :

```bash
widgetfusion-annotator
```

## Workflow

1. Au démarrage, choisir les méthodes (hover / YOLO / accessibilité) et le mode hover (manuel ou autoscan).
2. Enchaîner les phases avec **Entrée** : hover → YOLO → accessibilité → revue.
3. En revue, **← / →** change la vue :
   - vert = hover
   - orange = YOLO
   - bleu = accessibilité
   - 3 superposées = lecture seule
   - blanc = fusion
4. Sur la vue « 3 superposées », **Entrée** ouvre la fusion.
5. **S** ouvre le dialogue d’enregistrement (choix des sources, chemin, aperçu JSON).
6. Fermer / Annuler le dialogue de méthodes au démarrage quitte l’application.

## Raccourcis

| Touche | Action |
|--------|--------|
| **Entrée** | Étape suivante / fusion (ou nouvelle session si idle) |
| **M** | Mode manuel (édition des bbox, curseur `+`) |
| **S** | Enregistrer (ou nouvelle session si idle) |
| **Q** | Quitter |
| **← / →** | Changer de vue en revue |

## Fusion

Matching entre sources si **IoU ≥ seuil** **ou** **inclusion souple ≥ seuil** (part de la plus petite bbox dans l’intersection — léger débordement toléré).

- Priorité par défaut : **accessibilité → hover → YOLO** (modifiable dans le dialogue).
- Les **ancres** de matching suivent cet ordre de priorité.
- Mode auto : géométrie de la source prioritaire ; métadonnées a11y conservées si présentes dans le groupe.
- Mode manuel : choix source widget par widget.
- Option « bbox isolées » : garder les détections sans match inter-sources.

## Sauvegarde

Dossier par défaut : **Bureau/annotations** (créé automatiquement).

Le dialogue permet de :

- cocher les sources à exporter (hover / YOLO / a11y / fusion)
- choisir le dossier
- prévisualiser le JSON
- optionnellement écrire **un JSON + image par source**
- **Enregistrer** / **Ne pas sauvegarder** (rouvre la config) / **Annuler** (reste en session)

Couleurs des bbox sur l’image annotée = couleurs de l’overlay (sans ID).

Exemple d’entrée a11y :

```json
{
  "id": 0,
  "source": "a11y",
  "bbox": {"x": 10, "y": 20, "w": 80, "h": 24},
  "control_type": "Button",
  "class_name": "Button",
  "name": "OK"
}
```

## Fichiers

| Fichier | Description |
|---------|-------------|
| `widgetfusion_annotator.py` | Overlay Qt, capture, phases, export |
| `fusion_mode.py` | Matching, fusion, dialogues |
| `accessibility_boxes.py` | UI Automation (Windows) ; stubs Linux/macOS |

## Plateformes

- **Windows** : support complet (dont accessibilité UIA)
- **Linux / macOS** : expérimental (pas d’a11y pour l’instant)

## Licence

MIT — voir `LICENSE`.

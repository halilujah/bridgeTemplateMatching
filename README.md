# Bridge Template Matching

Automated detection, localization, and change detection of bridge structural elements from 3D point clouds using template matching.

This repository accompanies the paper **"Automated Detection and Localization of Bridge Elements Using Template Matching with Point Clouds"** (included as a PDF in the project root). The goal is to take a scanned point cloud of a bridge, segment out individual structural elements by matching them against known CAD/mesh templates, and then quantify how much each element has moved or deformed between two surveys.

## How it works

The pipeline runs in two stages:

1. **Template matching & segmentation** ([`Code/tamplate_matching.py`](Code/tamplate_matching.py))
   - Samples a dense point cloud from a full bridge mesh (`models/model_final.ply`).
   - For each element template (pier, bent cap, bearing, sign structure, etc.), crops the point cloud to the template's axis-aligned bounding box.
   - Uses a KD-tree nearest-neighbor query against the template vertices to isolate the matching points.
   - Writes one segmented point cloud per element (`models/<name>_segmented.ply`).

2. **Change detection** ([`Code/change_detection.py`](Code/change_detection.py))
   - Aligns each segmented element to its corresponding "after change" mesh using ICP (Iterative Closest Point) registration.
   - Computes several deviation metrics per element:
     - **Position change** — translation magnitude from the ICP transform.
     - **Rotation change** — rotation magnitude (degrees) from the ICP transform.
     - **Centroid position change** — distance between element centroids.
     - **Hausdorff distance** — worst-case surface deviation between the two states.
     - **Point-to-mesh distance** — max and mean nearest-surface distance.
   - Flags an element as having "moved significantly" when its position change exceeds `0.05` or its rotation exceeds `5°`.

Linear measurements are reported in feet (the raw values are scaled by `3.2808399`).

## Bridge elements

The pipeline processes five named elements, each with its own template mesh:

| Name          | Element type     |
|---------------|------------------|
| `P1`          | Pile / pier      |
| `BRRight1`    | Bearing (right)  |
| `PierColumn5` | Pier column      |
| `SIG4`        | Sign structure   |
| `BentCap1`    | Bent cap         |

For each element, the [`models/`](models/) directory contains:

- `<name>.ply` — the template mesh used for matching.
- `<name>_segmented.ply` — the segmented point cloud (output of stage 1).
- `<name>_afterchange.ply` — the post-change mesh used as the change-detection reference.

`models/model_final.ply` is the full bridge mesh that the working point cloud is sampled from.

## Requirements

- Python 3.8+
- [Open3D](http://www.open3d.org/) — point cloud / mesh I/O, sampling, ICP registration, visualization
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/) — KD-tree, rotation math, Hausdorff distance

Install the dependencies with:

```bash
pip install open3d numpy scipy
```

## Usage

Run the scripts from the **repository root** (the model paths are relative, e.g. `models/...`):

```bash
# Stage 1 — segment elements out of the bridge point cloud.
# Opens an Open3D window showing the segmented elements and writes
# models/<name>_segmented.ply for each element.
python Code/tamplate_matching.py

# Stage 2 — compare each segmented element against its post-change
# mesh and print the deviation metrics to the console.
python Code/change_detection.py
```

Stage 2 reads the `*_segmented.ply` files produced by stage 1, so run them in order.

### Example output (stage 2)

```
Processing P1...
Position Change: 0.1234 ft
Rotation Change: 2.45 degrees
Centroid Position Change: 0.0876 ft
Hausdorff Distance: 0.5421 ft
Max Point-to-Mesh Distance: 0.3210 ft, Mean: 0.0450 ft
No significant movement detected.
```

## Project structure

```
.
├── Code/
│   ├── tamplate_matching.py    # Stage 1: template matching + segmentation
│   └── change_detection.py     # Stage 2: ICP alignment + change metrics
├── models/                     # Bridge mesh, element templates, segmented & after-change clouds
├── AUTOMATED DETECTION ... .pdf  # Accompanying paper
└── README.md
```

## Notes

- The deviation thresholds (`POSITION_THRESHOLD`, `ROTATION_THRESHOLD`) and the unit conversion are defined at the top of [`change_detection.py`](Code/change_detection.py#L69-L70); adjust them for your data and units.
- Visualization of per-point deviations (heat-map coloring) is included but commented out at the end of [`change_detection.py`](Code/change_detection.py#L96-L100) — uncomment it to inspect deformation visually.
- To run on your own data, replace the meshes in `models/` and update the `objectnames` list in both scripts.

# 3D-FRONT Preprocessing Pipeline

This document describes the preprocessing pipeline implemented in `data_loader/threed_front/understanding_code.py` for the 3D-FRONT dataset. The script is designed to process 3D scene data, extract and validate room layouts, walls, doors, windows, and furniture, and generate structured outputs for downstream tasks such as layout analysis and visualization.

---

## Overview

The script processes each scene in the 3D-FRONT dataset, performing the following key steps:

1. **Scene Loading**: Loads scene data, including geometry and metadata, using the `ThreedFront` class.
2. **Wall and Room Analysis**: Identifies and validates wall meshes, checks for customized or fragmented walls, and analyzes wall geometry for potential doors and windows.
3. **Furniture Validation**: Checks if furniture is properly enclosed by walls and detects furniture-furniture collisions.
4. **Script and Mesh Generation**: Generates a script describing the room layout and exports concatenated mesh files for visualization.
5. **Logging and Error Handling**: Logs all steps and errors, and records invalid or problematic scenes for review.

---

## Requirements

- Python 3.7+
- [trimesh](https://trimsh.org/)
- numpy
- scipy

Other dependencies may be required for the broader project (see your environment setup).

---

## Preprocessing Steps

### 1. Initialization
- Set up logging to both console and a timestamped log file in `preprocessing_logs/`.
- Define output directories for processed scenes and logs.
- Load the 3D-FRONT dataset and model metadata.

### 2. Scene Iteration
- For each scene in the dataset:
  - Skip if already processed or marked invalid.
  - Create an output directory for the scene.

### 3. Wall and Room Processing
- For each object in the scene:
  - Identify wall meshes (including standard and customized types).
  - For each wall (For now WallInner, Front and Back):
    - Detect 'broken faces' (holes) that may correspond to doors or windows.
    - Analyze face orientations to classify wall alignment (YZ, XZ, tilted, etc.).
    - Check for fragmented walls (disconnected geometry) and mark such scenes as invalid.
    - Calculate projected areas to estimate the size of holes and determine if they are likely doors or windows.
    - Generate mesh objects for detected doors/windows and add them to the output.
    - Record wall geometry in a script format.

### 4. Furniture Processing
- For each furniture item:
  - Load and transform the mesh to the scene coordinate system.
  - Validate that the furniture is enclosed by the room's walls.
  - Check for collisions between furniture items.
  - Record valid furniture in the output script.

### 5. Output Generation
- If the scene passes all checks:
  - Save a script file describing the room, walls, doors, windows, and furniture.
  - Export concatenated mesh files for walls, doors/windows, and furniture for visualization.
  - Record the scene as valid in a summary file.
- If the scene fails any check:
  - Record the reason in an invalid or unknown reason file for later review.

### 6. Error Handling and Logging
- All steps are logged with detailed messages for debugging and traceability.
- Errors are caught and logged, and problematic scenes are skipped and recorded.

---

## Output Structure

- `preprocessed-outputs/<scene_id>/`
  - `concatenated_mesh.obj`: All meshes for the scene.
  - `concatenated_walls_mesh.obj`: All wall meshes.
  - `concatenated_windows_mesh.obj`: All door/window meshes.
  - `concatenated_furniture_mesh.obj`: All furniture meshes.
  - `<scene_unique_id>.txt`: Script describing the room layout and contents.
  - `<scene_unique_id>_vis.txt`: Visualization script (may differ in formatting).
- `preprocessed-outputs/valid_scene_ids.txt`: List of successfully processed scenes.
- `preprocessed-outputs/invalid_scene_ids.txt`: List of scenes marked invalid, with reasons.
- `preprocessing_logs/preprocessing_log_<timestamp>.txt`: Detailed log of the preprocessing run.
---

## Error Handling

- **Fragmented Walls**: Scenes with disconnected wall geometry are skipped and logged as invalid.
- **Customized Walls**: Scenes with unsupported wall types are skipped.
- **Insufficient Walls**: Rooms with fewer than two walls are skipped.
- **Furniture Collisions**: Scenes with furniture outside the room or with overlapping furniture are marked invalid.
- **Unexpected Errors**: Any other errors are logged and the scene is recorded in `unknown_reason.txt`.

---

## Usage

1. Ensure all dependencies are installed and the dataset paths are correctly set in the script.
2. Run the script:
   ```bash
   python data_loader/threed_front/understanding_code.py
   ```
3. Monitor progress in the console and in the log file under `preprocessing_logs/`.
4. Review outputs in `preprocessed-outputs/` and logs for any errors or skipped scenes.

---

## Notes
- The script is designed to be robust and resume processing if interrupted, by checking for already processed or invalid scenes.
- All major steps and decisions are logged for transparency and debugging.
- The output script format is designed for downstream layout analysis and visualization tools.

---

## Added Logic
- Discard the furniture with `unknown_category`.
- Pre checking with Bounding Box Intersection, before vertices intersection for less computation.

--- 

### For visualization of sample script

    PYTHONPATH=./ python vis/visualize.py -l data_loader/scene_0000_00.txt
(based on what arguments you are passing, you might need to remove some)
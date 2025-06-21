import os
import json
import trimesh
import numpy as np
from itertools import combinations
from datetime import datetime
from data_loader.threed_front import threed_front  # Assuming threed_front.py is in this path

import trimesh
import numpy as np
from data_loader.layout.entity import Wall, Door, Window, Bbox
from data_loader.threed_front.threed_front import ThreedFront
from scipy.spatial.distance import pdist, squareform
import os
from collections import Counter
import json


output_dir = "door_analysis_results"  # New output directory for door analysis
os.makedirs(output_dir, exist_ok=True)

# Paths
path_to_3d_front_dataset_directory = "/mnt/sv-share/3DFRONT/3D-FRONT"
path_to_model_info = "/mnt/sv-share/3DFRONT/3D-FUTURE-model/model_info.json"
path_to_3d_future_dataset_directory = "/mnt/sv-share/3DFRONT/3D-FUTURE-model"

# Load dataset
d = ThreedFront.from_dataset_directory(
    path_to_3d_front_dataset_directory,
    path_to_model_info,
    path_to_3d_future_dataset_directory,
    path_to_room_masks_dir=None,
    path_to_bounds=None,
    filter_fn=lambda s: s
)

all_door_types_found = set()
scenes_with_all_door_types = []

total_scenes = len(d.scenes)
processed_count = 0
chunk_size = 500

for i, scene in enumerate(d.scenes):
    scene_door_types = set()
    for extra in scene.extras:
        if hasattr(extra, 'model_type') and extra.model_type in ["Door", "interiorDoor", "entryDoor"]:
            scene_door_types.add(extra.model_type)
            all_door_types_found.add(extra.model_type)

    processed_count += 1
    if processed_count % chunk_size == 0:
        print(f"Processed {processed_count}/{total_scenes} scenes for door types (pass 1).")

print(f"\nAll unique door types found across the dataset: {all_door_types_found}")

# Now, let's iterate again to find scenes containing ALL these door types
processed_count_second_pass = 0
for i, scene in enumerate(d.scenes):
    scene_door_types = set()
    for extra in scene.extras:
        if hasattr(extra, 'model_type') and extra.model_type in ["Door", "interiorDoor", "entryDoor"]:
            scene_door_types.add(extra.model_type)

    if all_door_types_found.issubset(scene_door_types) and all_door_types_found:
        scenes_with_all_door_types.append({
            "scene_id": scene.scene_id,
            "json_path": scene.json_path
        })
        print(f"Found scene with all door types: {scene.scene_id}, JSON Path: {scene.json_path}")

    processed_count_second_pass += 1
    if processed_count_second_pass % chunk_size == 0:
        print(f"Processed {processed_count_second_pass}/{total_scenes} scenes (pass 2) for scenes with all door types.")

print(f"\nScenes containing all found door types:")
for scene_info in scenes_with_all_door_types:
    print(f"  Scene ID: {scene_info['scene_id']}, JSON Path: {scene_info['json_path']}")

# Save the list of scenes to a JSON file
output_filename = "scenes_with_all_door_types.json"
output_filepath = os.path.join(output_dir, output_filename)
with open(output_filepath, "w") as f:
    json.dump(scenes_with_all_door_types, f, indent=2)

print(f"\nList of scenes containing all door types saved to '{output_filepath}'")
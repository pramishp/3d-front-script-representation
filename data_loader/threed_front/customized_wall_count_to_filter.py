import os
import json
from data_loader.threed_front.threed_front import ThreedFront

output_dir = "customized_wall_analysis"
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

scenes_with_customized_walls = []
scenes_without_customized_walls = []

for scene in d.scenes:
    json_filename = scene.json_path.split("/")[-1].replace(".json", "")
    scene_unique_id = f"{json_filename}__{scene.scene_id}"
    has_customized_wall = False
    for extra in scene.extras:
        if hasattr(extra, 'model_type') and extra.model_type in ["CustomizedFeatureWall", "ExtrusionCustomizedBackgroundWall"]:
            has_customized_wall = True
            break

    if has_customized_wall:
        scenes_with_customized_walls.append(scene_unique_id)
    else:
        scenes_without_customized_walls.append(scene_unique_id)

# Save the results to JSON files
with open(os.path.join(output_dir, "scenes_with_customized_walls.json"), "w") as f:
    json.dump(scenes_with_customized_walls, f, indent=2)

with open(os.path.join(output_dir, "scenes_without_customized_walls.json"), "w") as f:
    json.dump(scenes_without_customized_walls, f, indent=2)

# Print the counts
count_with_customized = len(scenes_with_customized_walls)
count_without_customized = len(scenes_without_customized_walls)

print(f"Analysis complete. Results saved in '{output_dir}'.")
print(f"Number of scenes with 'CustomizedFeatureWall' or 'ExtrusionCustomizedBackgroundWall': {count_with_customized}")
print(f"Number of scenes without 'CustomizedFeatureWall' or 'ExtrusionCustomizedBackgroundWall': {count_without_customized}")
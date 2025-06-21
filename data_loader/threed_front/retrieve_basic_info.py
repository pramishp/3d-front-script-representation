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


output_dir = "basic_query_results_large"  # Changed output directory for large dataset
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

print("starting")

print("starting again")

new=d.scenes[:3]
for i, scene in enumerate(new):
    type= scene.scene_type
    objects= scene.object_types
    print(f"type:{type}")
    print(f"objects:{objects}")



print("end")
scenes_with_unknown_category_ids = []
count_scenes_with_unknown_category = 0
scenes_with_no_doors_ids = []
count_scenes_with_no_doors = 0
all_discardable_scene_ids = set()
count_all_discardable_scenes = 0
count_quality_scenes = 0

room_counts_per_json_path = {}
max_wall_type_count = set()
max_door_type_count = set()
max_window_type_count = set()

max_wall_types_len = 0
max_door_types_len = 0
max_window_types_len = 0

total_scenes = len(d.scenes)
chunk_size = 500
processed_count = 0

# New variables to store the information you requested
scene_type_counts = Counter()
object_type_counts_per_room = []
all_object_types = set()




for i, scene in enumerate(d.scenes):
    json_filename = scene.json_path.split("/")[-1].replace(".json", "")
    scene_unique_id = f"{json_filename}__{scene.scene_id}"

    has_unknown_category = "unknown_category" in scene.object_types
    has_no_doors = True  # Assume no doors initially
    for extra in scene.extras:
        if hasattr(extra, 'model_type') and extra.model_type in ["Door", "interiorDoor", "entryDoor"]:
            has_no_doors = False
            break

    is_discardable = False
    if has_unknown_category or has_no_doors:
        is_discardable = True
        all_discardable_scene_ids.add(scene_unique_id)
        count_all_discardable_scenes += 1
        if has_unknown_category:
            scenes_with_unknown_category_ids.append(scene_unique_id)
            count_scenes_with_unknown_category += 1
        if has_no_doors:
            scenes_with_no_doors_ids.append(scene_unique_id)
            count_scenes_with_no_doors += 1
    else:
        count_quality_scenes += 1

    scene_info = {
        "scene_id": scene.scene_id,
        "json_path": scene.json_path,
        "scene_type": scene.scene_type,
        "object_types": list(scene.object_types),
        "total_door_count": 0 if has_no_doors else 1,
        "interior_door_count": 0,
        "entry_door_count": 0,
        "door_types": set(),
        "total_wall_count": 0,
        "wall_types": set(),
        "window_types": set(),
    }

    for extra in scene.extras:
        if hasattr(extra, 'model_type'):
            if extra.model_type in ["Door", "interiorDoor", "entryDoor"]:
                scene_info["total_door_count"] += 1
                scene_info["door_types"].add(extra.model_type)
                if extra.model_type == "interiorDoor":
                    scene_info["interior_door_count"] += 1
                elif extra.model_type == "entryDoor":
                    scene_info["entry_door_count"] += 1
            elif "Wall" in extra.model_type:
                scene_info["total_wall_count"] += 1
                scene_info["wall_types"].add(extra.model_type)
            elif extra.model_type == "Window":
                scene_info["window_types"].add(extra.model_type)

    # Count rooms per JSON path
    json_path = scene.json_path.split("/")[-1]
    room_counts_per_json_path[json_path] = room_counts_per_json_path.get(json_path, 0) + 1

    # Update max counts for wall, door, and window types
    if len(scene_info["wall_types"]) > max_wall_types_len:
        max_wall_types_len = len(scene_info["wall_types"])
        max_wall_type_count = set(scene_info["wall_types"])
    elif len(scene_info["wall_types"]) == max_wall_types_len and scene_info["wall_types"]:
        max_wall_type_count.update(scene_info["wall_types"])

    if len(scene_info["door_types"]) > max_door_types_len:
        max_door_types_len = len(scene_info["door_types"])
        max_door_type_count = set(scene_info["door_types"])
    elif len(scene_info["door_types"]) == max_door_types_len and scene_info["door_types"]:
        max_door_type_count.update(scene_info["door_types"])

    if len(scene_info["window_types"]) > max_window_types_len:
        max_window_types_len = len(scene_info["window_types"])
        max_window_type_count = set(scene_info["window_types"])
    elif len(scene_info["window_types"]) == max_window_types_len and scene_info["window_types"]:
        max_window_type_count.update(scene_info["window_types"])

    # New calculations:
    scene_type_counts[scene.scene_type] += 1  # Count scene types
    object_type_counts_per_room.append(len(scene.object_types))  # Count object types per room
    all_object_types.update(scene.object_types)  # Collect all unique object types

    processed_count += 1
    if processed_count % chunk_size == 0:
        print(f"Processed {processed_count}/{total_scenes} scenes.")
        # Append results to files
        with open(os.path.join(output_dir, "scenes_with_unknown_category_ids.json"), "a") as f:
            json.dump(scenes_with_unknown_category_ids, f)
            f.write("\n")

        with open(os.path.join(output_dir, "scenes_with_no_doors_ids.json"), "a") as f:
            json.dump(scenes_with_no_doors_ids, f)
            f.write("\n")

        with open(os.path.join(output_dir, "all_discardable_scene_ids.json"), "a") as f:
            json.dump(list(all_discardable_scene_ids), f)
            f.write("\n")

        # Reset lists for the next chunk
        scenes_with_unknown_category_ids = []
        scenes_with_no_doors_ids = []
        all_discardable_scene_ids = set()

# Write the remaining data and other summaries
print(f"Finished processing all {total_scenes} scenes. Writing final results.")

# Write the accumulated dictionary
with open(os.path.join(output_dir, "room_counts_per_json_path.json"), "w") as f:
    json.dump(room_counts_per_json_path, f, indent=2)

# Write the final counts
with open(os.path.join(output_dir, "count_scenes_with_unknown_category.txt"), "w") as f:
    f.write(str(count_scenes_with_unknown_category))

with open(os.path.join(output_dir, "count_scenes_with_no_doors.txt"), "w") as f:
    f.write(str(count_scenes_with_no_doors))

with open(os.path.join(output_dir, "count_all_discardable_scenes.txt"), "w") as f:
    f.write(str(len(all_discardable_scene_ids)))

with open(os.path.join(output_dir, "count_quality_scenes.txt"), "w") as f:
    f.write(str(count_quality_scenes))

with open(os.path.join(output_dir, "max_wall_types.json"), "w") as f:
    json.dump(list(max_wall_type_count), f, indent=2)

with open(os.path.join(output_dir, "max_door_types.json"), "w") as f:
    json.dump(list(max_door_type_count), f, indent=2)

with open(os.path.join(output_dir, "max_window_types.json"), "w") as f:
    json.dump(list(max_window_type_count), f, indent=2)

# Concatenate the chunked JSON files (optional)
def concatenate_json_files(input_pattern, output_file):
    all_data = []
    for filename in os.listdir(output_dir):
        if input_pattern in filename:
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            all_data.append(data)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from line: {line.strip()} in {filepath}")
    with open(output_file, 'w') as outfile:
        json.dump(list(set(all_data)) if "discardable" in input_pattern else all_data, outfile, indent=2)

concatenate_json_files("scenes_with_unknown_category_ids.json",
                          os.path.join(output_dir, "final_scenes_with_unknown_category_ids.json"))
concatenate_json_files("scenes_with_no_doors_ids.json",
                          os.path.join(output_dir, "final_scenes_with_no_doors_ids.json"))
concatenate_json_files("all_discardable_scene_ids.json",
                          os.path.join(output_dir, "final_all_discardable_scene_ids.json"))

# Clean up the chunked files (optional)
for filename in os.listdir(output_dir):
    if "scenes_with_unknown_category_ids.json" in filename and "final" not in filename:
        os.remove(os.path.join(output_dir, filename))
    if "scenes_with_no_doors_ids.json" in filename and "final" not in filename:
        os.remove(os.path.join(output_dir, filename))
    if "all_discardable_scene_ids.json" in filename and "final" not in filename:
        os.remove(os.path.join(output_dir, filename))

# Calculate and print the new information:
total_rooms = len(d.scenes)
average_object_types_per_room = sum(object_type_counts_per_room) / total_rooms if total_rooms else 0

# Write the new information to a file:
with open(os.path.join(output_dir, "scene_and_object_type_analysis.txt"), "w") as f:
    f.write("Scene Type Counts:\n")
    for scene_type, count in scene_type_counts.items():
        f.write(f"  {scene_type}: {count}\n")
    f.write("\n")
    f.write(f"Average Object Types per Room: {average_object_types_per_room:.2f}\n")
    f.write("\n")
    f.write("Total Distinct Object Types:\n")
    for object_type in all_object_types:
        f.write(f"  {object_type}\n")

print(f"\nResults for all {total_scenes} scenes saved to '{output_dir}'.")
print(f"Total scenes with 'unknown_category': {count_scenes_with_unknown_category}")
print(f"Total scenes with no doors: {count_scenes_with_no_doors}")
print(f"Total discardable scenes (union): {len(all_discardable_scene_ids)}")
print(f"Total quality scenes (neither 'unknown_category' nor no doors): {count_quality_scenes}")
print(f"Scene type counts, average object types per room, and total distinct object types are in {os.path.join(output_dir, 'scene_and_object_type_analysis.txt')}")
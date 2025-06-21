import os
import json
from data_loader.threed_front.threed_front import ThreedFront
import trimesh
import numpy as np  # For numerical operations


import sys




output_dir = "all_logs"
os.makedirs(output_dir, exist_ok=True)




# Paths
path_to_3d_front_dataset_directory = "/mnt/sv-share/3DFRONT/3D-FRONT"
#path_to_3d_front_dataset_directory = "/home/ajad/Desktop/codes/LISA"
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

fragmented_walls = []
non_fragmented_walls = []
scene_fragment_summary = {}

def point_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def share_vertices(face1, face2, tolerance=1e-5):
    for v1 in face1:
        for v2 in face2:
            if point_distance(v1, v2) < tolerance:
                return True
    return False

def is_fragmented(faces, tolerance=1e-5):
    face_graph = {i: set() for i in range(len(faces))}

    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            if share_vertices(faces[i].tolist(), faces[j].tolist(), tolerance):
                face_graph[i].add(j)
                face_graph[j].add(i)

    if len(faces) == 0:
        return False

    visited = set()
    queue = [0]
    visited.add(0)

    while queue:
        face_id = queue.pop(0)
        for neighbor in face_graph[face_id]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) != len(faces)

for scene in d.scenes:
    json_filename = scene.json_path.split("/")[-1].replace(".json", "")
    scene_unique_id = f"{json_filename}__{scene.scene_id}"

    for extra in scene.extras:
        if hasattr(extra, 'model_type') and extra.model_type in ["WallInner", "WallOuter", "Front", "Back"]:
            wall_id = extra.model_id if hasattr(extra, 'model_id') else "unknown_id"
            try:
                mesh = trimesh.Trimesh(vertices=extra.xyz, faces=extra.faces)

                broken_faces_indices = trimesh.repair.broken_faces(mesh)
                if len(broken_faces_indices) > 0:
                    broken_faces = mesh.faces[broken_faces_indices]
                    broken_vertices = mesh.vertices[broken_faces]
                    broken_faces_list = broken_vertices.tolist()

                    is_frag = is_fragmented(np.array(broken_faces_list), tolerance=0.01)
                    if is_frag:
                        fragmented_walls.append({"scene_id": scene_unique_id, "wall_id": wall_id})
                        scene_fragment_summary.setdefault(scene_unique_id, {"broken_wall_count": 0, "wall_ids": []})
                        scene_fragment_summary[scene_unique_id]["broken_wall_count"] += 1
                        scene_fragment_summary[scene_unique_id]["wall_ids"].append(wall_id)
                    else:
                        non_fragmented_walls.append({
                            "scene_id": scene_unique_id,
                            "wall_id": wall_id,
                            "fragmentation_reason": "broken faces exist but are connected"
                        })
                else:
                    non_fragmented_walls.append({
                        "scene_id": scene_unique_id,
                        "wall_id": wall_id,
                        "fragmentation_reason": "no broken faces"
                    })

            except Exception as e:
                print(f"Error processing mesh for wall in scene {scene_unique_id} (id: {wall_id}): {e}")


with open(os.path.join(output_dir, "scene_broken_wall_summary.json"), "w") as f:
    json.dump(scene_fragment_summary, f, indent=2)

# Print counts
count_fragmented = len(fragmented_walls)
count_non_fragmented = len(non_fragmented_walls)

print(f"\nAnalysis of fragmented and non-fragmented walls complete. Results saved in '{output_dir}'.")
print(f"Number of fragmented standard walls (based on broken face connectivity): {count_fragmented}")
print(f"Number of non-fragmented standard walls: {count_non_fragmented}")
print(f"Number of scenes with at least one broken wall: {len(scene_fragment_summary)}")

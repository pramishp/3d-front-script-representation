import trimesh
import numpy as np
import os
import logging


# =====================
# Utility Functions
# =====================
def is_point_in_bbox(point, bbox_min, bbox_max, tolerance=1e-6):
    """Checks if a point is within a bounding box with a small tolerance."""
    return all(bbox_min[i] - tolerance <= point[i] <= bbox_max[i] + tolerance for i in range(3))

def calculate_vertex_enclosure_score(mesh1, mesh2):
    """Calculates the percentage of vertices of mesh1 inside the bounding box of mesh2."""
    bbox_min2, bbox_max2 = mesh2.bounding_box.bounds
    enclosed_count = sum(is_point_in_bbox(vertex, bbox_min2, bbox_max2) for vertex in mesh1.vertices)
    return enclosed_count / len(mesh1.vertices) if mesh1.vertices.size > 0 else 0

def concatenate_and_export(meshes, output_path):
    """Concatenate a list of meshes and export as .obj if not empty."""
    if meshes:
        concatenated = trimesh.util.concatenate(meshes)
        concatenated.export(output_path)
        logging.info(f"Exported: {output_path}")
    else:
        logging.info(f"No meshes to export for {output_path}")

# =====================
# Scene Processing
# =====================
def process_scene(scene, transform, output_dir):
    """Extract walls, doors/windows, and furniture meshes and export as .obj files."""
    json_filename = scene.json_path.split("/")[-1].replace(".json", "")
    # print(json_filename)
    room_id = scene.scene_id
    # output_dir_room = os.path.join(output_dir, f"{json_filename}_{room_id}")
    output_dir_room = os.path.join(output_dir, room_id)
    os.makedirs(output_dir_room, exist_ok=True)

    wall_meshes = []
    door_window_meshes = []
    furniture_meshes = []
    complete_mesh = []

    for obj in scene.extras:
        # print(obj.model_type)
        if obj.xyz is not None and obj.faces is not None and len(obj.xyz) > 0 and len(obj.faces) > 0:
            mesh = trimesh.Trimesh(vertices=obj.xyz, faces=obj.faces)
            mesh.apply_transform(transform)
            if hasattr(obj, 'model_type') and any(phrase in obj.model_type.lower() for phrase in ["wall" "front", "back", "floor"]):
                wall_meshes.append(mesh)
            # If doors/windows are separate objects, add logic here
            elif hasattr(obj, 'model_type') and any(phrase in obj.model_type.lower() for phrase in ["door", "window"]):
                door_window_meshes.append(mesh)
            complete_mesh.append(mesh)

             
    for furniture in scene.bboxes:
        try:
            mesh = furniture.raw_model()
            vertices = furniture._transform(mesh.vertices)
            mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
            mesh.apply_transform(transform)
            furniture_meshes.append(mesh)
            complete_mesh.append(mesh)
        except Exception as e:
            logging.warning(f"[Furniture Error] {furniture.label}: {e}")
            continue

    concatenate_and_export(wall_meshes, os.path.join(output_dir_room, "walls.obj"))
    concatenate_and_export(door_window_meshes, os.path.join(output_dir_room, "doors_windows.obj"))
    concatenate_and_export(furniture_meshes, os.path.join(output_dir_room, "furniture.obj"))
    concatenate_and_export(complete_mesh, os.path.join(output_dir_room, "complete.obj"))

# =====================
# Main Workflow
# =====================
def main():
    import sys; sys.path.append("/home/ajad/Desktop/codes/LISA")
    from data_loader.layout.entity import Wall, Door, Window, Bbox
    from data_loader.threed_front.threed_front import ThreedFront

    """Main function to load the ThreedFront dataset and visualize each scene as .obj files."""
    logging.basicConfig(level=logging.INFO)

    root_dataset_dir = "/mnt/sv-share/3DFRONT/data"
    current_dir = "/home/ajad/Desktop/codes/LISA"
    
    path_to_3d_front_dataset_directory = os.path.join(current_dir, "3D-FRONT-test")
    path_to_model_info = os.path.join(root_dataset_dir, "3D-FUTURE-model/model_info.json")
    path_to_3d_future_dataset_directory = os.path.join(root_dataset_dir, "3D-FUTURE-model")
    output_dir = os.path.join(current_dir, "unpreprocessed-outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Z-up transform
    transform = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    # Load dataset
    d = ThreedFront.from_dataset_directory(
        path_to_3d_front_dataset_directory,
        path_to_model_info,
        path_to_3d_future_dataset_directory,
        path_to_room_masks_dir=None,
        path_to_bounds=None,
        filter_fn=lambda s: s
    )

    for i, scene in enumerate(d.scenes):
        try:
            logging.info(f"Processing scene {i+1}/{len(d.scenes)}: {scene.scene_id}")
            process_scene(scene, transform, output_dir)
        except Exception as e:
            logging.error(f"Failed to process scene {scene.scene_id}: {e}")

if __name__ == "__main__":
    main()





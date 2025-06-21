import json
import math
import numpy as np  # Using numpy for potential vector/matrix operations
import os  # Used for constructing file paths
import glob  # Potentially useful for handling path patterns like 'objects/*' if needed later

# --- Configuration ---
# Path to the main dataset configuration file
BASE_PATH = "/mnt/sv-share/habitat-sim/data/scene_datasets/hssd-hab/"
DATASET_CONFIG_PATH = os.path.join(BASE_PATH, 'hssd-hab.scene_dataset_config.json')  # ADJUST PATH AS NEEDED

# Path to the specific scene file we want to convert (can be made dynamic later)
# This path might also be derived/found using dataset_config if listing scenes
SCENE_FILE_PATH = os.path.join(BASE_PATH, 'scenes/108736884_177263634.scene_instance.json')  # ADJUST PATH RELATIVE TO DATASET ROOT

# Output file path
OUTPUT_FILE_PATH = 'scene_converted_output.txt'


# --- External Library Placeholder ---
# You MUST implement this function using a library like trimesh or pygltflib
# (See previous response for example structure)
def parse_glb_for_dimensions(glb_filepath):
    """
    Placeholder: Loads a GLB file and returns its base bounding box dimensions.
    Requires implementation using a 3D geometry library (e.g., trimesh).
    """
    print(f"Error: Placeholder function `parse_glb_for_dimensions` called for {glb_filepath}.")
    print("       You must implement this using a library like trimesh to read GLB dimensions.")
    # Must return dimensions [X, Y, Z] in object's local frame
    return None


# --- Coordinate Transformation Functions ---

def transform_point_yup_to_zup(point_yup):
    """Converts a point from Habitat's Y-up to Right-Handed Z-up."""
    if point_yup is None or len(point_yup) != 3:
        return [0.0, 0.0, 0.0]
    # Transformation: (x, y, z)_yup -> (x, -z, y)_zup
    return [point_yup[0], -point_yup[2], point_yup[1]]


def quaternion_to_z_yaw(quat_xyzw):
    """Converts a Habitat quaternion (X, Y, Z, W) to a yaw angle around Z-up."""
    if quat_xyzw is None or len(quat_xyzw) != 4:
        return 0.0
    x, y, z, w = quat_xyzw
    yaw_y = math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + x * x))
    return yaw_y  # Radians


# --- Data Loading Functions ---

def load_json(filepath):
    """Loads a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None


# --- Path Helper Function ---
def get_base_dir_from_config(config, key, file_extension=".json"):
    """Extracts the first directory path for a given key from dataset config."""
    try:
        # Navigate through the config structure
        paths_dict = config.get(key, {}).get("paths", {})
        path_list = paths_dict.get(file_extension, [])
        if path_list:
            # Handle potential wildcards, just take the directory part
            base_path = path_list[0].split('*')[0]
            # Ensure it's treated as a directory (remove trailing slashes if any)
            return base_path.rstrip('/')
        else:
            print(f"Warning: No '{file_extension}' path found for key '{key}' in dataset config.")
            return None
    except Exception as e:
        print(f"Error parsing path for key '{key}': {e}")
        return None


# --- Processing Functions ---

def get_object_details(template_name, base_objects_dir):
    """
    Loads the object's JSON definition file and extracts GLB path and dimensions.
    Assumes object JSONs are directly within the base_objects_dir.
    """
    if not base_objects_dir:
        print("Error: Base object directory not provided.")
        return None, None

    # Construct the expected path to the object's JSON file
    object_json_path = os.path.join(base_objects_dir, f"{template_name}.json")
    object_data = load_json(object_json_path)

    if not object_data:
        # Try looking in subdirs like 'decomposed' if specified in config?
        # For now, we assume it's directly in the first path listed.
        print(f"Warning: Could not load object definition: {object_json_path}")
        return None, None

    render_asset_name = object_data.get('render_asset')
    if not render_asset_name:
        print(f"Warning: No 'render_asset' found in {object_json_path}")
        return None, None

    # Construct full path to GLB file relative to the objects directory
    glb_filepath = os.path.join(base_objects_dir, render_asset_name)

    # Get dimensions by parsing the GLB file (using the placeholder)
    base_dimensions = parse_glb_for_dimensions(glb_filepath)  # Returns None if placeholder hit
    semantic_id = object_data.get('semantic_id')

    return base_dimensions, semantic_id


# --- Main Conversion Logic ---

def convert_scene(scene_data, dataset_config):
    """Converts loaded scene data to the target format using paths from dataset_config."""
    output_lines = []

    # --- Determine Base Paths from dataset_config ---
    objects_base_dir = get_base_dir_from_config(dataset_config, "objects")
    stages_base_dir = get_base_dir_from_config(dataset_config, "stages")
    # Semantics path is more complex, derived later

    if not objects_base_dir:
        print("Error: Could not determine objects directory from dataset config.")
        # Decide how to handle this - maybe exit or use a default?
        return "[Error: Objects path missing in config]"
    if not stages_base_dir:
        print("Warning: Could not determine stages directory from dataset config.")
        # Stage processing will likely fail

    # --- 1. Process Stage Geometry (Walls, Doors, Windows) ---
    print("--- Stage Processing ---")
    stage_template_rel_path = scene_data.get('stage_instance', {}).get('template_name')
    if stage_template_rel_path and stages_base_dir:
        # Construct full path: dataset_root (implied) / stages_base_dir / template_name.json
        # Assuming template_name is like "stages/1234" and stages_base_dir is "stages"
        # We need to be careful not to double the directory name.
        # Let's assume template_name is just the ID "108736884_177263634"
        # If template_name includes the dir (like "stages/id"), adjust logic.

        # Robust way: remove base dir prefix if present in template path
        if stage_template_rel_path.startswith(stages_base_dir + '/'):
            stage_filename = stage_template_rel_path[len(stages_base_dir) + 1:] + ".json"
        else:
            stage_filename = stage_template_rel_path + ".json"  # Assuming it's just the ID

        stage_file_path = os.path.join(stages_base_dir, stage_filename)
        print(f"Attempting to parse stage file (logic not implemented): {stage_file_path}")

        # --- ADD STAGE PARSING LOGIC HERE ---
        # stage_definition = load_json(stage_file_path)
        # if stage_definition:
        #     Parse vertices, faces, etc.
        #     Apply transforms
        #     Format Wall(), Door(), Window() lines
        #     output_lines.extend(parsed_stage_elements)
        # else:
        #     print(f"Warning: Could not load stage file {stage_file_path}")
        output_lines.append("# Stage geometry parsing needed here")  # Placeholder
    else:
        print("Warning: Stage template name or stages base directory missing.")
        output_lines.append("# Stage geometry skipped (missing info)")

    # --- 2. Process Object Instances ---
    print("--- Object Processing ---")
    if scene_data and 'object_instances' in scene_data:
        for i, obj in enumerate(scene_data['object_instances']):
            template_name = obj.get('template_name')
            if not template_name:
                print(f"Warning: Skipping object instance {i} due to missing 'template_name'.")
                continue

            # Get object base dimensions and semantic ID
            # Pass the dynamically found objects_base_dir
            base_dims_xyz, semantic_id = get_object_details(template_name, objects_base_dir)

            if base_dims_xyz is None:
                print(f"Warning: Could not get dimensions for object {template_name}. Using placeholders [1,1,1].")
                base_dims_xyz = [1.0, 1.0, 1.0]

            pos_yup = obj.get('translation')
            pos_zup = transform_point_yup_to_zup(pos_yup)
            rot_quat_xyzw = obj.get('rotation')
            yaw_z = quaternion_to_z_yaw(rot_quat_xyzw)
            scale = obj.get('non_uniform_scale', [1.0, 1.0, 1.0])

            # Apply scale and map dimensions
            scaled_x = base_dims_xyz[0] * scale[0]
            scaled_y = base_dims_xyz[1] * scale[1]
            scaled_z = base_dims_xyz[2] * scale[2]
            dim_l, dim_w, dim_h = scaled_x, scaled_z, scaled_y

            object_label = f"object_{semantic_id}" if semantic_id else template_name
            bbox_line = f"bbox_{i}=Bbox({object_label},{pos_zup[0]},{pos_zup[1]},{pos_zup[2]},{yaw_z},{dim_l},{dim_w},{dim_h})"
            output_lines.append(bbox_line)

    # --- 3. Process Semantic Regions (Optional Load) ---
    print("--- Semantics Processing (Optional) ---")
    # Find the semantic map key from the scene file
    semantic_map_key = scene_data.get('default_attributes', {}).get('semantic_scene_instance')
    if semantic_map_key:
        # Look up the relative path in the dataset config
        semantic_rel_path = dataset_config.get('semantic_scene_descriptor_instances', {}).get(semantic_map_key)
        if semantic_rel_path:
            # Construct full path (assuming relative to dataset root)
            # Note: semantic_rel_path might already include the base dir like "semantics/..."
            semantic_file_path = semantic_rel_path  # Adjust with os.path.join if it's just a filename
            print(f"Semantic file path identified: {semantic_file_path}")
            # semantics_data = load_json(semantic_file_path)
            # if semantics_data:
            #     Process regions if needed for the output format
            #     print("Semantic data loaded (processing not implemented in output).")
            output_lines.append(f"# Semantic file: {semantic_file_path}")
        else:
            print(f"Warning: Semantic map key '{semantic_map_key}' not found in dataset config.")
            output_lines.append(f"# Semantic file mapping missing for key: {semantic_map_key}")
    else:
        print("Warning: No 'semantic_scene_instance' key found in scene attributes.")
        output_lines.append("# Semantic file key missing in scene data")

    return "\n".join(output_lines)


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Loading dataset configuration from: {DATASET_CONFIG_PATH}")
    dataset_config = load_json(DATASET_CONFIG_PATH)

    if not dataset_config:
        print("Error: Failed to load dataset configuration. Aborting.")
        exit()

    # Assume SCENE_FILE_PATH is relative to the dataset root (where dataset config is)
    print(f"Loading scene data from: {SCENE_FILE_PATH}")
    scene_data = load_json(SCENE_FILE_PATH)

    if scene_data:
        print("Converting scene using dataset config for paths...")
        converted_data = convert_scene(scene_data, dataset_config)

        print(f"Saving converted data to: {OUTPUT_FILE_PATH}")
        try:
            with open(OUTPUT_FILE_PATH, 'w') as f:
                f.write(converted_data)
            print("Conversion complete.")
            print("Reminder: Implement stage geometry parsing and GLB dimension reading.")
        except IOError:
            print(f"Error: Could not write to file {OUTPUT_FILE_PATH}")
    else:
        print(f"Scene data could not be loaded from {SCENE_FILE_PATH}. Conversion aborted.")
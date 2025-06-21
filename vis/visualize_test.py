import argparse
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import sys; sys.path.append("/home/ajad/Desktop/codes/LISA")

from data_loader.layout.layout import Layout

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Layout Visualization with rerun")

    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        required=True,
        help="Path to the layout txt file",
    )

    parser.add_argument(
        "-output",
        "--output_path",
        type=str,
        required=False,
        help="Path to the output directory",
    )
    

    rr.script_add_args(parser)
    args = parser.parse_args()

    import os
    import trimesh

    with open(args.layout, "r") as f:
        layout_content = f.read()

    # parse layout_content
    layout = Layout(layout_content)
    layout.undiscretize_and_unnormalize()
    floor_plan = layout.to_boxes()

    # --- Save as OBJ ---
    # Each box is a 3D box, so we can export all as a single mesh
    meshes = []
    for box in floor_plan:
        center = np.array(box["center"]).reshape(3)
        scale = np.array(box["scale"]).reshape(3)
        rotation = np.array(box["rotation"]).reshape(3, 3)
        # Create a unit cube centered at origin
        unit_box = trimesh.creation.box(extents=scale)
        # Move to center
        unit_box.apply_translation(center)
        # Apply rotation
        unit_box.apply_transform(np.vstack([np.hstack([rotation, np.zeros((3,1))]), [0,0,0,1]]))
        meshes.append(unit_box)

    if meshes:
        scene = trimesh.util.concatenate(meshes)
        output_dir = args.output_path if args.output_path else "."
        os.makedirs(output_dir, exist_ok=True)
        obj_path = os.path.join(output_dir, "layout_boxes.obj")
        scene.export(obj_path)
        print(f"Saved OBJ to {obj_path}")

    # --- Save as Image (screenshot) ---
    # We'll use trimesh's built-in scene viewer to render and save an image
    # try:
    #     png_path = os.path.join(output_dir, "layout_boxes.png")
    #     # Trimesh's save_image returns bytes, so we write to file
    #     png_bytes = scene.scene().save_image(resolution=(1024, 768), visible=True)
    #     with open(png_path, "wb") as f:
    #         f.write(png_bytes)
    #     print(f"Saved image to {png_path}")
    # except Exception as e:
    #     print(f"Could not save image: {e}")

    # Optionally, you can still run the rerun visualization if desired
    # (Uncomment below if you want to keep the interactive visualization)
    # blueprint = rrb.Blueprint(
    #     rrb.Spatial3DView(name="3D", origin="/world", background=[255, 255, 255]),
    #     collapse_panels=True,
    # )
    # rr.script_setup(args, "rerun_spatiallm", default_blueprint=blueprint)
    # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    # num_entities = len(floor_plan)
    # seconds = 0.5
    # for ti in range(num_entities + 1):
    #     sub_floor_plan = floor_plan[:ti]
    #     rr.set_time_seconds("time_sec", ti * seconds)
    #     for box in sub_floor_plan:
    #         uid = box["id"]
    #         group = box["class"]
    #         label = box["label"]
    #         rr.log(
    #             f"world/pred/{group}/{uid}",
    #             rr.Boxes3D(
    #                 centers=box["center"],
    #                 half_sizes=0.5 * box["scale"],
    #                 labels=label,
    #             ),
    #             rr.InstancePoses3D(mat3x3=box["rotation"]),
    #             static=False,
    #         )
    # rr.script_teardown(args)


# Example usage
# python visualize_saugat.py -l /home/ajad/Desktop/codes/LISA/VISUALIZE_SCRIPTS/1da9156e-927b-42e0-a242-e13513e9fcad__MasterBedroom-59217.txt -output ./vis
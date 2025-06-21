import argparse
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import sys; sys.path.append("/home/ajad/Desktop/codes/LISA")

from dataset_toolkits.layout.layout import Layout

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Layout Visualization with rerun")

    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        required=True,
        help="Path to the layout txt file",
    )

    rr.script_add_args(parser)
    args = parser.parse_args()

    with open(args.layout, "r") as f:
        layout_content = f.read()

    # parse layout_content
    layout = Layout(layout_content)
    # layout.undiscretize_and_unnormalize()
    floor_plan = layout.to_boxes()

    # ReRun visualization
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(name="3D", origin="/world", background=[255, 255, 255]),
        collapse_panels=True,
    )
    rr.script_setup(args, "rerun_spatiallm", default_blueprint=blueprint)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    num_entities = len(floor_plan)
    seconds = 0.5
    for ti in range(num_entities + 1):
        sub_floor_plan = floor_plan[:ti]

        rr.set_time_seconds("time_sec", ti * seconds)
        for box in sub_floor_plan:
            uid = box["id"]
            group = box["class"]
            label = box["label"]

            rr.log(
                f"world/pred/{group}/{uid}",
                rr.Boxes3D(
                    centers=box["center"],
                    half_sizes=0.5 * box["scale"],
                    labels=label,
                ),
                rr.InstancePoses3D(mat3x3=box["rotation"]),
                static=False,
            )
    rr.script_teardown(args)

# Example usage
# python visualize_bbox.py -l /home/ajad/Desktop/codes/LISA/data_loader/scene0000_00.txt
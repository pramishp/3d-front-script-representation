import argparse
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh

from data_loader.layout.layout import Layout



# def load_mesh(mesh_path):
#     mesh=trimesh.load(mesh_path, force="mesh")
#     return mesh

# def get_mesh_data(mesh):
#     vertices= mesh.vertices
#     faces= mesh.faces
#     return vertices, faces

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Layout+ mesh trying hehe Visualization with rerun")

    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        required=True,
        help="Path to the layout txt file",
    )

    # parser.add_argument(
    #     "-m",
    #     "--mesh",
    #     type=str,
    #     required=False,
    #     help="Path to the mesh file",
    # )

    # parser.add_argument(
    #     "-w",
    #     "--windows_mesh",
    #     type=str,
    #     required=False,
    #     help="Path to the window mesh file",
    # )
    # parser.add_argument(
    #     "-wa",
    #     "--walls_mesh",
    #     type=str,
    #     required=False,
    #     help="Path to the walls mesh file",
    # )

    rr.script_add_args(parser)
    args = parser.parse_args()


    # mesh = load_mesh(args.mesh)
    # vertices, faces= get_mesh_data(mesh)

    # windows_mesh = load_mesh(args.windows_mesh)
    # windows_vertices, windows_faces= get_mesh_data(windows_mesh)

    # walls_mesh = load_mesh(args.walls_mesh)
    # walls_vertices, walls_faces= get_mesh_data(walls_mesh)

    with open(args.layout, "r") as f:
        layout_content = f.read()


    # parse layout_content
    layout = Layout(layout_content)
    layout.undiscretize_and_unnormalize()
    floor_plan = layout.to_boxes()
  	
  


    # ReRun visualization
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(name="3D", origin="/world", background=[255, 255, 255]),
        collapse_panels=True,
    )
    rr.script_setup(args, "rerun_spatiallm", default_blueprint=blueprint)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # rr.log("world/scene_mesh", rr.Mesh3D(vertex_positions= vertices, triangle_indices= faces,vertex_colors = np.tile([255,255,0], (vertices.shape[0],1))), static= True,)

    # rr.log("world/windows_mesh", rr.Mesh3D(vertex_positions= windows_vertices, triangle_indices= windows_faces,vertex_colors = np.tile([135,206,235], (vertices.shape[0],1))), static= True,)

    # rr.log("world/walls_mesh", rr.Mesh3D(vertex_positions= walls_vertices, triangle_indices= walls_faces,vertex_colors = np.tile([210, 105, 30], (vertices.shape[0],1))), static= True,)


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

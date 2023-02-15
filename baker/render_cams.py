import numpy as np
import mathutils
import bpy
import os


def render_all_cam_views(render_dir):
    scene = bpy.context.scene
    for ob in scene.objects:
        if ob.type == "CAMERA":
            print(ob)
            bpy.context.scene.camera = ob
            # Renders to a directory where each image is titled "{camera_name}.png"
            bpy.context.scene.render.filepath = os.path.join(render_dir, f"{ob.name}")
            bpy.ops.render.render(write_still=True)

    return {"FINISHED"}


res = render_all_cam_views("C:\\Users\\hello\\Desktop\\baker\\render")
print(res)
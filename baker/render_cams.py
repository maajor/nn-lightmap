import numpy as np
import mathutils
import bpy
import os

RESOLUTION=128

def setup_resolution(resolution):
    for scene in bpy.data.scenes:
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        scene.render.resolution_percentage = 100

def setup_for_basecolor():
    for scene in bpy.data.scenes:
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '16'
        scene.render.image_settings.color_management = 'FOLLOW_SCENE'
        scene.display_settings.display_device = 'sRGB'
        scene.view_settings.view_transform = 'Standard'
        scene.sequencer_colorspace_settings.name = 'sRGB'

def setup_for_pn():
    for scene in bpy.data.scenes:
        scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '16'
        scene.render.image_settings.color_management = 'FOLLOW_SCENE'
        scene.display_settings.display_device = 'None'
        scene.view_settings.view_transform = 'Standard'
        scene.sequencer_colorspace_settings.name = 'Raw'
        for layer in scene.view_layers:
            layer.use_pass_combined = False
            layer.use_pass_position = True
            layer.use_pass_normal = True


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

import json


def dump_camera_position(out_dir):
    scene = bpy.context.scene
    pos = {}
    for ob in scene.objects:
        if ob.type == "CAMERA":
            pos[ob.name] = [ob.location.x, ob.location.y, ob.location.z]
            
    print(pos)
    with open(os.path.join(out_dir, "cam_pos.json"), 'w') as f:
        json.dump(pos, f)


# only cyclys has position pass
bpy.context.scene.render.engine = 'CYCLES'

# render transparent background for pixel dropout
bpy.context.scene.render.film_transparent = True

setup_resolution(RESOLUTION)
setup_for_basecolor()
res = render_all_cam_views(bpy.path.abspath('//render/render'))
setup_for_pn()
res = render_all_cam_views(bpy.path.abspath('//render/pn'))
dump_camera_position(bpy.path.abspath('//render/'))
print(res)
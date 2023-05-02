import bpy
import os

LIGHTMAP_RESOLUTION = 512

bpy.data.images.new(name='baker5', width=LIGHTMAP_RESOLUTION, height=LIGHTMAP_RESOLUTION, alpha=False, float_buffer=True, )
image = bpy.data.images['baker5']
image.file_format = 'OPEN_EXR'

scene = bpy.context.scene
for ob in scene.objects:
    if ob.type == "MESH":
        mat = ob.active_material
        # Enable 'Use nodes':
        mat.use_nodes = True
        nodes = mat.node_tree.nodes

        # Add a diffuse shader and set its location:    
        node = nodes.new('ShaderNodeTexImage')
        node.location = (100,100)
        node.image = image
        print(ob.active_material)
            #bpy.ops.node.add_node(type='ShaderNodeTexImage')
        ob.select_set(True)

bpy.context.scene.render.image_settings.color_mode = 'RGB'
bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True, properties=True)
bpy.context.scene.cycles.bake_type = 'NORMAL'
bpy.context.scene.render.bake.normal_space = 'OBJECT'
bpy.context.scene.render.bake.margin = 4

bpy.ops.object.bake(type='NORMAL')

image.update()
image.save_render(filepath=os.path.join(bpy.path.abspath('//render/'), 'normal.exr'))

bpy.context.scene.cycles.bake_type = 'POSITION'
bpy.ops.object.bake(type='POSITION')

image.update()
image.save_render(filepath=os.path.join(bpy.path.abspath('//render/'), 'position.exr'))
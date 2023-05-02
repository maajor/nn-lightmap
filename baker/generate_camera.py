'''
This script generates a camera for each vertex of a sphere.
'''
import bpy
import mathutils


collection = bpy.data.collections.new('Cameras')
bpy.context.scene.collection.children.link(collection)


# https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
def point_at(obj, target, roll=0):
    """
    Rotate obj to look at target

    :arg obj: the object to be rotated. Usually the camera
    :arg target: the location (3-tuple or Vector) to be looked at
    :arg roll: The angle of rotation about the axis from obj to target in radians. 

    Based on: https://blender.stackexchange.com/a/5220/12947 (ideasman42)      
    """
    if not isinstance(target, mathutils.Vector):
        target = mathutils.Vector(target)
    loc = obj.location
    # direction points from the object to the target
    direction = target - loc
    tracker, rotator = (('-Z', 'Y'),'Z') if obj.type=='CAMERA' else (('X', 'Z'),'Y') #because new cameras points down(-Z), usually meshes point (-Y)
    quat = direction.to_track_quat(*tracker)
    
    # /usr/share/blender/scripts/addons/add_advanced_objects_menu/arrange_on_curve.py
    quat = quat.to_matrix().to_4x4()
    rollMatrix = mathutils.Matrix.Rotation(roll, 4, rotator)

    # remember the current location, since assigning to obj.matrix_world changes it
    loc = loc.to_tuple()
    #obj.matrix_world = quat * rollMatrix
    # in blender 2.8 and above @ is used to multiply matrices
    # using * still works but results in unexpected behaviour!
    obj.matrix_world = quat @ rollMatrix
    obj.location = loc


'''bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=2.0)
object = bpy.data.objects['Icosphere']
subd = object.modifiers.new("subd", "SUBSURF")
subd.levels = 1
subd.render_levels = 1
bpy.ops.object.modifier_apply(modifier="subd")'''

bpy.ops.mesh.primitive_cube_add(size=4.0)
object = bpy.context.active_object
object.location = (0,0,0)
subd = object.modifiers.new("subd", "SUBSURF")
subd.levels = 2
subd.render_levels = 2
bpy.ops.object.modifier_apply(modifier="subd")

vertices = object.data.vertices
for vert in vertices:
    cam = bpy.data.cameras.new("Camera")
    cam.lens = 40
    cam_obj = bpy.data.objects.new("Camera", cam)
    cam_obj.location  = vert.co
    point_at(cam_obj, (0.0, 0.0, 0.0))
    collection.objects.link(cam_obj)

bpy.data.objects.remove(object, do_unlink=True)
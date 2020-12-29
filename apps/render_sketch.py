# coding: utf-8
# author: pinakinathc

import os
import glob
import random
import argparse
import numpy as np
import bpy
import bpy_extras
import numpy
import math
import mathutils
from mathutils import Matrix
from mathutils import Vector

def update_camera(camera, degrees, focus_point=mathutils.Vector((0.0, 13.0, 0.0)), distance=10.0):
    radians = math.radians(degrees)
    eul = mathutils.Euler((math.radians(90.0), math.radians(0.0), radians), 'XYZ')
    camera.rotation_euler = eul
    camera.location = mathutils.Vector((0.0, -distance, 13))
    camera.location = mathutils.Vector((distance*math.sin(radians), -distance*math.cos(radians), 12))


def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    print ('resolution: ', resolution_x_in_px, resolution_y_in_px)
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: 
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K


def get_3x4_RT_matrix_from_blender(cam):
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    T_world2bcam = -1*R_world2bcam @ location

    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Created render of 2D sketch from 3D')
    parser.add_argument('--input_dir', type=str, default='/vol/research/sketchcaption/extras/adobe-dataset/shirt_dataset_rest/*/shirt_mesh_r.obj',
            help='Enter input dir to raw dataset')
    parser.add_argument('--output_dir', type=str, default='../training_data', help='Enter output dir to rendered dataset')
    opt = parser.parse_args()

    objects_shirt_list = glob.glob(opt.input_dir)
    output_dir = opt.output_dir
    angle_step = 1

    for shirt_idx, shirt_data_path in enumerate(objects_shirt_list):
        objs = [ob for ob in bpy.context.scene.objects if ob.type in ('MESH')]
        bpy.ops.object.delete({'selected_objects': objs})
        bpy.ops.import_scene.obj(filepath=shirt_data_path)

        # Setup env and center object
        bpy.ops.transform.translate(value=(0,0,0))
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.camera.data.type = 'ORTHO'
        bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[0].default_value = (1,1,1,1)

        folder_name = os.path.split(shirt_data_path)[0]
        folder_name = os.path.split(folder_name)[-1]

        obj_path = os.path.join(output_dir, 'GEO', 'OBJ', folder_name)
        render_path = os.path.join(output_dir, 'RENDER', folder_name)
        param_path = os.path.join(output_dir, 'PARAM', folder_name)
        mask_path = os.path.join(output_dir, 'MASK', folder_name)

        os.makedirs(obj_path, exist_ok=True)
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(param_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)

        # copy obj file
        cmd = 'cp %s %s' % (shirt_data_path, obj_path)
        print(cmd)
        os.system(cmd)

        for degree in range(0, 360, angle_step):
            cam = bpy.data.objects['Camera']
            bpy.context.scene.render.resolution_x = 512
            bpy.context.scene.render.resolution_y = 512
            # Add randomness to angle
            update_camera(cam, degree+0.2*np.pi*(random.random()-0.5))
            calib, _, _ = get_3x4_P_matrix_from_blender(cam)

            #############################################
            ''' Render Sketch '''
            tree = bpy.context.scene.node_tree
            if tree is not None:
                links = tree.links
                for node in tree.nodes:
                    tree.nodes.remove(node)
                render_node = tree.nodes.new(type='CompositorNodeRLayers')
                composite_node = tree.nodes.new(type='CompositorNodeComposite')
                links.new(render_node.outputs['Image'], composite_node.inputs['Image'])

            # bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[0].default_value = (1,1,1,1)
            val = bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[0].default_value
            bpy.context.scene.render.use_freestyle = True
            bpy.context.scene.view_layers['View Layer'].use_solid = False
            bpy.context.scene.view_layers['View Layer'].use_ao = False
            bpy.context.scene.view_layers['View Layer'].use_volumes = False
            bpy.context.scene.view_layers['View Layer'].use_strand = False

            bpy.context.scene.render.filepath=os.path.join(render_path, str(degree)+'_0_00')
            bpy.ops.render.render(write_still=True)
            np.save(os.path.join(param_path, str(degree)+'_0_00.npy'), calib)

            ##############################################
            ''' Render Mask '''
            bpy.context.scene.render.use_freestyle = False
            bpy.context.scene.view_layers['View Layer'].use_solid = True
            bpy.context.scene.view_layers['View Layer'].use_ao = True
            bpy.context.scene.view_layers['View Layer'].use_volumes = True
            bpy.context.scene.view_layers['View Layer'].use_strand = True

            bpy.context.scene.view_layers['View Layer'].use_pass_object_index = True
            bpy.context.view_layer.objects.active = bpy.data.objects[2]
            bpy.context.object.pass_index = 1
            bpy.context.scene.use_nodes = True
            tree = bpy.context.scene.node_tree

            for node in tree.nodes:
                tree.nodes.remove(node)

            render_node = tree.nodes.new(type='CompositorNodeRLayers')
            id_node = tree.nodes.new(type='CompositorNodeIDMask')
            composite_node = tree.nodes.new(type='CompositorNodeComposite')

            id_node.index = 1
            links = tree.links
            link1 = links.new(render_node.outputs['IndexOB'], id_node.inputs['ID value'])
            link2 = links.new(id_node.outputs['Alpha'], composite_node.inputs['Image'])
            bpy.context.scene.render.filepath='mask'

            bpy.context.scene.render.filepath=os.path.join(mask_path, 
                str(degree)+'_0_00')
            bpy.ops.render.render(write_still=True)

from skimage import measure
import numpy as np
import torch
from torch.nn import functional as F
from .sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure
import tqdm
import trimesh


def ndf_reconstruction(net, cuda, calib_tensor, pos_emb, resolution, b_min, b_max,
                        num_steps=500, filter_val=0.5, num_samples=30000, transform=None):
    ''' Experimental where sample points are random.
    Complexity goes very high when calculating more than 50*50*50 points.
    Calculating faces is also difficult
    
    Generate Dense Point Cloud from NDF and then a mesh.
    Calculates the NDF for a single garment at a time. '''

    for param in net.parameters():
        param.requires_grad = False
    
    resolution = 50
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)

    def batch_eval(points):
        samples_cpu = np.zeros((3, 0))

        points = np.expand_dims(points, axis=0) # 1 x 3 x num_points
        points = np.repeat(points, net.num_views, axis=0) # num_views x 3 x num_points
        points = torch.tensor(points).to(device=cuda).float()
        points.requires_grad = True # num_views X 3 X num_points

        num_points = int(points.shape[-1]*0.6)

        i = 0
        while samples_cpu.shape[1] < num_points:
            for j in range(num_steps):
                net.zero_grad()
                points.requires_grad = True
                net.query(points, calib_tensor, pos_emb)
                pred = torch.clamp(net.get_preds()[0][0], max=0.5) # num_points x 1

                pred.sum().backward(retain_graph=True)

                gradient = points.grad.detach()
                points = points.detach()
                pred = pred.detach()
                points = points - F.normalize(gradient, dim=2) * pred.reshape(1, -1)

                points = points.detach()
                points = torch.tensor(points.cpu().detach().numpy()).to(device=cuda).float()

            if i != 0:
                samples_cpu = np.hstack((samples_cpu, points[:, :, pred < filter_val][0].detach().cpu().numpy()))

            points = points[:, :, pred < 1.3]
            points += (0.1/3) * torch.randn(points.shape).to(device=cuda) # 3 sigma rule
            points = points.detach()
            points.requires_grad = True

            i += 1
        return samples_cpu

    coords = np.reshape(coords, [3, -1])
    ndf_point_cloud = np.zeros((3, 0))

    num_batches = max(coords.shape[1] // num_samples, 1)
    for i in tqdm.tqdm(range(num_batches)):
        x = coords[:, i * num_samples:i * num_samples + num_samples]
        batch_pc = batch_eval(x)
        ndf_point_cloud = np.hstack((ndf_point_cloud, batch_pc))

    np.savetxt('faster_test.ply',
        ndf_point_cloud.T,
        fmt='%.6f %.6f %.6f',
        comments='',
        header=(
          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nend_header').format(
          ndf_point_cloud.shape[1])
        )
    input ('check')


def ndf_template_reconstruction(net, cuda, calib_tensor, pos_emb, resolution, b_min, b_max,
                        num_steps=200, filter_val=0.5, num_samples=30000, transform=None):
    ''' Generate Dense Point Cloud from NDF and then a mesh.
    Calculates the NDF for a single garment at a time. '''

    for param in net.parameters():
        param.requires_grad = False

    def batch_eval(points):
        points = np.expand_dims(points, axis=0) # 1 x 3 x num_points
        points = np.repeat(points, net.num_views, axis=0) # num_views x 3 x num_points
        points = torch.tensor(points).to(device=cuda).float()
        points.requires_grad = True # num_views X 3 X num_points

        for j in range(num_steps):
            net.zero_grad()
            points.requires_grad = True
            net.query(points, calib_tensor, pos_emb)
            pred = torch.clamp(net.get_preds()[0][0], max=0.5) # num_points x 1
            pred.sum().backward(retain_graph=True)

            gradient = points.grad.detach()
            points = points.detach()
            pred = pred.detach()
            points = points - F.normalize(gradient, dim=2) * pred.reshape(1, -1)

            points = points.detach()
            points = torch.tensor(points.cpu().detach().numpy()).to(device=cuda).float()

        return points[0].detach().cpu().numpy()

    # Choosing a template
    template = trimesh.load('/vol/research/NOBACKUP/CVSSP/scratch_4weeks/pinakiR/extra/training_data/GEO/OBJ/C0IWNQRKACKY/shirt_mesh_r_tmp.obj')
    coords = np.array(template.vertices.T)
    ndf_point_cloud = batch_eval(coords).T
    return ndf_point_cloud, template.faces


def visualise_NDF(ndf):
    ''' Visualise NDF '''
    assert ndf.shape == (256, 256, 256), 'NDF shape not equal to 256x256x256'
    import matplotlib.pyplot as plt

    for z in range(0, 256, 25):
        xy = ndf[:,::-1,z].T
        plt.imshow(xy)
        plt.savefig(str(z)+'.png')
        print ('saved ', z)


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()

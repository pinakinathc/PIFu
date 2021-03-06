from torch.utils.data import Dataset
import numpy as np
import os
import glob
import random
from svg.path import parse_path
from svg.path.path import Line
from xml.dom import minidom
from bresenham import bresenham
import scipy
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging
import math
from apps.render_data import make_rotate

log = logging.getLogger('trimesh')
log.setLevel(40)

np.random.seed(0)

def load_trimesh(root_dir):
    folders = os.listdir(root_dir)
    # folders = ['T7UNJRFVTMZV', 'L9UVRMFHQWVK']
    meshs = {}
    for i, f in enumerate(folders):
        sub_name = f
        # meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, '%s_100k.obj' % sub_name))
        meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, 'shirt_mesh_r_tmp_watertight.obj'))

    return meshs

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


def svg2img(render_path, is_train=False, p_mask=None):
    '''Reads SVG image and converts to rendered partial sketch'''
    doc = minidom.parse(render_path)
    height = int(doc.getElementsByTagName('svg')[0].getAttribute('height'))
    width = int(doc.getElementsByTagName('svg')[0].getAttribute('width'))
    raster_image = np.zeros((height, width), dtype=np.float32)
    path_strings = [path.getAttribute('d') for path
                in doc.getElementsByTagName('path')]
    doc.unlink()

    Len = len(path_strings)
    mask_len = int(Len*p_mask)
    start_idx = np.random.randint(0, Len-mask_len)
    path_strings = path_strings[:start_idx] + path_strings[start_idx+mask_len:]

    for path_string in path_strings:
        path = parse_path(path_string)
        for e in path:
            if isinstance(e, Line):
                x0 = round(e.start.real)
                y0 = round(e.start.imag)
                x1 = round(e.end.real)
                y1 = round(e.end.imag)
                cordList = list(bresenham(x0, y0, x1, y1))
                for cord in cordList:
                    raster_image[cord[1], cord[0]] = 255.0
    raster_image = 255.0 - scipy.ndimage.binary_dilation(raster_image) * 255.0
    return Image.fromarray(raster_image).convert('RGB')


class TrainDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.root = self.opt.dataroot
        self.SVG = os.path.join(self.root, 'SVG')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.UV_MASK = os.path.join(self.root, 'UV_MASK')
        self.UV_NORMAL = os.path.join(self.root, 'UV_NORMAL')
        self.UV_RENDER = os.path.join(self.root, 'UV_RENDER')
        self.UV_POS = os.path.join(self.root, 'UV_POS')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')

        self.B_MIN = np.array([-128, -28, -128])
        self.B_MAX = np.array([128, 228, 128])

        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize

        self.num_views = self.opt.num_views

        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color

        self.yaw_list = list(range(0,360,1))
        self.pitch_list = [0]
        self.subjects = self.get_subjects()

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

        self.mesh_dic = load_trimesh(self.OBJ)

    def get_subjects(self):
        all_subjects = os.listdir(self.SVG)
        # all_subjects = ['T7UNJRFVTMZV', 'L9UVRMFHQWVK']
        # var_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str)
        var_subjects = np.loadtxt('val.txt', dtype=str)
        if len(var_subjects) == 0:
            return all_subjects

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        else:
            return sorted(list(var_subjects))

    def __len__(self):
        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)

    def get_render(self, subject, num_views, yid=0, pid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''
        pitch = self.pitch_list[pid]

        # The ids are an even distribution of num_views around view_id
        view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
                    for offset in range(num_views)]
        
        if random_sample and self.is_train:
            local_state = np.random.RandomState()
            view_ids = local_state.choice(self.yaw_list, num_views, replace=False)
        #     view_ids = np.random.choice(self.yaw_list, num_views, replace=False)

        # if not self.is_train:
        #     view_ids = [0, 0, 0]

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []

        for idx, vid in enumerate(view_ids):
            if idx <= 1 and False:
                print ('getting random')
                # tmp_subject = np.random.choice(self.subjects, 1)[0]
                tmp_subject = 'L9UVRMFHQWVK'
            else:
                tmp_subject = subject
            vid = (vid+180) % 360
            # param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            render_path = os.path.join(self.SVG, tmp_subject, '%d_%d_%06d.svg' % (vid, pitch, 1))
            mask_path = os.path.join(self.MASK, tmp_subject, '%d_%d_%02d.png' % (vid, pitch, 0))

            # pixel unit / world unit
            ortho_ratio = 0.4 * (512 / self.load_size) # ortho_ratio = param.item().get('ortho_ratio')
            
            # world unit / model unit
            vertices = self.mesh_dic[tmp_subject].vertices
            vmin = vertices.min(0)
            vmax = vertices.max(0)
            # vmin = np.min(vertices)
            # vmax = np.max(vertices)
            # up_axis = 1 if (vmax-vmin).argmax() == 1 else 2
            up_axis = 1
            
            vmed = np.median(vertices, 0)
            vmed[up_axis] = 0.5*(vmax[up_axis]+vmin[up_axis])
            scale = 0.9 / max(vmax - vmin)

            # camera center world coordinate
            center = vmed

            # model rotation
            R = np.matmul(make_rotate(math.radians(pitch), 0, 0), make_rotate(0, math.radians(vid), 0))
            if up_axis == 2:
                R = np.matmul(R, make_rotate(math.radians(90),0,0))

            translate = -np.matmul(R, center).reshape(3, 1)
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            # Match camera space to image pixel space
            scale_intrinsic = np.identity(4)

            scale_intrinsic[0, 0] = scale
            scale_intrinsic[1, 1] = -scale
            scale_intrinsic[2, 2] = scale

            # Match image pixel space to image uv space
            uv_intrinsic = np.identity(4)
            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            mask = Image.open(mask_path).convert('RGBA').split()[-1].convert('L')
            render = svg2img(render_path, self.is_train, p_mask=0.5)
            # render = Image.open(render_path).convert('RGBA').split()[-1].convert('RGB')

            if self.is_train:
                # Pad images
                pad_size = int(0.1 * self.load_size)
                render = ImageOps.expand(render, pad_size, fill=0)
                mask = ImageOps.expand(mask, pad_size, fill=0)

                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random flip
                if self.opt.random_flip and np.random.rand() > 0.5:
                    scale_intrinsic[0, 0] *= -1
                    render = transforms.RandomHorizontalFlip(p=1.0)(render)
                    mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

                # random scale
                if self.opt.random_scale:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)
                    mask = mask.resize((w, h), Image.NEAREST)
                    scale_intrinsic *= rand_scale
                    scale_intrinsic[3, 3] = 1

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = random.randint(-int(round((w - tw) / 10.)),
                                        int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)),
                                        int(round((h - th) / 10.)))
                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dx / float(self.opt.loadSize // 2)
                trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                render = render.crop((x1, y1, x1 + tw, y1 + th))
                mask = mask.crop((x1, y1, x1 + tw, y1 + th))

                render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)

            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()

            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()
            mask_list.append(mask)

            render = self.to_tensor(render)
            ''' We ignore the mask and use only sketch to create 3D models'''
            # render = mask.expand_as(render) * render 

            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)

        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            # 'mask': torch.stack(mask_list, dim=0)
        }

    def select_sampling_method(self, subject):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)
        mesh = self.mesh_dic[subject]
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout)
        # sample_points = surface_points + np.random.normal(scale=self.opt.sigma, size=surface_points.shape)
        sample_points = surface_points + np.random.normal(scale=self.opt.sigma, size=surface_points.shape)

        # add random points within image space
        # length = self.B_MAX - self.B_MIN
        # random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN

        B_MAX = sample_points.max(0)
        B_MIN = sample_points.min(0)
        length = (B_MAX - B_MIN)
        # delta = length*1.0 //3
        # B_MIN =  B_MIN - delta
        # B_MAX = B_MAX + delta
        # length = B_MAX - B_MIN
        random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + B_MIN
        sample_points = np.concatenate([sample_points, random_points], 0)
        np.random.shuffle(sample_points)
        # if not self.is_train:
        #     samples_points = np.random.rand(self.num_sample_inout*2, 3) * length + B_MIN

        inside = mesh.contains(sample_points)
        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        nin = inside_points.shape[0]
        inside_points = inside_points[
                        :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
        outside_points = outside_points[
                         :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
                                                                                               :(self.num_sample_inout - nin)]

        samples = np.concatenate([inside_points, outside_points], 0).T
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)

        # save_samples_truncted_prob('%s.ply'%subject, samples.T, labels.T)
        # input('check')
        # exit()

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()
        
        del mesh

        return {
            'samples': samples,
            'labels': labels,
            'b_min': B_MIN,
            'b_max': B_MAX,
        }


    def get_color_sampling(self, subject, yid, pid=0):
        yaw = self.yaw_list[yid]
        pitch = self.pitch_list[pid]
        uv_render_path = os.path.join(self.UV_RENDER, subject, '%d_%d_%02d.jpg' % (yaw, pitch, 0))
        uv_mask_path = os.path.join(self.UV_MASK, subject, '%02d.png' % (0))
        uv_pos_path = os.path.join(self.UV_POS, subject, '%02d.exr' % (0))
        uv_normal_path = os.path.join(self.UV_NORMAL, subject, '%02d.png' % (0))

        # Segmentation mask for the uv render.
        # [H, W] bool
        uv_mask = cv2.imread(uv_mask_path)
        uv_mask = uv_mask[:, :, 0] != 0
        # UV render. each pixel is the color of the point.
        # [H, W, 3] 0 ~ 1 float
        uv_render = cv2.imread(uv_render_path)
        uv_render = cv2.cvtColor(uv_render, cv2.COLOR_BGR2RGB) / 255.0

        # Normal render. each pixel is the surface normal of the point.
        # [H, W, 3] -1 ~ 1 float
        uv_normal = cv2.imread(uv_normal_path)
        uv_normal = cv2.cvtColor(uv_normal, cv2.COLOR_BGR2RGB) / 255.0
        uv_normal = 2.0 * uv_normal - 1.0
        # Position render. each pixel is the xyz coordinates of the point
        uv_pos = cv2.imread(uv_pos_path, 2 | 4)[:, :, ::-1]

        ### In these few lines we flattern the masks, positions, and normals
        uv_mask = uv_mask.reshape((-1))
        uv_pos = uv_pos.reshape((-1, 3))
        uv_render = uv_render.reshape((-1, 3))
        uv_normal = uv_normal.reshape((-1, 3))

        surface_points = uv_pos[uv_mask]
        surface_colors = uv_render[uv_mask]
        surface_normal = uv_normal[uv_mask]

        if self.num_sample_color:
            sample_list = random.sample(range(0, surface_points.shape[0] - 1), self.num_sample_color)
            surface_points = surface_points[sample_list].T
            surface_colors = surface_colors[sample_list].T
            surface_normal = surface_normal[sample_list].T

        # Samples are around the true surface with an offset
        normal = torch.Tensor(surface_normal).float()
        samples = torch.Tensor(surface_points).float() \
                  + torch.normal(mean=torch.zeros((1, normal.size(1))), std=self.opt.sigma).expand_as(normal) * normal

        # Normalized to [-1, 1]
        rgbs_color = 2.0 * torch.Tensor(surface_colors).float() - 1.0

        return {
            'color_samples': samples,
            'rgbs': rgbs_color
        }

    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        # try:
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw_list)
        pid = tmp // len(self.yaw_list)

        # name of the subject 'rp_xxxx_xxx'
        subject = self.subjects[sid]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
            'sid': sid,
            'yid': yid,
            'pid': pid,
            # 'b_min': self.B_MIN,
            # 'b_max': self.B_MAX,
        }
        render_data = self.get_render(subject, num_views=self.num_views, yid=yid, pid=pid,
                                        random_sample=self.opt.random_multiview)
        res.update(render_data)

        if self.opt.num_sample_inout:
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)
        
        # img = np.uint8((np.transpose(render_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
        # rot = render_data['calib'][0,:3, :3]
        # trans = render_data['calib'][0,:3, 3:4]
        # pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # [3, N]
        # pts = 0.5 * (pts.numpy().T + 1.0) * render_data['img'].size(2)
        # for p in pts:
        #     img = cv2.circle(img, (p[0], p[1]), 2, (0,255,0), -1)
        # cv2.imshow('test', img)
        # cv2.waitKey(1)

        if self.num_sample_color:
            color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
            res.update(color_data)
        return res
        # except Exception as e:
        #     print(e)
        #     return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)

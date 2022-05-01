from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import cv2
from util.SplineInterpolation import SplineInterpolation

resolution = 512
bias = 0


def save_point(point_tensor, save_path):
    data_numpy = point_tensor[0].cpu().float().numpy()
    data_numpy = data_numpy.T
    fout = open(save_path, 'w')
    for i in range(data_numpy.shape[0]):
        fout.write(str(int((data_numpy[i,0]+1)/2*resolution)) + ' ' + str(int((data_numpy[i,1]+1)/2*resolution)) + '\n')
    fout.close()
    

def tensor2im(image_tensor, imtype=np.uint8):
    data_numpy = image_tensor[0].cpu().float().numpy()
    data_numpy = data_numpy.T
    img = np.ones((resolution, resolution, 3), dtype=imtype) * 255
    for i in range(data_numpy.shape[0]):
        cv2.circle(img, tuple([int((data_numpy[i,0]+1)/2*resolution)+bias, int((data_numpy[i,1]+1)/2*resolution)+bias]), 3, (0,0,255), -1)
    return img.astype(imtype)

def get_im_from_tensor(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def img_warp(image_tensor, pts_ori, pts_tar, imtype=np.uint8):
    pts = pts_ori
    vals = pts_tar - pts_ori
    pts[:, 0] = pts[:, 0] * resolution
    pts[:, 1] = pts[:, 1] * resolution
    pts = pts.cpu()
    vals = vals.cpu()
    image_tensor_cpu = image_tensor.cpu()
    dsi = SplineInterpolation((resolution, resolution))
    flow_shift = dsi(pts, vals)
    coord_x = torch.arange(-1, 1, 2/255).unsqueeze(0).expand((resolution, resolution))
    coord_y = torch.arange(-1, 1, 2/255).unsqueeze(1).expand((resolution, resolution))
    coord = torch.cat((coord_x.unsqueeze(2), coord_y.unsqueeze(2)), 2)
    flow_grid = coord - flow_shift
    flow_grid = flow_grid.unsqueeze(0)
    image_tensor_cpu = F.grid_sample(image_tensor_cpu, flow_grid, padding_mode='border')
    image_numpy = image_tensor_cpu[0].data.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
        self.parser.add_argument('--numPoints', type=int, default=63, help='number of points to be considered')
        self.parser.add_argument('--fineDim', type=int, default=2, help='dimension of point coordinates')
        self.parser.add_argument('--input_nc', type=int, default=3, help='num of input channel')
        self.parser.add_argument('--output_nc', type=int, default=3, help='num of output channel')
        self.parser.add_argument('--height', type=int, default=256, help='height of input image')
        self.parser.add_argument('--width', type=int, default=256, help='width of input image')
        self.parser.add_argument('--fineSize', type=int, default=304, help='width of cropped image')
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='new_experiment_name_new_new_new', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--load_name', type=str, default=None, help='name of the previous experiment(stages). It decides where to restore samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='point', help='chooses how datasets are loaded. [unaligned | aligned | single | point]')
        self.parser.add_argument('--model', type=str, default='cycle_gan',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=2, type=int)
        self.parser.add_argument('--checkpoints_dir', type=str, default='new', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--with_content_D', action='store_true')
        self.parser.add_argument('--concat', action='store_true')
        self.parser.add_argument('--lms', action='store_true')
        self.parser.add_argument('--stage1', action='store_true')
        self.parser.add_argument('--stage2', action='store_true')
        self.parser.add_argument('--classify', action='store_true')
        self.parser.add_argument('--DimClass', type=int, default=25,  help='class number')
        self.parser.add_argument('--omegas',metavar='float', type=float, nargs='*', default=[1.0,1.0,1.0], help='weights for loss_gen_adv,loss_gen_recon_c_a,classify_loss')
        self.parser.add_argument('--C_paths', type=str, default=None)
        self.parser.add_argument('--share_EC_s1_s2', action='store_true')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
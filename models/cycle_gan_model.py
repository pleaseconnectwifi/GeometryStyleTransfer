import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from .DifferentiableSplineInterpolation import DifferentiableSplineInterpolation
from . import networks

import joblib
import sys
from .utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler



class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def transform(self, mat):
        mat[:, 0] = (mat[:, 0] / 256 - 0.5) * 2
        mat[:, 1] = (mat[:, 1] / 256 - 0.5) * 2
        return mat.reshape((1, -1))

    def initialize(self, opt):
        self.with_content_D = opt.with_content_D
        self.concat = opt.concat
        self.lms = opt.lms
        self.classify = opt.classify
        self.omegas = opt.omegas
        self.load_translated_A = opt.C_paths is not None
        self.share_EC_s1_s2 = opt.share_EC_s1_s2
        try:
            self.get_txt = opt.get_txt
        except:
            self.get_txt = False
        if self.get_txt:
            self.result_dir = os.path.join(opt.results_dir, opt.name)
            self.result_dir = os.path.join(self.result_dir, '%s_%s' % (opt.phase, opt.which_epoch))
            self.result_dir = os.path.join(self.result_dir, 'images')
            
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        self.input_A = self.Tensor(nb, opt.fineDim, opt.numPoints)
        self.input_B = self.Tensor(nb, opt.fineDim, opt.numPoints)
        self.input_A_img = self.Tensor(nb, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B_img = self.Tensor(nb, opt.input_nc, opt.fineSize, opt.fineSize)
        if self.load_translated_A:
            self.input_C = self.Tensor(nb, opt.fineDim, opt.numPoints)
            self.input_C_img = self.Tensor(nb, opt.input_nc, opt.fineSize, opt.fineSize)
        self.fineSize = opt.fineSize
        
        
        self.celeba_pca = joblib.load('merged_pca_68_32.pkl')
        self.celeba_pca_mean = Variable(torch.from_numpy(self.celeba_pca.mean_), requires_grad=False).float().cuda()
        self.celeba_mean_points = np.loadtxt('celeba_mean.txt')
        self.celeba_mean_points = self.transform(self.celeba_mean_points)
        self.celeba_mean_points = Variable(torch.from_numpy(self.celeba_mean_points), requires_grad=False).float().cuda()
        
        
        self.art_pca = joblib.load('merged_pca_68_32.pkl')
        self.art_pca_mean = Variable(torch.from_numpy(self.art_pca.mean_), requires_grad=False).float().cuda()
        self.art_mean_points = np.loadtxt('art_mean.txt')
        self.art_mean_points = self.transform(self.art_mean_points)
        self.art_mean_points = Variable(torch.from_numpy(self.art_mean_points), requires_grad=False).float().cuda()
        celeba_pca_sample_num = 37700
        art_pca_sample_num = 37700
        celeba_pca_mixing_to = np.sqrt(celeba_pca_sample_num) * self.celeba_pca.components_ / self.celeba_pca.singular_values_[:, np.newaxis]
        celeba_pca_mixing_return = self.celeba_pca.singular_values_[:, np.newaxis] * self.celeba_pca.components_ / np.sqrt(celeba_pca_sample_num)
        self.celeba_pca_mixing_to = Variable(torch.from_numpy(celeba_pca_mixing_to), requires_grad=False).float().cuda()
        self.celeba_pca_mixing_return = Variable(torch.from_numpy(celeba_pca_mixing_return), requires_grad=False).float().cuda()
        art_pca_mixing_to = np.sqrt(art_pca_sample_num) * self.art_pca.components_ / self.art_pca.singular_values_[:, np.newaxis]
        art_pca_mixing_return = self.art_pca.singular_values_[:, np.newaxis] * self.art_pca.components_ / np.sqrt(art_pca_sample_num)
        self.art_pca_mixing_to = Variable(torch.from_numpy(art_pca_mixing_to), requires_grad=False).float().cuda()
        self.art_pca_mixing_return = Variable(torch.from_numpy(art_pca_mixing_return), requires_grad=False).float().cuda()
        
        
        self.ratio = np.loadtxt('ratio.txt')
        self.ratio = self.ratio.reshape((1, -1))
        self.ratio = Variable(torch.from_numpy(self.ratio), requires_grad=False).float().cuda()
        self.stage1 = opt.stage1
        self.stage2 = opt.stage2

        
        
        

        
        self.netG_A = networks.define_G(num_points=opt.numPoints, k=opt.fineDim, use_dropout=False, init_type='kaiming', gpu_ids=self.gpu_ids,concat=self.concat,no_enc_style=(opt.stage1 or opt.stage2)).cuda()
        
        
        
        
        if not opt.stage1: 
            self.netG_B = networks.define_G(num_points=opt.numPoints, k=opt.fineDim, use_dropout=False, init_type='kaiming', gpu_ids=self.gpu_ids,concat=self.concat).cuda()
        
        
        
        self.dsi = DifferentiableSplineInterpolation((opt.fineSize, opt.fineSize))

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if not opt.stage1 and not opt.stage2:
                self.netD_A = networks.define_D(num_points=opt.numPoints, k=opt.fineDim, init_type='normal', gpu_ids=self.gpu_ids).cuda()
                self.netD_B = networks.define_D(num_points=opt.numPoints, k=opt.fineDim, init_type='normal', gpu_ids=self.gpu_ids).cuda()
                if self.with_content_D:
                    self.netD_c = networks.define_D(num_points=opt.numPoints, k=opt.fineDim, init_type='normal',
                                                    gpu_ids=self.gpu_ids).cuda()
            elif opt.stage2:
                
                self.netD = networks.define_D(num_points=32, k=opt.fineDim, init_type='normal', gpu_ids=self.gpu_ids).cuda() 
                if self.classify:
                    
                    self.netC = networks.define_C(num_points=32, k=opt.DimClass, init_type='normal', gpu_ids=self.gpu_ids).cuda()
                if self.with_content_D:
                    self.netD_c = networks.define_D(num_points=opt.numPoints, k=opt.fineDim, init_type='normal', gpu_ids=self.gpu_ids).cuda()
        
        
        
        
        
        

        if not self.isTrain or opt.continue_train:
            print('------load model-------')
            which_epoch = opt.which_epoch
            if opt.stage2: 
                self.load_network(self.netG_A, 'G_A', which_epoch, load_prevstage=True)
            else:
                self.load_network(self.netG_A, 'G_A', which_epoch)
            
            if not opt.stage1:
                try:
                    self.load_network(self.netG_B, 'G_B', which_epoch)
                    if self.isTrain:
                        if opt.stage2:
                            self.load_network(self.netD, 'D', which_epoch)
                            if self.classify:
                                self.load_network(self.netC, 'C', which_epoch)
                            if self.with_content_D:
                                self.load_network(self.netD_c, 'D_c', which_epoch)
                        else:
                            self.load_network(self.netD_A, 'D_A', which_epoch)
                            self.load_network(self.netD_B, 'D_B', which_epoch)
                            if self.classify:
                                self.load_network(self.netC, 'C', which_epoch)
                            if self.with_content_D:
                                self.load_network(self.netD_c, 'D_c', which_epoch)
                except:
                    print('Error loading models G_B, Ds')
            else:
                self.netG_A.eval()
                if not opt.stage1:
                    self.netG_B.eval()
        if opt.stage1:
            which_epoch = opt.which_epoch
            save_dir_temp = self.save_dir
            try:
                self.save_dir = 'new/s2ourdatanostyle_lv3_cari4_32'
                self.netG_B = networks.define_G(num_points=opt.numPoints, k=opt.fineDim, use_dropout=False, init_type='kaiming', gpu_ids=self.gpu_ids,concat=self.concat).cuda()
                self.load_network(self.netG_B, 'G_B', which_epoch)
                self.netG_B.eval()
            except:
                pass
            self.save_dir = save_dir_temp
        if opt.stage2:
            which_epoch = opt.which_epoch
            save_dir_temp = self.save_dir
            try:
                self.save_dir = 'new/ourdatanostyle_lv3_cari4_32'
                self.netG_prevA = networks.define_G(num_points=opt.numPoints, k=opt.fineDim, use_dropout=False, init_type='kaiming', gpu_ids=self.gpu_ids,concat=self.concat,no_enc_style=(opt.stage1 or opt.stage2)).cuda()
                self.load_network(self.netG_prevA, 'G_A', which_epoch)
                self.netG_B.eval()
            except:
                pass
            self.save_dir = save_dir_temp
            self.netG_A.eval() 
            if self.share_EC_s1_s2 and self.isTrain:
                pass
                
                



        self.cosine_loss = torch.nn.CosineEmbeddingLoss()
        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            
            self.fake_B_pool = ImagePool(opt.pool_size)
            
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionMatchingA = networks.MatchingLoss(num_features=32)
            self.criterionMatchingB = networks.MatchingLoss(num_features=32)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            
            if not opt.stage1:
                if opt.stage2:
                    self.optimizer_G = torch.optim.Adam(self.netG_B.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-4)
                else:
                    self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-4)
            else:
                self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-4)
            
            
            if not opt.stage1:
                if opt.stage2:
                    self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-4)
                    if self.classify:
                        self.optimizer_C = torch.optim.Adam(self.netC.parameters(), lr=5*opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-4)
                    if self.with_content_D:
                        self.optimizer_D_c = torch.optim.Adam(self.netD_c.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                                                        weight_decay=1e-4)
                else:
                    self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-4)
                    
                    self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-4)
                    
                    if self.with_content_D:
                        self.optimizer_D_c = torch.optim.Adam(self.netD_c.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                                                        weight_decay=1e-4)
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            
            if not opt.stage1:
                if opt.stage2:
                    self.optimizers.append(self.optimizer_D)
                    if self.classify:
                        self.optimizers.append(self.optimizer_C)
                    if self.with_content_D:
                        self.optimizers.append(self.optimizer_D_c)
                else:
                    self.optimizers.append(self.optimizer_D_A)
                    
                    self.optimizers.append(self.optimizer_D_B)
                    
                    if self.with_content_D:
                        self.optimizers.append(self.optimizer_D_c)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        
        if not opt.stage1:
            networks.print_network(self.netG_B)
            
            if self.isTrain:
                if opt.stage2:
                    networks.print_network(self.netD)
                    if self.classify:
                        networks.print_network(self.netC)
                    if self.with_content_D:
                        networks.print_network(self.netD_c)
                else:
                    networks.print_network(self.netD_A)
                    
                    networks.print_network(self.netD_B)
                    
                    if self.with_content_D:
                        networks.print_network(self.netD_c)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_A_img = input['A_img' if AtoB else 'B_img']
        input_B_img = input['B_img' if AtoB else 'A_img']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        if not self.isTrain:
            self.input_A_img.resize_(input_A_img.size()).copy_(input_A_img)
            self.input_B_img.resize_(input_B_img.size()).copy_(input_B_img)
        self.image_paths = input['A_paths']
        if self.classify:
            self.B_class = input['B_class']
        
        if self.load_translated_A:
            input_C = input['C'] 
            input_C_img = input['C_img']
            self.input_C.resize_(input_C.size()).copy_(input_C)
            if not self.isTrain:
                self.input_C_img.resize_(input_C_img.size()).copy_(input_C_img)

    def set_input_fix_B(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_A_img = input['A_img' if AtoB else 'B_img']
        input_B_img = input['B_img' if AtoB else 'A_img']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        
        self.input_A_img.resize_(input_A_img.size()).copy_(input_A_img)
        
        self.image_paths = input['A_paths']
        
        if self.classify:
            self.B_class = input['B_class']
        if self.load_translated_A:
            input_C = input['C'] 
            input_C_img = input['C_img']
            self.input_C.resize_(input_C.size()).copy_(input_C)
            if not self.isTrain:
                self.input_C_img.resize_(input_C_img.size()).copy_(input_C_img)

    def forward(self):
        self.real_A = Variable(self.input_A).cuda()
        self.real_B = Variable(self.input_B).cuda()
        if self.classify:
            self.B_class = Variable(self.B_class.type(torch.LongTensor)).cuda()
        if not self.isTrain or (not self.stage1 and not self.stage2):
            self.real_A_img = Variable(self.input_A_img).cuda()
            self.real_B_img = Variable(self.input_B_img).cuda()
        if self.load_translated_A:
            self.real_C = Variable(self.input_C).cuda()
            self.real_C_img = Variable(self.input_C_img).cuda()

    def test(self,rescale=-1.0):
        
        
        real_B = Variable(self.input_B, volatile=True)
        real_B_img = Variable(self.input_B_img, volatile=True)
        real_B_in = real_B.transpose(1,2).contiguous().view(real_B.size(0), -1)
        real_B_PCA = torch.mm(real_B_in - self.celeba_pca_mean, self.celeba_pca_mixing_to.t()) 
        
        
        
        
        
        
        
        

        
        
        real_A = Variable(self.input_A, volatile=True)
        real_A_img = Variable(self.input_A_img, volatile=True)
        real_A_in = real_A.transpose(1,2).contiguous().view(real_A.size(0), -1)
        real_A_PCA = torch.mm(real_A_in - self.art_pca_mean, self.art_pca_mixing_to.t()) 
        
        
        
        
        
        
        
        
        if self.stage1:
            content_A = self.netG_A.encode(real_A_PCA)
            
            x_ab = self.netG_A.decode(content_A) 
            content = self.netG_A.encode(real_B_PCA)
            x_b = self.netG_A.decode(content) 
            fake_A = torch.mm(x_ab, self.celeba_pca_mixing_return) + self.celeba_pca_mean
            fake_A = fake_A.view(fake_A.size(0), -1, 2).transpose(1, 2).contiguous()
            fake_A_img = self.img_warp(real_A_img, real_A, fake_A)
            rec_B = torch.mm(x_b, self.celeba_pca_mixing_return) + self.celeba_pca_mean
            rec_B = rec_B.view(rec_B.size(0), -1, 2).transpose(1, 2).contiguous()
            rec_B_img = self.img_warp(real_B_img, real_B, rec_B)
            fake_B = torch.mm(content_A, self.celeba_pca_mixing_return) + self.celeba_pca_mean
            fake_B = fake_B.view(fake_B.size(0), -1, 2).transpose(1, 2).contiguous()
            fake_B_img = self.img_warp(real_A_img, real_A, fake_B)
        elif self.stage2:
            content = self.netG_A.encode(real_A_PCA)
            if self.share_EC_s1_s2:
                content_b = self.netG_A.encode(real_B_PCA)
                _, style = self.netG_B.encode(real_B_PCA)
            else:
                content_b, style = self.netG_B.encode(real_B_PCA)
            
            s_b = Variable(torch.randn(real_B_PCA.size(0), 16).cuda()) 
            x_ab = self.netG_B.decode(content, style)
            x_b = self.netG_A.decode(content_b)
            
            

            
            
            
            
            
            fake_A = torch.mm(x_b, self.celeba_pca_mixing_return) + self.celeba_pca_mean
            fake_A = fake_A.view(fake_A.size(0), -1, 2).transpose(1, 2).contiguous()
            fake_A_img = self.img_warp(real_B_img, real_B, fake_A)

            x_recon = self.netG_B.decode(content_b, style)
            
            
            
            
            
            
            rec_B = torch.mm(content, self.celeba_pca_mixing_return) + self.celeba_pca_mean
            rec_B = rec_B.view(rec_B.size(0), -1, 2).transpose(1, 2).contiguous()
            rec_B_img = self.img_warp(real_A_img, real_A, rec_B)

            fake_B = torch.mm(x_ab, self.celeba_pca_mixing_return) + self.celeba_pca_mean
            if rescale > 0:
                
                res_B = fake_B-real_A_in
                fake_B = real_A_in + rescale*res_B
            fake_B = fake_B.view(fake_B.size(0), -1, 2).transpose(1, 2).contiguous()
            fake_B_img = self.img_warp(real_A_img, real_A, fake_B)
        else:
            content, _ = self.netG_A.encode(real_A_PCA)
            _, style = self.netG_B.encode(real_B_PCA) 
            x_ab = self.netG_B.decode(content, style) 
        



        
        
        self.fake_B = fake_B.data
        self.fake_B_img = fake_B_img.data
        self.fake_A = fake_A.data
        self.fake_A_img = fake_A_img.data
        
        
        
        
        
        

        content, _ = self.netG_B.encode(real_B_PCA)
        x_ab = self.netG_A.decode(content)
        rec_A = torch.mm(x_ab, self.celeba_pca_mixing_return) + self.celeba_pca_mean
        rec_A = rec_A.view(rec_A.size(0), -1, 2).transpose(1, 2).contiguous()
        self.rec_A = rec_A.data
        self.rec_A_img = self.img_warp(real_B_img, real_B, rec_A).data
        
        
        
        
        
        self.rec_B = rec_B.data
        self.rec_B_img = rec_B_img.data

        if self.get_txt:
            img_path=self.get_image_paths()[0]
            name = os.path.split(img_path)[-1]
            
            save_path = os.path.join(self.result_dir, name)
            print(self.result_dir,save_path)
            
            util.save_point(self.fake_B,save_path)
            input_style_name = name.split('.')[0] + '_instyle.txt'
            save_path = os.path.join(self.result_dir, input_style_name)
            util.save_point(real_B,save_path)


    
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        
        pred_real, pred_real_feat = netD(real)
        
        loss_D_real = self.criterionGAN(pred_real, True)
        
        
        pred_fake, pred_fake_feat = netD(fake.detach())
        
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5 
        
        return loss_D
    
    def backward_D_A(self):
        fake_B_PCA = self.fake_B_pool.query(self.fake_B_PCA)
        
        loss_D_A = self.backward_D_basic(self.netD_A, Variable(self.real_B_PCA), fake_B_PCA)
        
        self.loss_D_A = loss_D_A.data.item()
        
        loss_D = loss_D_A
        loss_D.backward()

    def backward_D_B(self):
        fake_A_PCA = self.fake_A_pool.query(self.fake_A_PCA)
        
        loss_D_B = self.backward_D_basic(self.netD_B, Variable(self.real_A_PCA), fake_A_PCA)
        
        self.loss_D_B = loss_D_B.data.item()
        
        loss_D = loss_D_B
        loss_D.backward()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target.view(input.shape[0],input.shape[1])))


    def backward_C(self):
        if self.stage2:
            real_B = self.real_B.transpose(1, 2).contiguous().view(self.real_B.size(0), -1)
            real_B_PCA = torch.mm(real_B - self.art_pca_mean, self.art_pca_mixing_to.t())
            pred_class = self.netC(Variable(real_B_PCA.cuda()))
            
            self.classifier_loss = torch.nn.functional.cross_entropy(pred_class, self.B_class.cuda()).mean()
            print(self.classifier_loss)
            self.classifier_loss.backward()

    def backward_D(self):
        if self.stage2:
            real_A = self.real_A.transpose(1, 2).contiguous().view(self.real_A.size(0), -1)
            real_A_PCA = torch.mm(real_A - self.celeba_pca_mean, self.celeba_pca_mixing_to.t())
            real_B = self.real_B.transpose(1, 2).contiguous().view(self.real_B.size(0), -1)
            real_B_PCA = torch.mm(real_B - self.art_pca_mean, self.art_pca_mixing_to.t())
            c_a = self.netG_A.encode(real_A_PCA).detach()
            
            s_b = Variable(torch.randn(real_B_PCA.size(0), 16).cuda())
            x_ab = self.netG_B.decode(c_a, s_b)

            
            self.loss_dis_total = self.netD.calc_dis_loss(x_ab.detach(), real_B_PCA)

            if self.with_content_D:
                
                s_b = Variable(torch.randn(real_B_PCA.size(0), 16).cuda()) 
                x_ab = self.netG_B.decode(c_a, s_b)
                if self.share_EC_s1_s2:
                    c_a_recon = self.netG_A.encode(x_ab)
                    _, s_b_recon = self.netG_B.encode(x_ab)
                else:
                    c_a_recon, s_b_recon = self.netG_B.encode(x_ab)
                self.loss_dis_total += self.netD_c.calc_dis_loss(c_a_recon.detach(), c_a)
            self.loss_dis_total.backward()
        else:
            lambda_idt = self.opt.identity
            lambda_A = self.opt.lambda_A
            lambda_B = self.opt.lambda_B
            real_A = self.real_A.transpose(1, 2).contiguous().view(self.real_A.size(0), -1)
            real_A_PCA = torch.mm(real_A - self.celeba_pca_mean, self.celeba_pca_mixing_to.t())
            
            real_B = self.real_B.transpose(1, 2).contiguous().view(self.real_B.size(0), -1)
            real_B_PCA = torch.mm(real_B - self.art_pca_mean, self.art_pca_mixing_to.t())
            
            
            
            
            if self.concat:
                
                
                s_a = Variable(torch.randn(real_A_PCA.size(0), 16).cuda())  
                s_b = Variable(torch.randn(real_B_PCA.size(0), 16).cuda())  
            else:
                s_a = Variable(torch.randn(real_A_PCA.size(0), 64).cuda())  
                s_b = Variable(torch.randn(real_B_PCA.size(0), 64).cuda())  
            
            c_a, _ = self.netG_A.encode(real_A_PCA)
            c_b, _ = self.netG_B.encode(real_B_PCA)
            
            x_ba = self.netG_A.decode(c_b, s_a)
            x_ab = self.netG_B.decode(c_a, s_b)
            
            self.loss_dis_a = self.netD_A.calc_dis_loss(x_ba.detach(), real_A_PCA)
            self.loss_dis_b = self.netD_B.calc_dis_loss(x_ab.detach(), real_B_PCA)
            if self.with_content_D:
                self.loss_dis_c = self.netD_c.calc_dis_loss(c_a,c_b,printinf=True)
            gan_w = 1 
            self.loss_dis_total = gan_w * self.loss_dis_a + gan_w * self.loss_dis_b
            if self.with_content_D:
                self.loss_dis_total+=self.loss_dis_c
            self.loss_dis_total.backward()





    def backward_G(self):
        if self.stage1:
            real_A = self.real_A.transpose(1,2).contiguous().view(self.real_A.size(0), -1)
            real_A_PCA = torch.mm(real_A - self.celeba_pca_mean, self.celeba_pca_mixing_to.t())

            c_a = self.netG_A.encode(real_A_PCA)
            x_a_recon = self.netG_A.decode(c_a)
            self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, real_A_PCA)

            
            
            
            

            
            
            
            

            
            
            
            
            
            
            if self.load_translated_A:
                trans_A = self.real_C.transpose(1,2).contiguous().view(self.real_C.size(0), -1)
                trans_A_PCA = torch.mm(trans_A - self.celeba_pca_mean, self.celeba_pca_mixing_to.t())
            else:
                s_b = Variable(torch.randn(real_A_PCA.size(0), 16).cuda()) 
                trans_A_PCA = self.netG_B.decode(c_a, s_b)
            trans_c_a = self.netG_A.encode(trans_A_PCA.detach())
            x_a_trans = self.netG_A.decode(trans_c_a)
            loss_trans_content = self.recon_criterion(x_a_trans, real_A_PCA)
            
            
            
            c_rand = Variable(torch.randn(real_A_PCA.size(0), 32).cuda()) 
            loss_consist_contentcode = torch.nn.KLDivLoss(reduce=True)(torch.nn.LogSoftmax(dim=-1)(trans_c_a),torch.nn.Softmax(dim=-1)(c_rand)) + torch.nn.KLDivLoss(reduce=True)(torch.nn.LogSoftmax(dim=-1)(c_a),torch.nn.Softmax(dim=-1)(c_rand))
            
            print('loss_gen_recon_x_a,loss_trans_content,loss_consist_contentcode: ',self.loss_gen_recon_x_a,loss_trans_content,loss_consist_contentcode)
            
            loss = 0.5*self.loss_gen_recon_x_a + 5.0*loss_trans_content
            loss.backward()

            rec_A_PCA = x_a_recon.clone()
            rec_A = torch.mm(rec_A_PCA, self.celeba_pca_mixing_return) + self.celeba_pca_mean
            rec_A = rec_A.view(rec_A.size(0), -1, 2).transpose(1, 2).contiguous()
            
            self.rec_A_PCA = rec_A_PCA.data
            
        elif self.stage2:
            
            real_A = self.real_A.transpose(1,2).contiguous().view(self.real_A.size(0), -1)
            real_A_PCA = torch.mm(real_A - self.celeba_pca_mean, self.celeba_pca_mixing_to.t())
            c_a = self.netG_A.encode(real_A_PCA).detach()

            
            real_B = self.real_B.transpose(1,2).contiguous().view(self.real_B.size(0), -1)
            real_B_PCA = torch.mm(real_B - self.celeba_pca_mean, self.celeba_pca_mixing_to.t())
            if self.share_EC_s1_s2:
                c_b = self.netG_A.encode(real_B_PCA)
                _, s_b_prime = self.netG_B.encode(real_B_PCA)
            else:
                c_b, s_b_prime = self.netG_B.encode(real_B_PCA)
            x_b_recon = self.netG_B.decode(c_b, s_b_prime)

            if self.share_EC_s1_s2:
                
                s_b_rand = Variable(torch.randn(real_B_PCA.size(0), 16).cuda()) 
                x_a = self.netG_B.decode(c_b,s_b_rand)
                c_a_reconB = self.netG_A.encode(x_a)

            
            
            s_b = Variable(torch.randn(real_B_PCA.size(0), 16).cuda()) 
            x_ab = self.netG_B.decode(c_a, s_b)
            
            
            

            
            if self.share_EC_s1_s2:
                c_a_recon = self.netG_A.encode(x_ab)
                _, s_b_recon = self.netG_B.encode(x_ab)
            else:
                c_a_recon, s_b_recon = self.netG_B.encode(x_ab)

            
            self.loss_gen_adv = self.netD.calc_gen_loss(x_ab)
            
            
            self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, real_B_PCA)
            self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b) 
            self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a) 

            if self.share_EC_s1_s2:
                
                self.loss_gen_recon_c_bB = self.recon_criterion(c_a_reconB, c_b) 

            
            
            
            
            
            
            
            self.loss_kl_s_b = torch.nn.KLDivLoss(reduce=True)(torch.nn.LogSoftmax(dim=-1)(s_b),torch.nn.Softmax(dim=-1)(s_b_prime))
            self.loss_kl_c_a = torch.nn.KLDivLoss(reduce=True)(torch.nn.LogSoftmax(dim=-1)(c_a),torch.nn.Softmax(dim=-1)(c_a_recon))

            
            if self.classify:
                pred_class = self.netC(x_ab)
                self.classify_loss = torch.nn.functional.cross_entropy(pred_class, self.B_class.cuda()).mean()
                

            
            
            
            self.loss_gen_total = self.omegas[0]*self.loss_gen_adv+1.0*self.loss_gen_recon_x_b+5.0*self.loss_gen_recon_s_b+self.omegas[1]*self.loss_gen_recon_c_a+ 10.0*self.loss_kl_s_b + 10.0*self.loss_kl_c_a 
            if self.share_EC_s1_s2:
                self.loss_gen_total += self.loss_gen_recon_c_bB
            if self.classify:
                self.loss_gen_total+=self.omegas[2]*self.classify_loss
                print(self.loss_gen_adv,self.loss_gen_recon_x_b,self.loss_gen_recon_s_b,self.loss_gen_recon_c_a,self.loss_kl_s_b,self.loss_kl_c_a,self.classify_loss)
            else:
                print(self.loss_gen_adv,self.loss_gen_recon_x_b,self.loss_gen_recon_s_b,self.loss_gen_recon_c_a,self.loss_kl_s_b,self.loss_kl_c_a)
            if self.with_content_D:
                self.loss_gen_c_a = self.netD.calc_gen_loss(c_a_recon)
                self.loss_gen_total += self.loss_gen_c_a
                print(self.loss_gen_adv,self.loss_gen_recon_x_b,self.loss_gen_recon_s_b,self.loss_gen_recon_c_a, self.loss_gen_c_a)
            
            
            self.loss_gen_total.backward()
        else:
            lambda_idt = self.opt.identity
            lambda_A = self.opt.lambda_A
            lambda_B = self.opt.lambda_B
            real_A = self.real_A.transpose(1,2).contiguous().view(self.real_A.size(0), -1)
            real_A_PCA = torch.mm(real_A - self.celeba_pca_mean, self.celeba_pca_mixing_to.t())
            
            real_B = self.real_B.transpose(1,2).contiguous().view(self.real_B.size(0), -1)
            real_B_PCA = torch.mm(real_B - self.art_pca_mean, self.art_pca_mixing_to.t())
            
            
            if lambda_idt > 0:
                
                
                
                
                
                if self.concat:
                    
                    
                    s_a = Variable(torch.randn(real_A_PCA.size(0), 16).cuda())  
                    s_b = Variable(torch.randn(real_B_PCA.size(0), 16).cuda())  
                    if self.lms:
                        
                        
                        s_a2 = Variable(torch.randn(real_A_PCA.size(0), 16).cuda())  
                        s_b2 = Variable(torch.randn(real_B_PCA.size(0), 16).cuda())  
                else:
                    s_a = Variable(torch.randn(real_A_PCA.size(0), 64).cuda())  
                    s_b = Variable(torch.randn(real_B_PCA.size(0), 64).cuda())  
                    if self.lms:
                        s_a2 = Variable(torch.randn(real_A_PCA.size(0), 64).cuda())  
                        s_b2 = Variable(torch.randn(real_B_PCA.size(0), 64).cuda())  
                
                c_a, s_a_prime = self.netG_A.encode(real_A_PCA)
                c_b, s_b_prime = self.netG_B.encode(real_B_PCA)
                
                x_a_recon = self.netG_A.decode(c_a, s_a_prime)
                x_b_recon = self.netG_B.decode(c_b, s_b_prime)
                
                x_ba = self.netG_A.decode(c_b, s_a)
                x_ab = self.netG_B.decode(c_a, s_b)
                if self.lms:
                    x_ba2 = self.netG_A.decode(c_b, s_a2)
                    x_ab2 = self.netG_B.decode(c_a, s_b2)
                
                c_b_recon, s_a_recon = self.netG_A.encode(x_ba)
                c_a_recon, s_b_recon = self.netG_B.encode(x_ab)
                
                recon_x_cyc_w = 10 
                x_aba = self.netG_A.decode(c_a_recon, s_a_prime) if recon_x_cyc_w > 0 else None
                x_bab = self.netG_B.decode(c_b_recon, s_b_prime) if recon_x_cyc_w > 0 else None

                
                self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, real_A_PCA)
                self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, real_B_PCA)
                
                self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a) 
                self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b) 
                self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a) 
                self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b) 
                self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, real_A_PCA) if recon_x_cyc_w > 0 else 0
                self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, real_B_PCA) if recon_x_cyc_w > 0 else 0
                
                self.loss_gen_adv_a = self.netD_A.calc_gen_loss(x_ba) 
                self.loss_gen_adv_b = self.netD_B.calc_gen_loss(x_ab) 

                if self.with_content_D:
                    self.loss_gen_adv_c = self.netD_c.calc_gen_loss(c_a,content=True)  
                    self.loss_gen_adv_c += self.netD_c.calc_gen_loss(c_b, content=True)  
                    print(self.loss_gen_adv_c, self.loss_gen_recon_s_a, self.loss_gen_recon_s_b, self.loss_gen_cycrecon_x_a,
                        self.loss_gen_cycrecon_x_b)
                else:
                    
                    
                    print(self.loss_gen_recon_s_a,self.loss_gen_recon_s_b,self.loss_gen_cycrecon_x_a,self.loss_gen_cycrecon_x_b)
                
                vgg_w = 0 
                self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, real_B_PCA) if vgg_w > 0 else 0
                self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, real_A_PCA) if vgg_w > 0 else 0
                
                gan_w = 1 
                recon_x_w = 10 
                recon_s_w = 1 
                recon_c_w = 1 
                
                
                
                
                
                
                
                
                
                
                
                
                adv_weight = 1
                style_rec_weight = 1
                self.loss_gen_total = style_rec_weight*(self.loss_gen_recon_s_a + self.loss_gen_recon_s_b) + adv_weight*(self.loss_gen_adv_a + self.loss_gen_adv_b)
                self.loss_gen_total += self.loss_gen_recon_c_a + self.loss_gen_recon_c_b
                self.loss_gen_total += self.loss_gen_recon_x_a + self.loss_gen_recon_x_b 
                
                if self.with_content_D:
                    self.loss_gen_total += self.loss_gen_adv_c
                if self.lms:
                    lz_AB = torch.mean(torch.abs(x_ab2 - x_ab)) / torch.mean(torch.abs(s_b2 - s_b))
                    lz_BA = torch.mean(torch.abs(x_ba2 - x_ba)) / torch.mean(torch.abs(s_a2 - s_a))
                    eps = 1 * 1e-5
                    loss_lz_AB = 1 / (lz_AB + eps)
                    loss_lz_BA = 1 / (lz_BA + eps)
                    self.loss_gen_total += loss_lz_AB+loss_lz_BA
                self.loss_gen_total.backward()
                
                fake_A_PCA =x_ba.clone()
                fake_A = torch.mm(fake_A_PCA, self.celeba_pca_mixing_return) + self.celeba_pca_mean
                fake_A = fake_A.view(fake_A.size(0), -1, 2).transpose(1, 2).contiguous()
                fake_A_img = self.img_warp(self.real_B_img, self.real_B, fake_A)
                
                fake_B_PCA = x_ab.clone()
                fake_B = torch.mm(fake_B_PCA, self.celeba_pca_mixing_return) + self.celeba_pca_mean
                fake_B = fake_B.view(fake_B.size(0), -1, 2).transpose(1, 2).contiguous()
                fake_B_img = self.img_warp(self.real_A_img, self.real_A, fake_B)

                
                rec_A_PCA = x_a_recon.clone()
                rec_B_PCA = x_a_recon.clone()
                rec_A = torch.mm(rec_A_PCA, self.celeba_pca_mixing_return) + self.celeba_pca_mean
                rec_A = rec_A.view(rec_A.size(0), -1, 2).transpose(1, 2).contiguous()
                rec_A_img = self.img_warp(fake_B_img, fake_B, rec_A)
                rec_B = torch.mm(rec_B_PCA, self.celeba_pca_mixing_return) + self.celeba_pca_mean
                rec_B = rec_B.view(rec_B.size(0), -1, 2).transpose(1, 2).contiguous()
                rec_B_img = self.img_warp(fake_A_img, fake_A, rec_B)

                idt_A_PCA = self.netG_A(real_A_PCA)
                idt_A = torch.mm(idt_A_PCA, self.art_pca_mixing_return) + self.art_pca_mean
                idt_A = idt_A.view(idt_A.size(0), -1, 2).transpose(1,2).contiguous()
                self.idt_A = idt_A.data
                idt_B_PCA = self.netG_B(real_B_PCA)
                idt_B = torch.mm(idt_B_PCA, self.celeba_pca_mixing_return) + self.celeba_pca_mean
                idt_B = idt_B.view(idt_B.size(0), -1, 2).transpose(1,2).contiguous()
                self.idt_B = idt_B.data



                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

            else:
                loss_idt_A = 0
                loss_idt_B = 0
                self.loss_idt_A = 0
                self.loss_idt_B = 0

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            self.real_A_PCA = real_A_PCA.data
            self.real_B_PCA = real_B_PCA.data
            self.fake_A_PCA = fake_A_PCA.data
            self.fake_B_PCA = fake_B_PCA.data
            self.rec_A_PCA = rec_A_PCA.data
            self.rec_B_PCA = rec_B_PCA.data
            self.fake_B = fake_B.data
            self.fake_B_img = fake_B_img.data
            self.fake_A = fake_A.data
            self.fake_A_img = fake_A_img.data
            self.rec_A = rec_A.data
            self.rec_A_img = rec_A_img.data
            self.rec_B = rec_B.data
            self.rec_B_img = rec_B_img.data
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            self.loss_G_A = 0
            self.loss_G_A_delta = 0
            self.loss_G_B = 0
            self.loss_G_B_delta = 0
            self.loss_cycle_A = 0
            self.loss_cycle_B = 0
            

    def optimize_c(self):
        self.forward()
        self.optimizer_C.zero_grad()
        self.backward_C()
        self.optimizer_C.step()
    
    
    
    def optimize_parameters(self,epoch=0):
        if self.classify and epoch<50:
            self.optimize_c()
            return self.classifier_loss
        
        self.forward()

        if not self.stage1:
            if self.stage2:
                self.optimizer_D.zero_grad()
                self.backward_D()
                self.optimizer_D.step()
                if self.classify:
                    self.optimizer_C.zero_grad()
                    self.backward_C()
                    self.optimizer_C.step()
                if self.with_content_D:
                    self.optimizer_D_c.zero_grad()
                    self.backward_D()
                    self.optimizer_D_c.step()
            else:
                if self.with_content_D:
                    
                    self.optimizer_D_c.zero_grad()
                    self.backward_D()
                    self.optimizer_D_c.step()

                
                self.optimizer_D_A.zero_grad()
                self.backward_D()
                self.optimizer_D_A.step()

                self.optimizer_D_B.zero_grad()
                self.backward_D()
                self.optimizer_D_B.step()

                
                self.forward()
                
        self.optimizer_G.zero_grad()
        
        self.backward_G()
        
        self.optimizer_G.step()
        
        
        
        
        
        
        
        
        
        
        
        
        
        if self.stage2:
            return self.loss_gen_adv,self.loss_gen_recon_x_b,self.loss_gen_recon_s_b,self.loss_gen_recon_c_a,self.loss_kl_s_b,self.loss_kl_c_a,self.classify_loss

    def img_warp(self, img, pts_ori, pts_tar):
        
        
        if img.shape[0] == 1: 
            pts_ori = pts_ori.squeeze(0).t()
            pts_tar = pts_tar.squeeze(0).t()
            pts = pts_tar
            vals = (pts_tar - pts_ori)
            
            pts = (pts + 1) / 2 * 512 + 0
            flow_shift = self.dsi(pts, vals) 
        else:
            flow_shift_list = []
            for i in range(img.shape[0]):
                pts_ori_temp = pts_ori[i]
                pts_tar_temp = pts_tar[i]
                pts_ori_temp = pts_ori_temp.squeeze(0).t()
                pts_tar_temp = pts_tar_temp.squeeze(0).t()
                pts = pts_tar_temp
                vals = (pts_tar_temp - pts_ori_temp)
                
                pts = (pts + 1) / 2 * 512 + 0
                flow_shift_list.append(self.dsi(pts, vals).unsqueeze(0)) 
            flow_shift = torch.cat(flow_shift_list,0) 
        
        
        coord_x = torch.arange(-1, 1, 2/(self.fineSize)).unsqueeze(0).expand((self.fineSize, self.fineSize))
        coord_y = torch.arange(-1, 1, 2/(self.fineSize)).unsqueeze(1).expand((self.fineSize, self.fineSize))
        coord = torch.cat((coord_x.unsqueeze(2), coord_y.unsqueeze(2)), 2)
        coord = Variable(coord.cuda()) 
        if img.shape[0] > 1:
            coord_list = []
            for i in range(img.shape[0]):
                coord_list.append(coord.unsqueeze(0))
            coord = torch.cat(coord_list, 0) 
        flow_grid = coord - flow_shift
        if img.shape[0] == 1:
            flow_grid = flow_grid.unsqueeze(0).cuda()
        else:
            flow_grid = flow_grid.cuda()
        img = F.grid_sample(img, flow_grid, padding_mode='border')
        return img

    def get_current_errors(self):
        
        
        ret_errors = OrderedDict([('D_A', 0), ('G_A', 0), ('G_A_delta', 0),
                                  ('Cyc_A', 0),
                                  ('D_B', 0), ('G_B', 0), ('G_B_delta', 0),
                                  ('Cyc_B', 0)])
        if self.opt.identity > 0.0:
            
            
            ret_errors['idt_A'] = 0
            ret_errors['idt_B'] = 0
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        real_A_img = util.get_im_from_tensor(self.input_A_img)
        fake_B = util.tensor2im(self.fake_B)
        fake_B_img = util.get_im_from_tensor(self.fake_B_img)
        rec_A = util.tensor2im(self.rec_A)
        rec_A_img = util.get_im_from_tensor(self.rec_A_img)
        real_B = util.tensor2im(self.input_B)
        real_B_img = util.get_im_from_tensor(self.input_B_img)
        fake_A = util.tensor2im(self.fake_A)
        fake_A_img = util.get_im_from_tensor(self.fake_A_img)
        rec_B = util.tensor2im(self.rec_B)
        rec_B_img = util.get_im_from_tensor(self.rec_B_img)
        ret_visuals = OrderedDict([('real_A', real_A), ('real_A_img', real_A_img), ('fake_B', fake_B), ('fake_B_img', fake_B_img), ('rec_A', rec_A), ('rec_A_img', rec_A_img),
                                   ('real_B', real_B), ('real_B_img', real_B_img), ('fake_A', fake_A), ('fake_A_img', fake_A_img), ('rec_B', rec_B), ('rec_B_img', rec_B_img),
                                   ('real_B_Points', self.input_B), ('fake_B_Points', self.fake_B)])
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        
        if not self.stage1:
            if self.stage2:
                self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
                self.save_network(self.netD, 'D', label, self.gpu_ids)
                if self.classify:
                    self.save_network(self.netC, 'C', label, self.gpu_ids)
                if self.with_content_D:
                    self.save_network(self.netD_c, 'D_c', label, self.gpu_ids)
            else:
                self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
                
                self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
                
                self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
                
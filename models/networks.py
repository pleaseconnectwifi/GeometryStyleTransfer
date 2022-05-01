import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np





def weights_init_normal(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class define_G(nn.Module):
    def __init__(self,num_points=60, k=2, use_dropout=False, init_type='normal', gpu_ids=[],concat=False,no_enc_style=False):
        super(define_G, self).__init__()
        self.use_gpu = len(gpu_ids) > 0
        self.concat = concat
        self.no_enc_style = no_enc_style

        if self.use_gpu:
            assert(torch.cuda.is_available())

        
        self.enc_content= ContenEncoder(num_points=num_points, k=2)
        if not self.no_enc_style:
            self.enc_style = StyleEncoder(num_points=num_points, k=2,concat =self.concat)
        self.dec = Decoder(num_points=num_points, k=2, concat =self.concat, no_enc_style=self.no_enc_style)
        if not self.concat and not self.no_enc_style:
            self.mlp = MLP(64, self.get_num_adain_params(self.dec), 64, 3, norm='none', activ='lrelu') 

        if len(gpu_ids) > 0:
            
            self.enc_content.cuda(gpu_ids[0])
            if not self.no_enc_style:
                self.enc_style.cuda(gpu_ids[0])
            self.dec.cuda(gpu_ids[0])
            if not self.concat and not self.no_enc_style:
                self.mlp.cuda(gpu_ids[0])
        
        init_weights(self.enc_content, init_type=init_type)
        if not self.no_enc_style:
            init_weights(self.enc_style, init_type=init_type)
        init_weights(self.dec, init_type=init_type)
        if not self.concat and not self.no_enc_style:
            init_weights(self.mlp, init_type=init_type)
        

    def forward(self, x):
        if not self.no_enc_style:
            content, style_fake = self.encode(x)
            x_recon = self.decode(content, style_fake)
        else:
            content = self.encode(x)
            x_recon = self.decode(content)
        return x_recon

    def encode(self, images):
        
        content = self.enc_content(images)
        if not self.no_enc_style:
            style_fake = self.enc_style(images)
            return content, style_fake
        return content

    def decode(self, content, style=None):
        
        if self.concat and not self.no_enc_style:
            
            images = self.dec(torch.cat((content,style),1),input_content=content) 
        elif not self.no_enc_style:
            images = self.dec(content)
            adain_params = self.mlp(style)
            self.assign_adain_params(adain_params, self.dec)
        else: 
            images = self.dec(content) 
        return images

    def assign_adain_params(self, adain_params, model):
        
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params






def define_G_img(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG

class define_D(nn.Module):
    def __init__(self,num_points=60, k=2, init_type='normal', gpu_ids=[]):
        super(define_D, self).__init__()
        
        use_gpu = len(gpu_ids) > 0

        if use_gpu:
            assert(torch.cuda.is_available())
        self.netD = PointNetDis(num_points=num_points, k=2)
        if use_gpu:
            self.netD.cuda(gpu_ids[0])  
        init_weights(self.netD, init_type=init_type)
        

    def forward(self, x):
        outputs = []
        outputs = self.netD(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real, printinf=False):
        
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0
        if printinf:
            print(input_fake[0][0],input_real[0][0],outs0[0][0],outs1[0][0])

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if printinf:
                loss += torch.mean((out0) ** 2) + torch.mean((1-out1) ** 2)  
            else:
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2) 
            
            
            
            
            
            
            
            
            
        return loss

    def calc_gen_loss(self, input_fake, content=False):
        
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if content:
                
                loss += torch.mean((out0 - 0.5) ** 2)
            else:
                loss += torch.mean((out0 - 1) ** 2)  
            
            
            
            
            
            
        return loss



class define_C(nn.Module):
    def __init__(self,num_points=68, k=2, init_type='normal', gpu_ids=[]):
        super(define_C, self).__init__()
        
        use_gpu = len(gpu_ids) > 0

        if use_gpu:
            assert(torch.cuda.is_available())
        
        self.netC = PointNetDis(num_points=num_points, k=k, fmap=[68,128,256,64])
        if use_gpu:
            self.netC.cuda(gpu_ids[0])  
        init_weights(self.netC, init_type=init_type)
        

    def forward(self, x):
        outputs = self.netC(x)[0]
        
        return torch.nn.functional.softmax(outputs,dim=1)




def define_D_img(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)











class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class MatchingLoss(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(MatchingLoss, self).__init__()
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features).cuda())
        self.loss = nn.MSELoss()
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
    
    def __call__(self, input, target):
        target_mean = torch.mean(target, dim=0)
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * target_mean.data
        target_tensor = Variable(self.running_mean.unsqueeze(0).expand_as(input), requires_grad=False)
        return self.loss(input, target_tensor)




class STN2d(nn.Module):
    def __init__(self, num_points = 30):
        super(STN2d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(2, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        
        x, _ = torch.max(x, 2)
        
        x = x.view(-1, 128)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.Tensor([1,0,0,1]).repeat(batchsize,1))
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 2, 2)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, num_points = 30, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN2d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(2, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)

        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)

        
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        
        
        
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 256)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 256, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans

class PointNetDis(nn.Module):
    
    def __init__(self, num_points = 30, k = 2,fmap=[32,64,256,32]):
        super(PointNetDis, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(num_points, fmap[0])
        self.fc2 = nn.Linear(fmap[0], fmap[1])
        self.fc3 = nn.Linear(fmap[1], fmap[2])
        self.fc4 = nn.Linear(fmap[2], fmap[3])
        self.fc5 = nn.Linear(fmap[3], k)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x_feat = self.fc4(x)
        x = F.relu(x_feat)
        x = self.fc5(x)
        return x, x_feat


class PointNetEncoder(nn.Module):
    def __init__(self, num_points = 60, global_feat = True):
        super(PointNetEncoder, self).__init__()
        self.stn = STN2d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(2, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)

        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)

        
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        
        
        
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 256)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 256, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans

class PointDecoder(nn.Module):
    def __init__(self, num_points = 60):
        super(PointDecoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)
        
        self.fc3 = nn.Linear(128, 32)
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


























class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = None
        self.bias = None
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        if x.shape[2] > 1:
            out = F.batch_norm(
                x_reshaped, running_mean, running_var, self.weight, self.bias,
                True, self.momentum, self.eps)
        else: 
            out = F.instance_norm(x_reshaped[:,:,:1], running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', no_act=False):
        super(ResBlock, self).__init__()

        model = []
        model += [nn.Linear(dim ,dim),nn.ReLU()]
        if no_act:
            model += [nn.Linear(dim ,dim)]
        else:
            model += [nn.Linear(dim, dim), nn.ReLU()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks-1):
            self.model += [ResBlock(dim)]
        self.model += [ResBlock(dim,no_act=True)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ContenEncoder(nn.Module):

    def __init__(self, num_points=30, k=2):
        super(ContenEncoder, self).__init__()
        self.num_points = num_points
        
        
        
        
        
        
        self.fc1 = MLP(32, 64, 32, 2, norm='none', activ='none') 
        self.fc2 = MLP(64, 128, 64, 2, norm='none', activ='none')
        self.fc3 = MLP(128, 64, 32, 2, norm='none', activ='none')
        
        self.fc4 = MLP(64, 32, 16, 2, norm='none', activ='none')
        self.resblocks = None


    def forward(self, input):
        x = input 

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        if self.resblocks is not None:
            x = self.resblocks(x)
        
        
        
        
        
        x = x + input 
        return x

class StyleEncoder(nn.Module):

    def __init__(self, num_points=30, k=2,concat =False):
        super(StyleEncoder, self).__init__()
        self.num_points = num_points
        self.concat = concat
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        self.fc0 = nn.Sequential(MLP(32, 32, 32, 4, norm='none', activ='none'))
        self.fc1 = nn.Sequential(MLP(32, 64, 64, 4, norm='none', activ='none'))
        
        if self.concat:
            
            
            self.fc2 = nn.Sequential(MLP(64, 32, 64, 4, norm='none', activ='none'))
            self.fc3 = nn.Sequential(MLP(32, 32, 32, 4, norm='none', activ='none'),nn.ReLU(),MLP(32, 16, 32, 4, norm='none', activ='none'))
            
        else:
            self.fc3 = MLP(16, 64, 64, 2, norm='none', activ='none')

    def forward(self, input):
        x = input 

        f0 = F.relu(self.fc0(x))
        x = F.relu(self.fc1(f0))
        
        if self.concat:
            x = F.relu(self.fc2(x)) + f0
            x = self.fc3(x)
        
        
        
        
        
        
        
        
        
        
        return x

class Decoder(nn.Module):
    def __init__(self, num_points=30, k=2, in_dim=16, concat=False, no_enc_style=False):
        super(Decoder, self).__init__()
        self.num_points = num_points
        self.concat = concat
        self.no_enc_style=no_enc_style
        
        
        
        
        
        
        
        
        
        
        
        

        if self.concat and self.no_enc_style:
            
            
            self.fc1 = MLP(32, 32, 32, 2, norm='none', activ='none')
        elif self.concat:
            
            self.fc1 = nn.Sequential(MLP(48, 48, 48, 2, norm='none', activ='none'),nn.ReLU(),MLP(48, 32, 32, 2, norm='none', activ='none'))
        else:
            
            self.fc1 = MLP(32, 32, 32, 2, norm='none', activ='none')
        self.fc2 = MLP(32, 64, 64, 2, norm='none', activ='none')
        if not self.concat:
            self.norm = AdaptiveInstanceNorm2d(64) 
        self.fc3 = MLP(64, 32, 32, 2, norm='none', activ='none')
        
        self.fc4 = MLP(32, 32, 32, 2, norm='none', activ='none')

    def forward(self, input, input_content=None):
        x = input 

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.concat or self.no_enc_style:
            x = F.relu(x)
        else:
            x = self.norm(x.view(x.shape[0],64,1).contiguous())
        x = F.relu(self.fc3(x.view(x.shape[0],-1).contiguous()))
        x = self.fc4(x)
        
        
        
        if not self.no_enc_style:
            
            
            zero_mask = Variable(torch.ByteTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                             requires_grad=False).cuda()
            
            x.masked_fill_(zero_mask.bool(), 0)
        
        if self.concat and not self.no_enc_style:
            x = x + input_content
        else:
            x = x + input 
        return x

import math
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [FCBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [FCBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [FCBlock(dim, output_dim, norm='none', activation='none')] 
        self.model = nn.Sequential(*self.model)
        print('-------init MLP------')
        self.apply(weights_init('kaiming'))

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class FCBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(FCBlock, self).__init__()
        use_bias = True
        
        
        
        
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        
        
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out




class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
        





class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)



class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out






class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)





class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
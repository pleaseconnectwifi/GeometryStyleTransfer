import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.load_name is None:
             opt.load_name = opt.name
        self.load_dir = os.path.join(opt.checkpoints_dir, opt.load_name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    
    def load_network(self, network, network_label, epoch_label, load_prevstage=False):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        
        if load_prevstage:
            save_path = os.path.join(self.load_dir, save_filename)
        else:
            save_path = os.path.join(self.save_dir, save_filename)
        save_model = torch.load(save_path)
        network.load_state_dict(save_model)

    
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

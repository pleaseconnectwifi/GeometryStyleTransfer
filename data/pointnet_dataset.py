import os
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
import PIL
from PIL import Image
import random
import numpy as np
import glob
import random

class PointnetDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        if opt.C_paths is not None:
            self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        self.serial_batches = opt.serial_batches
        self.is_test = (opt.phase=='test' or 'test' in opt.phase)
        if self.is_test:
            self.each_c_each_s = opt.each_c_each_s

        
        
        self.A_paths = glob.iglob(os.path.join(self.dir_A, '**/*.txt'), recursive=True)
        self.B_paths = glob.iglob(os.path.join(self.dir_B, '**/*.txt'), recursive=True)
        if opt.C_paths is not None:
            self.A_paths = glob.iglob(os.path.join(self.dir_C, '**/*content*.txt'), recursive=True)
            self.C_paths = glob.iglob(os.path.join(self.dir_C, '**/*style*.txt'), recursive=True)
            self.A_paths = sorted(self.A_paths)
            self.C_paths = sorted(self.C_paths)
        self.classify = opt.classify
        if opt.classify:
            self.classes_names = {}
            cnt = 0
            for _,dirs,_ in os.walk(self.dir_B):
                for dir in dirs:
                    if dir:
                        self.classes_names[dir]=cnt
                        cnt += 1
            
            print('classes_names: ',self.classes_names)
        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        if self.is_test and self.each_c_each_s:
            self.A_paths_temp = []
            self.A_output_paths_temp = []
            self.B_paths_temp = []
            for i in range(self.A_size):
                for j in range(self.B_size):
                    self.A_paths_temp.append(self.A_paths[i])
                    self.A_output_paths_temp.append(self.A_paths[i].replace('.txt', '_'+str(j)+'.txt'))
                    self.B_paths_temp.append(self.B_paths[j])
            self.A_paths = self.A_paths_temp
            self.B_paths = self.B_paths_temp
            self.A_size = len(self.A_paths)
            self.B_size = len(self.B_paths)
        if opt.C_paths is not None:
            self.C_paths = sorted(self.C_paths)
            self.C_size = len(self.C_paths)
        else:
            self.C_paths = None
        self.img_transform = self.get_img_transform()
        

    def transformA(self, mat, flip, shift):
        
        mat[:, 0] -= - shift[0]
        mat[:, 1] -= - shift[1]
        
        
        mat[:, 0] = ((mat[:, 0]-0) / 512 - 0.5) * 2
        mat[:, 1] = ((mat[:, 1]-0) / 512 - 0.5) * 2
        
        
        a=1.2/3.0
        b=0.9/3.0
        c=0.9/3.0
        mat[60,:] = a*mat[48,:]+b*mat[61,:]+c*mat[67,:]
        mat[64,:] = a*mat[54,:]+b*mat[63,:]+c*mat[65,:]
        mat[61,0] = 0.5*mat[50,0]+0.5*mat[58,0]
        mat[61,1] = 0.4*mat[60,1]+0.6*mat[62,1]
        mat[63,0] = 0.5*mat[52,0]+0.5*mat[56,0]
        mat[63,1] = 0.4*mat[64,1]+0.6*mat[62,1]
        mat[65,0] = 0.5*mat[52,0]+0.5*mat[56,0]
        mat[65,1] = 0.4*mat[64,1]+0.6*mat[66,1]
        mat[67,0] = 0.5*mat[50,0]+0.5*mat[58,0]
        mat[67,1] = 0.4*mat[60,1]+0.6*mat[66,1]
        if flip:
            mat[:, 0] = - mat[:, 0] 
            
            
            index = [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,26,25,24,23,22,21,20,19,18,17,27,28,29,30,35,34,33,32,31,45,44,43,42,47,46,39,38,37,36,41,40,
            54,53,52,51,50,49,48,59,58,57,56,55,64,63,62,61,60,67,66,65]

            mat = mat[index, :]
            
        mat = mat.T
        return mat

    def transformB(self, mat, flip, shift):
        
        mat[:, 0] -= - shift[0]
        mat[:, 1] -= - shift[1]
        
        
        mat[:, 0] = ((mat[:, 0]-0) / 512 - 0.5) * 2
        mat[:, 1] = ((mat[:, 1]-0) / 512 - 0.5) * 2
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        if flip:
            mat[:, 0] = - mat[:, 0] 
            
            
            index = [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,26,25,24,23,22,21,20,19,18,17,27,28,29,30,35,34,33,32,31,45,44,43,42,47,46,39,38,37,36,41,40,54,53,52,51,50,49,48,59,58,57,56,55,64,63,62,61,60,67,66,65]

            mat = mat[index, :]
            
        mat = mat.T
        return mat

    def get_img_transform(self):
        transform_list = [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)
        
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.C_paths is not None:
            
            C_path = A_path.replace('content','style')
        if not self.serial_batches:
            B_path = self.B_paths[random.randint(0, self.B_size - 1)]
        else:
            B_path = self.B_paths[index % self.B_size]
        
        A = np.loadtxt(A_path)
        B = np.loadtxt(B_path)
        if self.C_paths is not None:
            C = np.loadtxt(C_path)
        A_img_path = A_path.replace('.txt', '.jpg')
        B_img_path = B_path.replace('.txt', '.jpg')
        if self.C_paths is not None:
            C_img_path = C_path.replace('.txt', '.jpg')
        A_img = Image.open(A_img_path).convert('RGB')
        B_img = Image.open(B_img_path).convert('RGB')
        if self.C_paths is not None:
            C_img = Image.open(C_img_path).convert('RGB')
        if self.is_test:
            flip_A = False
            flip_B = False
        else:
            flip_A = (random.random() < 0.5)
            flip_B = (random.random() < 0.5)
        if self.is_test:
            shift_A = [0,0]
            shift_B = [0,0]
        else:
            shift_A = [random.randrange(-4, 4), random.randrange(-4, 4)]
            shift_B = [random.randrange(-4, 4), random.randrange(-4, 4)]
        
        
        
        
        
        A = self.transformB(A, flip_A, shift_A)
        if self.C_paths is not None:
            C = self.transformB(C, flip_A, shift_A)
        
        
        
        
        


        if flip_A:
            A_img = transforms.functional.hflip(A_img)
        
        if A_img.size != (304,304): 
            
            A_img = transforms.functional.crop(A_img, - shift_A[1], - shift_A[0], A_img.size[0], A_img.size[1])
        else:
            A_img = transforms.functional.crop(A_img, - shift_A[1], - shift_A[0], 304, 304)
        A_img = self.img_transform(A_img)
        if self.C_paths is not None:
            C_img = self.img_transform(C_img)
        B = self.transformB(B, flip_B, shift_B)
        if flip_B:
            B_img = transforms.functional.hflip(B_img)
        if B_img.size != (304,304):
            
            B_img = transforms.functional.crop(B_img, - shift_B[1], - shift_B[0], B_img.size[0],  B_img.size[1])
        else:
            B_img = transforms.functional.crop(B_img, - shift_B[1], - shift_B[0], 304, 304)
        B_img = self.img_transform(B_img)  
        if self.is_test and self.each_c_each_s:
            A_path = self.A_output_paths_temp[index % self.A_size]
        if self.classify:   
            try:
                B_class = B_path.split('/')[-2]
                B_class = self.classes_names[B_class]
            except:
                B_class=0
            if self.C_paths is not None:
                return {'A': A, 'B': B, 'C': C, 'A_img': A_img, 'B_img': B_img, 'C_img': C_img, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path, 'B_class':B_class}
            else:
                return {'A': A, 'B': B, 'A_img': A_img, 'B_img': B_img, 'A_paths': A_path, 'B_paths': B_path, 'B_class':B_class}
        
        if self.C_paths is not None:
            return {'A': A, 'B': B, 'C': C, 'A_img': A_img, 'B_img': B_img, 'C_img': C_img, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path}
        else:
            return {'A': A, 'B': B, 'A_img': A_img, 'B_img': B_img, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'PointnetDataset'

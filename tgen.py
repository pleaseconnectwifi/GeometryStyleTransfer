import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   
    opt.batchSize = 1  
    opt.serial_batches = True  
    opt.no_flip = True  

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_intermidiate(visuals, img_path)
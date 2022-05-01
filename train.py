import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('num training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    writer = SummaryWriter()

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            
            if opt.stage2:
                try:
                    loss_gen_adv,loss_gen_recon_x_b,loss_gen_recon_s_b,loss_gen_recon_c_a,loss_kl_s_b,loss_kl_c_a,classify_loss = model.optimize_parameters(epoch)
                    writer.add_scalar("loss_gen_adv/train", loss_gen_adv, total_steps)
                    writer.add_scalar("loss_gen_recon_x_b/train", loss_gen_recon_x_b, total_steps)
                    writer.add_scalar("loss_gen_recon_s_b/train", loss_gen_recon_s_b, total_steps)
                    writer.add_scalar("loss_gen_recon_c_a/train", loss_gen_recon_c_a, total_steps)
                    writer.add_scalar("loss_kl_s_b/train", loss_kl_s_b, total_steps)
                    writer.add_scalar("loss_kl_c_a/train", loss_kl_c_a, total_steps)
                    writer.add_scalar("classify_loss/train", classify_loss, total_steps)
                    writer.flush()
                except:
                    classifier_loss = model.optimize_parameters(epoch)
                    writer.add_scalar("classifier_loss/pretrain", classifier_loss, total_steps)
                    writer.flush()
            else:
                model.optimize_parameters(epoch)

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    writer.close()

#! /bin/bash
echo 'clear results folder ...'
if [ -d results/ ]
then
    rm -rf results/*
fi
eval "$(conda shell.bash hook)"
conda activate pytorch1.7

# s1 train
python train.py --dataroot datasets/point/ --model cycle_gan --no_dropout --niter 100 --niter_decay 100 --stage1 --name s1run --concat --batchSize 64 --numPoints 68 --height 512 --width 512 --fineSize 304 --C_paths datasets/point/ --no_html --display_id -1;

# s1 test
python test.py --dataroot datasets/point/ --model cycle_gan --no_dropout --stage1 --name s1run --concat --numPoints 68 --height 512 --width 512 --fineSize 512 --display_winsize 256;

# s2 train
python train.py --dataroot datasets/point/ --model cycle_gan --no_dropout --niter 450 --niter_decay 350 --stage2 --concat --name s2run --load_name s1run --continue_train --batchSize 32 --lr 0.0005 --numPoints 68 --DimClass 25 --height 512 --width 512 --fineSize 304 --classify --omega 1.0 0.5 0.1 --share_EC_s1_s2 --no_html;

# s2 test
python test.py --dataroot datasets/point/ --model cycle_gan --no_dropout --stage2 --concat --name s2run --load_name s1run --numPoints 68 --height 512 --width 512 --fineSize 512 --display_winsize 256 --classify --get_txt --phase test --how_many -1 --rescale 1.0 --share_EC_s1_s2;
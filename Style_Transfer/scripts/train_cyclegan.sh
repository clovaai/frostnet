set -ex
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan --pool_size 50 --no_dropout --lr 0.0001 --netG resnet_9blocks --display_port 60066 --q_optim True --clip_by 0.001 --print_freq 100 --fp_warmup 1 --norm instance  --save_epoch_freq 1  --display_env cycleGAN --display_freq 100 --display_id 2 --batch_size 2 --n_epochs 100 --n_epochs_decay 100

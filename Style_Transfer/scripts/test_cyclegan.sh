set -ex
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --phase test --no_dropout --gpu_ids -1 --norm instance
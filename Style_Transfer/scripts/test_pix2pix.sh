set -ex
python test.py --dataroot ./datasets/maps --name maps_pix2pix --model pix2pix --netG resnet_9blocks --direction BtoA --dataset_mode aligned --norm instance --gpu_ids -1

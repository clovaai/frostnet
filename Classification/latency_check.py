import torch
import glob
import os
from argparse import ArgumentParser
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
from utilities.print_utils import *
from utilities.data_transforms import MEAN, STD
from utilities.utils import model_parameters, compute_flops
import torch.quantization
from utilities.train_eval_seg import train_seg_one_iter as train
from utilities.train_eval_seg import val_seg_latency as val
from utilities.optimizer import *
from loss_fns.segmentation_loss import SegmentationLoss
os.environ['CUDA_VISIBLE_DEVICES'] = ""
num_gpu = 0

def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img


def data_transform(img, crop_size):
    img = img.resize(crop_size, Image.BILINEAR)
    img = F.to_tensor(img)  # convert to tensor (values between 0 and 1)
    img = F.normalize(img, MEAN, STD)  # normalize the tensor
    return img


def evaluate(args, model, image_list, device):
    crop_size = tuple(args.crop_size)

    # get color map for pascal dataset
    if args.dataset == 'pascal':
        from utilities.color_map import VOCColormap
        cmap = VOCColormap().get_color_map_voc()
    else:
        from utilities.color_map import CityColormap
        cmap = CityColormap().get_color_map_city()

    model.eval()
    for i, imgName in tqdm(enumerate(image_list)):
        img = Image.open(imgName).convert('RGB')
        w, h = img.size

        img = data_transform(img, crop_size)
        img = img.unsqueeze(0)  # add a batch dimension
        img = img.to(device)
        img_out = model(img)
        img_out = img_out.squeeze(0)  # remove the batch dimension
        img_out = img_out.max(0)[1].byte()  # get the label map
        img_out = img_out.to(device='cpu').numpy()

        if args.dataset == 'city':
            # cityscape uses different IDs for training and testing
            # so, change from Train IDs to actual IDs
            img_out = relabel(img_out)

        img_out = Image.fromarray(img_out)
        # resize to original size
        img_out = img_out.resize((w, h), Image.NEAREST)

        # pascal dataset accepts colored segmentations
        #if args.dataset == 'pascal':
        #    img_out.putpalette(cmap)
        img_out.putpalette(cmap)
        # save the segmentation mask
        name = imgName.split('/')[-1]
        img_extn = imgName.split('.')[-1]
        name = '{}/{}'.format(args.savedir, name.replace(img_extn, 'png'))
        img_out.save(name)
        


def main(args):
    # read all the images in the folder
    crop_size = args.crop_size
    if args.dataset == 'city':
        image_path = os.path.join(args.data_path, "leftImg8bit", args.split, "*", "*.png")
        image_list = glob.glob(image_path)        
        from data_loader.segmentation.cityscapes import CityscapesSegmentation, CITYSCAPE_CLASS_LIST
        train_dataset = CityscapesSegmentation(root=args.data_path, train=True, size=crop_size, scale=args.scale, coarse=False)
        val_dataset = CityscapesSegmentation(root=args.data_path, train=False, size=crop_size, scale=args.scale,
                                             coarse=False)
        seg_classes = len(CITYSCAPE_CLASS_LIST)
        class_wts = torch.ones(seg_classes)
    elif args.dataset == 'pascal':
        from data_loader.segmentation.voc import VOCSegmentation, VOC_CLASS_LIST
        train_dataset = VOCSegmentation(root=args.data_path, train=True, crop_size=crop_size, scale=args.scale,
                                        coco_root_dir=args.coco_path)
        val_dataset = VOCSegmentation(root=args.data_path, train=False, crop_size=crop_size, scale=args.scale)
        seg_classes = len(VOC_CLASS_LIST)
        class_wts = torch.ones(seg_classes)
        data_file = os.path.join(args.data_path, 'VOC2012', 'list', '{}.txt'.format(args.split))
        if not os.path.isfile(data_file):
            print_error_message('{} file does not exist'.format(data_file))
        image_list = []
        with open(data_file, 'r') as lines:
            for line in lines:
                rgb_img_loc = '{}/{}/{}'.format(args.data_path, 'VOC2012', line.split()[0])
                if not os.path.isfile(rgb_img_loc):
                    print_error_message('{} image file does not exist'.format(rgb_img_loc))
                image_list.append(rgb_img_loc)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))

    if len(image_list) == 0:
        print_error_message('No files in directory: {}'.format(image_path))

    print_info_message('# of images for testing: {}'.format(len(image_list)))

    if args.model == 'espnetv2':
        from model.espnetv2 import espnetv2_seg
        args.classes = seg_classes
        model = espnetv2_seg(args)
    elif args.model == 'espnet':
        from model.espnet import espnet_seg
        args.classes = seg_classes
        model = espnet_seg(args)
    elif args.model == 'mobilenetv2_1_0':
        from model.mobilenetv2 import get_mobilenet_v2_1_0_seg
        args.classes = seg_classes
        model = get_mobilenet_v2_1_0_seg(args)
    elif args.model == 'mobilenetv2_0_35':
        from model.mobilenetv2 import get_mobilenet_v2_0_35_seg
        args.classes = seg_classes
        model = get_mobilenet_v2_0_35_seg(args)   
    elif args.model == 'mobilenetv2_0_5':
        from model.mobilenetv2 import get_mobilenet_v2_0_5_seg
        args.classes = seg_classes
        model = get_mobilenet_v2_0_5_seg(args)           
    elif args.model == 'mobilenetv3_small':
        from model.mobilenetv3 import get_mobilenet_v3_small_seg
        args.classes = seg_classes
        model = get_mobilenet_v3_small_seg(args) 
    elif args.model == 'mobilenetv3_large':
        from model.mobilenetv3 import get_mobilenet_v3_large_seg
        args.classes = seg_classes
        model = get_mobilenet_v3_large_seg(args)
    elif args.model == 'mobilenetv3_RE_small':
        from model.mobilenetv3 import get_mobilenet_v3_RE_small_seg
        args.classes = seg_classes
        model = get_mobilenet_v3_RE_small_seg(args) 
    elif args.model == 'mobilenetv3_RE_large':
        from model.mobilenetv3 import get_mobilenet_v3_RE_large_seg
        args.classes = seg_classes
        model = get_mobilenet_v3_RE_large_seg(args)                
    else:
        print_error_message('Arch: {} not yet supported'.format(args.model))
        exit(-1)      

    # model information
    train_params = []
    params_dict = dict(model.named_parameters()) 
    others= args.weight_decay*0.01    
    for key, value in params_dict.items():
        if len(value.data.shape) == 4:
            if value.data.shape[1] == 1:
                train_params += [{'params': [value], 'lr': args.lr, 'weight_decay': 0.0}]
                # names_set.append(key)
            else:
                train_params += [{'params': [value], 'lr': args.lr, 'weight_decay': args.weight_decay}]
        else:
            train_params += [{'params': [value],'lr': args.lr, 'weight_decay': others}]  
    args.learning_rate = args.lr
    optimizer = get_optimizer(args.optimizer, train_params, args)
    
    num_params = model_parameters(model)
    flops = compute_flops(model, input=torch.Tensor(1, 3, args.crop_size[0], args.crop_size[1]))
    print_info_message('FLOPs for an input of size {}x{}: {:.2f} million'.format(args.crop_size[0], args.crop_size[1], flops))
    print_info_message('# of parameters: {}'.format(num_params))
    

    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')
        
    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu' 
    
    model = model.to(device=device)
        
    print("========== MODEL CALIBRATION ===========") 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               pin_memory=True, num_workers=args.workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=args.workers,drop_last=True)
    criterion = SegmentationLoss(n_classes=seg_classes, loss_type=args.loss_type,
                                 device=device, ignore_idx=args.ignore_idx,
                                 class_wts=class_wts.to(device))    
   
    print('========== ORIGINAL MODEL SIZE ==========')
    print_size_of_model(model)
    model.quantized.fuse_model()
    model.quantized.qconfig =  torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model.quantized, inplace=True)
    qat_miou_val, qat_val_loss = val(model, val_loader, criterion, seg_classes, device=device)  
    print("========== QUANTIZED MODEL SIZE ==========")
    torch.quantization.convert(model.quantized.eval(),inplace = True)       
    print_size_of_model(model)
    miou_val, val_loss = val(model, val_loader, criterion, seg_classes, device=device)    
    evaluate(args, model, image_list, device=device)

    
if __name__ == '__main__':
    from commons.general_details import segmentation_models, segmentation_schedulers, segmentation_loss_fns, \
        segmentation_datasets

    parser = ArgumentParser()
    # mdoel details
    parser.add_argument('--model', default="espnetv2", choices=segmentation_models, help='Model name')
    parser.add_argument('--weights_test', default='', help='Pretrained weights directory.')
    parser.add_argument('--s', default=2.0, type=float, help='scale')
    # dataset details
    parser.add_argument('--data_path', default="", help='Data directory')
    parser.add_argument('--dataset', default='city', choices=segmentation_datasets, help='Dataset name')
    # input details
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='data split')
    parser.add_argument('--model_width', default=224, type=int, help='Model width')
    parser.add_argument('--model_height', default=224, type=int, help='Model height')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num_classes', default=1000, type=int,
                        help='ImageNet classes. Required for loading the base network')
    parser.add_argument('--p', default=2, type=int, help='depth multiplier')
    parser.add_argument('--q', default=8, type=int, help='depth multiplier')
    
    #additional
    # scheduler details
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--ignore_idx', type=int, default=255, help='Index or label to be ignored during training')    
    parser.add_argument('--optimizer', default='QSGD', 
                        help='Optimizers')    
    parser.add_argument('--nesterov', action='store_true', default=True, help='nesterov')
    parser.add_argument('--amsgrad', action='store_true', default=False, help='amsgrad')    
    parser.add_argument('--clip_by', default=1e-3, type=float, help='clip_by')
    parser.add_argument('--noise_decay', default=1e-2, type=float, help='noise_decay') 
    parser.add_argument('--toss_coin', action='store_true', default=True, help='toss_coin')    
    parser.add_argument('--scheduler', default='hybrid', choices=segmentation_schedulers,
                        help='Learning rate scheduler (fixed, clr, poly)')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--step_size', default=51, type=int, help='steps at which lr should be decreased')
    parser.add_argument('--lr', default=1e-2, type=float, help='initial learning rate')
    parser.add_argument('--lr_mult', default=1e-2, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float, help='factor by which lr should be decreased')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=4e-5, type=float, help='weight decay (default: 4e-5)')
    # for Polynomial LR
    parser.add_argument('--power', default=0.9, type=float, help='power factor for Polynomial LR')

    # for hybrid LR
    parser.add_argument('--clr_max', default=61, type=int, help='Max number of epochs for cylic LR before '
                                                                'changing last cycle to linear')
    parser.add_argument('--cycle_len', default=5, type=int, help='Duration of cycle')

    # input details
    parser.add_argument('--batch_size', type=int, default=40, help='list of batch sizes')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[512, 256],
                        help='list of image crop sizes, with each item storing the crop size (should be a tuple).')
    parser.add_argument('--loss_type', default='ce', choices=segmentation_loss_fns, help='Loss function (ce or miou)')
    parser.add_argument('--fp_epoch', default=1, type=int, help='FP warmup epoch')   
  
    
    args = parser.parse_args()

    if not args.weights_test:
        from model.weight_locations.segmentation import model_weight_map

        model_key = '{}_{}'.format(args.model, args.s)
        dataset_key = '{}_{}x{}'.format(args.dataset, args.crop_size[0], args.crop_size[1])
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        assert dataset_key in model_weight_map[model_key].keys(), '{} does not exist'.format(dataset_key)
        args.weights_test = model_weight_map[model_key][dataset_key]['weights']
        if not os.path.isfile(args.weights_test):
            print_error_message('weight file does not exist: {}'.format(args.weights_test))

    if args.dataset == 'pascal':
        args.scale = (0.5, 2.0)
        args.crop_scale = 0.25
        
    elif args.dataset == 'city':
        if args.crop_size[0] == 512:
            args.crop_scale = 0.25
            args.scale = (0.25, 0.5)
        elif args.crop_size[0] == 1024:
            args.crop_scale = 0.5
            args.scale = (0.35, 1.0)  # 0.75 # 0.5 -- 59+
        elif args.crop_size[0] == 2048:
            args.crop_scale = 1.0
            args.scale = (1.0, 2.0)
        else:
            print_error_message('Select image size from 512x256, 1024x512, 2048x1024')
        print_log_message('Using scale = ({}, {})'.format(args.scale[0], args.scale[1]))
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))
    args.crop_size = tuple(args.crop_size)        
    # set-up results path
    if args.dataset == 'city':
        args.savedir = '{}_{}_{}/results'.format('results', args.dataset, args.split)
    elif args.dataset == 'pascal':
        args.savedir = '{}_{}/results/VOC2012/Segmentation/comp6_{}_cls'.format('results', args.dataset, args.split)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))

    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    # This key is used to load the ImageNet weights while training. So, set to empty to avoid errors
    args.weights = ''

    main(args)

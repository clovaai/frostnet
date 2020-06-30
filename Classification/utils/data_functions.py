import requests
import torchvision
import torchvision.transforms as transforms
import os
import torch
import torchvision.datasets as datasets


import torchvision
import torchvision.datasets as datasets

def download_data(dataset_name='imagenet_tiny',datapath = '~/../data/'):
    if dataset_name == 'imagenet_tiny':
        #url = 'https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip'
        #filename = datapath + 'imagenet_1k/imagenet_1k_data.zip'

        #r = requests.get(url)
        #with open(filename, 'wb') as f:
        #    f.write(r.content)
            
        traindir = os.path.join(datapath+'imagenet_1k/', 'train')
        valdir = os.path.join(datapath+'imagenet_1k/', 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        dataset = torchvision.datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

        dataset_test = torchvision.datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))            
            
    elif dataset_name == 'imagenet':
        dataset = torchvision.datasets.ImageNet(
                             datapath + 'imagenet',
                             split='train',
                             download=True,
                            transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                            ])) 
            
        dataset_test = torchvision.datasets.ImageNet(
                             datapath + 'imagenet',
                             split='val',
                             download=True,
                            transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                            ])) 
    elif dataset_name == 'ILSVRC2015':
        datapath = os.path.join(datapath, 'train/Data/CLS-LOC')
        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'val')
        dataset = datasets.ImageFolder(
                            traindir,
                            transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                            ])) 
            
        dataset_test = datasets.ImageFolder(
                            valdir,
                            transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                            ]))

        elif dataset_name == 'cifar100':
            dataset = torchvision.datasets.CIFAR100(
                             datapath + 'cifar100',
                             train = True,
                             download=False,
                            transform = transforms.Compose([
                            transforms.RandomResizedCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                                 std=[0.2673, 0.2564, 0.2762])
                            ]))
            dataset_test = torchvision.datasets.CIFAR100(
                             datapath + 'cifar100',
                             train = False,
                             download=False,
                            transform = transforms.Compose([
                            #transforms.RandomResizedCrop(224),
                            #transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                                 std=[0.2673, 0.2564, 0.2762])
                            ]))   
            
    elif dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
                             datapath + 'cifar10',
                             train = True,
                             download=True,
                            transform = transforms.Compose([
                            transforms.RandomResizedCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                 std=[0.247, 0.243, 0.261])
                            ]))
        dataset_test = torchvision.datasets.CIFAR10(
                             datapath + 'cifar10',
                             train = False,
                             download=True,
                            transform = transforms.Compose([
                            #transforms.RandomResizedCrop(224),
                            #transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                 std=[0.247, 0.243, 0.261])
                            ]))  
          
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
                             datapath + 'cifar100',
                             train = True,
                             download=True,
                            transform = transforms.Compose([
                            transforms.RandomResizedCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                                 std=[0.2673, 0.2564, 0.2762])
                            ]))
        dataset_test = torchvision.datasets.CIFAR100(
                             datapath + 'cifar100',
                             train = False,
                             download=True,
                            transform = transforms.Compose([
                            #transforms.RandomResizedCrop(224),
                            #transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                                 std=[0.2673, 0.2564, 0.2762])
                            ]))        
    elif dataset_name == 'svhn':
        dataset = torchvision.datasets.SVHN(
                             datapath + 'svhn',
                             split='train',
                             download=True,
                            transform = transforms.Compose([
                            transforms.RandomResizedCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                                 std=[0.1980, 0.2010, 0.1970]),
                            ])) 
        dataset_test = torchvision.datasets.SVHN(
                             datapath + 'svhn',
                             split='test',
                             download=True,
                            transform = transforms.Compose([
                            #transforms.RandomResizedCrop(32),
                            #transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                                 std=[0.1980, 0.2010, 0.1970]),
                            ]))            
    elif dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST(
                             datapath + 'mnist',
                             train=True,
                             download=True,
                            transform = transforms.Compose([
                            #transforms.RandomResizedCrop(28),
                            #transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           #                      std=[0.229, 0.224, 0.225]),
                            ]))  
        dataset_test = torchvision.datasets.MNIST(
                             datapath + 'mnist',
                             train=False,
                             download=True,
                            transform = transforms.Compose([
                            #transforms.RandomResizedCrop(28),
                            #transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           #                      std=[0.229, 0.224, 0.225]),
                            ]))             
    return dataset, dataset_test

def get_torch_dataloader(data_path):
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ])),
        batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    def batch_fn(batch):
        return batch[0], batch[1]

    return train_loader, val_loader, batch_fn

def prepare_data_loaders(dataset, dataset_test, train_batch_size, eval_batch_size):
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test            
import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import clip_w_local


def set_model_clip(args):
    model, _ = clip_w_local.load(args.CLIP_ckpt)

    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    return model, val_preprocess


def set_val_loader(args, preprocess=None):
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "imagenet":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.root, 'imagenet/images/val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        raise NotImplementedError
    return val_loader


def set_ood_loader_ImageNet(args, out_dataset, preprocess=None):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    if out_dataset == 'iNaturalist':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'Places'), transform=preprocess)
    elif out_dataset == 'Texture':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'Texture', 'images'),
                                          transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                                shuffle=False, num_workers=4)
    return testloaderOut

def set_id_ood_loader(args, in_dataset, out_dataset, preprocess, noise=False, gaussian_rate=0.125, inject_noise_type="gaussian"):
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    testsetin = datasets.ImageFolder(os.path.join(args.root, 'imagenet/images/val'), transform=preprocess)
    if out_dataset == 'iNaturalist':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'Places'), transform=preprocess)
    elif out_dataset == 'Texture':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'Texture', 'images'),
                                          transform=preprocess)
    
    testsetout = FixedOODTargetDataset(dataset=testsetout, fixed_target=args.class_num)
    testset = torch.utils.data.ConcatDataset([testsetin, testsetout])
    
    noise_size = len(testset) * gaussian_rate
    #print(len(testsetout), len(testset), noise_size)
    #breakpoint()
    if noise:
        noise_data = NoiseDataset(transform=preprocess, data_size=noise_size, fixed_target=-1000, is_carry_index=False, noise_type=inject_noise_type)
        testset = torch.utils.data.ConcatDataset([testset, noise_data])
        
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, **kwargs)
    #print(args.batch_size)
    #breakpoint()
    return testloader

import numpy as np
from PIL import Image
import torch.utils.data
from torchvision import datasets
from torchvision.transforms import ToPILImage

class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, transform, data_size, fixed_target=-1000, ratio=1, is_carry_index=False, noise_type='gaussian'):
        self.noise_type = noise_type
        self.number = int(data_size * ratio) # 50000
        self.fixed_target = fixed_target
        self.transform = transform
        self.is_carry_index = is_carry_index
        self.to_pil = ToPILImage()

    def __getitem__(self, index:int):
        if self.noise_type == 'gaussian':
            image = torch.randn(3, 224, 224)
        elif self.noise_type  == 'uniform':
            image = torch.rand(3, 224, 224)
        elif self.noise_type == 'salt_and_pepper':
            image = self._salt_and_pepper_noise(torch.zeros(3, 224, 224))
        elif self.noise_type == 'poisson':
            image = self._poisson_noise(torch.ones(3, 224, 224))
        else:
            raise NotImplementedError
        target = self.fixed_target
        if self.transform is not None:
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.byte()
            image = self.to_pil(image)
            image = self.transform(image)
        if self.is_carry_index:
            if type(image) == list:
                    image.append(index)
            else:
                image = [image, index]

        return image, target

    def _salt_and_pepper_noise(self, image):
        prob = 0.05
        rnd = torch.rand(image.shape)
        salt = (rnd < prob/2).float()
        pepper = ((rnd >= prob/2) & (rnd < prob)).float()
        image = image * ((~salt.bool()) & (~pepper.bool())).float() + salt * 255
        return image
    
    def _poisson_noise(self, image):
        # Scale the image by the noise parameter
        noise_param = 1.0
        scaled_image = image * noise_param
        # Generate Poisson noise
        noisy_image = torch.poisson(scaled_image)
        # Normalize back to the original range
        noisy_image = noisy_image / noise_param
        return noisy_image
    
    def __len__(self):
        return self.number
    
class FixedOODTargetDataset(torch.utils.data.Dataset):
    """A dataset wrapper that sets a fixed target value for all samples."""
    def __init__(self, dataset, fixed_target):
        self.dataset = dataset
        self.fixed_target = fixed_target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, _ = self.dataset[idx]  # Ignore original target
        return data, self.fixed_target     
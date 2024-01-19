# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Paolo Dimasi, Politecnico di Torino. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
# from lib.fpn.box_utils import intersection_area
from utils.misc import interpolate, NestedTensor

from typing import Tuple, List, Dict



def rand_bbox(size, alpha):
    W = size[2]
    H = size[3]
    cut_ratio = torch.sqrt(1. - alpha)
    
    # use same notation from  https://arxiv.org/abs/1905.04899
    rw_half = int(W * cut_ratio *0.5 )
    rh_half = int(H * cut_ratio *0.5 )

    # uniform
    rcx = torch.randint(W)
    rcy = torch.randint(H)

    # cxcywwh bbox coords format
    bbx1 = torch.clamp(rcx - rw_half, 0, W)
    bby1 = torch.clamp(rcy - rh_half, 0, H)
    bbx2 = torch.clamp(rcx + rw_half, 0, W)
    bby2 = torch.clamp(rcy + rh_half, 0, H)

    return bbx1, bby1, bbx2, bby2   





def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, _ = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area
        
      

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between n trasforms uniformly
    """
    def __init__(self, *args):
        self.transform= [ a for a in args]
        self.n = len(args)
    def __call__(self, img, target):
        return self.transform[random.randint(0, self.n-1)](img,target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, deterministic:bool=True):
        
        self.deterministic = deterministic
        
        if self.deterministic:
            self.bright=brightness 
            self.contrast = contrast
            self.sat = saturation 
            self.hue= hue
        else:
            self.bright = torch.distributions.uniform.Uniform(max(0, 1 - brightness), 1 + brightness)
            self.contrast =  torch.distributions.uniform.Uniform(max(0, 1 - contrast), 1 + contrast)
            self.sat = torch.distributions.uniform.Uniform(max(0, 1 - saturation), 1 + saturation)
            self.hue = torch.distributions.uniform.Uniform(0, hue)
            

    def __call__(self, image, target=None):
        if self.deterministic:
            bright_val = self.bright
            contrast_val =self.contrast
            sat_val = self.sat  
            hue_val = self.hue
            
        else:
            bright_val = self.bright.sample().item()
            contrast_val = self.contrast.sample().item()
            sat_val = self.sat.sample().item()
            hue_val = self.hue.sample().item()
            
            
        image = F.adjust_brightness(image,bright_val)
        image = F.adjust_contrast(image, contrast_val)
        image = F.adjust_saturation(image,sat_val)
        image = F.adjust_hue(image, hue_val)

        return image, target
    
class  JitterBbox(object):
    def __init__(self, brightness=0, contrast=0):
        self.bright=brightness 
        self.contrast = contrast


    def __call__(self, image, target=None):
        
        # select bounding box to remove which is not 
        # involved in relationship
        n_bb = target['boxes'].size()[0]
        
       
        assert n_bb !=0, 'each img must have at least one bounding box'
        # print(target['boxes'])
        # print(f'nn bb is {n_bb}')
        
        if n_bb>0:
            bb_id1 = random.randint(0, n_bb-1 )
            bb_id2 = random.randint(0, n_bb-1 )
        else:
            bb_id1 = bb_id2 = 0


        bb1 = torch.Tensor.int(target['boxes'][bb_id1])
        bb2 = torch.Tensor.int(target['boxes'][bb_id2])
        
        
        image[:, bb1[0]:bb1[2],bb1[1]:bb1[3]] = F.adjust_brightness(image[:, bb1[0]:bb1[2],bb1[1]:bb1[3]], self.bright)
        image[:, bb2[0]:bb2[2],bb2[1]:bb2[3]] = F.adjust_contrast(image[:, bb2[0]:bb2[2],bb2[1]:bb2[3]], self.contrast)

        return image, target
    
class Gamma(object):
    def __init__(self, gain=1):
        self.gain = gain

    def __call__(self, image, target=None):
        image = F.adjust_gamma(image, self.gain)

        return image, target
    
  

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


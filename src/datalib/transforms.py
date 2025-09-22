import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
import time

class ParameterCompose:
    def __init__(self, transforms):
        self.transforms = transforms
        self.should_hflip = None
        self.should_vflip = None

    def __call__(self, rgbs, masks, flows, coord):
        
        # Make sequence decisions only once (on first frame of sequence)
        if self.should_hflip is None or self.should_vflip is None:
            self._make_sequence_decisions()
        
        for t in self.transforms:
            if isinstance(t, RandomHorizontalFlip):
                rgbs, masks, flows, coord = t(rgbs, masks, flows, coord, self.should_hflip)
            elif isinstance(t, RandomVerticalFlip):
                rgbs, masks, flows, coord = t(rgbs, masks, flows, coord, self.should_vflip)
            else: 
                rgbs, masks, flows, coord = t(rgbs, masks, flows, coord)
        if(len(self.transforms)==0):
            return {}, {}, {}, {}
        return rgbs, masks, flows, coord

    def _make_sequence_decisions(self):
        """Make random decisions for this sequence"""

        # Using current time for variety
        random_seed = int(time.time() * 1000) % 10000
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.should_hflip = random.random() < 0.5
        self.should_vflip = random.random() < 0.3
        
        
    def reset_sequence(self):
        """Is called when starting a new sequence to make new random decisions"""
        self.should_hflip = None
        self.should_vflip = None
        
        
        
class RandomVerticalFlip:
    """
    Flips the image frame vertically for the entire sequence.
    """
    
    def __init__(self):
        pass

    def __call__(self, rgbs, masks, flows, coord, should_vflip):
        
        if should_vflip:
            rgbs = F.vflip(rgbs)
            flows = F.vflip(flows)
            masks['mask'] = F.vflip(masks['mask'])

            Height = rgbs.shape[1]  # 128
            
            # Step 1: Adjust bounding box coordinates
            flipped_bboxes = coord['bbox'].clone()  # Create a copy to avoid modifying the original
            flipped_bboxes[:, 1] = Height - coord['bbox'][:, 3]  # New y_min = H - y_max
            flipped_bboxes[:, 3] = Height - coord['bbox'][:, 1]  # New y_max = H - y_min
            # x_min (index 0) and x_max (index 2) remain unchanged
            
            # Optional: Ensure coordinates are within bounds [0, H]
            flipped_bboxes = torch.clamp(flipped_bboxes, 0, Height)

            coord['com'][:, 1] = Height - coord['com'][:, 1]
            # Optional: Ensure coms are within bounds [0, H]
            coord['com'] = torch.clamp(coord['com'], 0, Height)
            coord['bbox'] = flipped_bboxes
            return rgbs, masks, flows, coord

        return rgbs, masks, flows, coord

class RandomHorizontalFlip:
    """
    Flips the image frame horizontally for the entire sequence.
    """
    
    def __init__(self):
        pass
    
    def __call__(self, rgbs, masks, flows, coord, should_hflip):
        
        if should_hflip:
            rgbs = F.hflip(rgbs)
            flows = F.hflip(flows)
            masks['mask'] = F.hflip(masks['mask'])

            Width = rgbs.shape[2]  # 128
            
            # Step 1: Adjust bounding box coordinates
            flipped_bboxes = coord['bbox'].clone()  # Create a copy to avoid modifying the original
            flipped_bboxes[:, 0] = Width - coord['bbox'][:, 2]  # New x_min = W - x_max
            flipped_bboxes[:, 2] = Width - coord['bbox'][:, 0]  # New x_max = W - x_min
            # y_min (index 1) and y_max (index 3) remain unchanged
            
            # Optional: Ensure coordinates are within bounds [0, W]
            flipped_bboxes = torch.clamp(flipped_bboxes, 0, Width)

            coord['com'][:, 0] = Width - coord['com'][:, 0]
            # Optional: Ensure coms are within bounds [0, W]
            coord['com'] = torch.clamp(coord['com'], 0, Width)
            coord['bbox'] = flipped_bboxes
            return rgbs, masks, flows, coord

        return rgbs, masks, flows, coord


class NormalizeTransform:
    """
    Normalize images and flows by dividing by 255 to convert from [0, 255] to [0, 1] range.
    """
    def __init__(self):
        pass
    
    def __call__(self, rgbs, masks, flows, coord):
        # Normalize RGB and flow by dividing by 255
        rgbs = rgbs.float() / 255.0
        flows = flows.float() / 255.0
        
        # coord and masks remain unchanged
        return rgbs, masks, flows, coord


class ResizeTransform:
    """
    Resize images to a specific size.
    """
    def __init__(self, size=(128, 128)):
        self.size = size
    
    def __call__(self, rgbs, masks, flows, coord):
        rgbs = F.resize(rgbs, self.size)
        flows = F.resize(flows, self.size)
        masks['mask'] = F.resize(masks['mask'].unsqueeze(0), self.size).squeeze(0)  # Add channel dim for resize, then remove
        
        # Scale bbox and com coordinates to match new size
        old_h, old_w = rgbs.shape[1], rgbs.shape[2]
        new_h, new_w = self.size
        
        if old_h != new_h or old_w != new_w:
            scale_y = new_h / old_h
            scale_x = new_w / old_w
            
            # Scale bounding boxes
            bbox_scaled = coord['bbox'].clone()
            bbox_scaled[:, 0] *= scale_x  # x_min
            bbox_scaled[:, 1] *= scale_y  # y_min
            bbox_scaled[:, 2] *= scale_x  # x_max
            bbox_scaled[:, 3] *= scale_y  # y_max
            
            # Scale center of mass coordinates
            com_scaled = coord['com'].clone()
            com_scaled[:, 0] *= scale_x  # x coordinate
            com_scaled[:, 1] *= scale_y  # y coordinate
            
            coord['bbox'] = bbox_scaled
            coord['com'] = com_scaled
            return rgbs, masks, flows, coord
        
        return rgbs, masks, flows, coord


def get_train_transforms():
    """
    Get augmentation transforms for training
    """
    return ParameterCompose([
        ResizeTransform(size=(128, 128)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        NormalizeTransform()
    ])


def get_validation_transforms():
    """
    Get transforms for validation (no augmentation)
    """
    return ParameterCompose([
        ResizeTransform(size=(128, 128)),
        NormalizeTransform()
    ])


import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import io
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')

class MOVIC(Dataset):

    def __init__(self, data_path, split='train' ,transforms=None):
        data_directory=os.path.join(data_path, split)
        if not os.path.exists(data_directory):
            if not os.path.exists(os.path.abspath(data_directory)):
                raise Exception("Dataset was not found!")
        
        number_of_frames_per_video=24
        
        self.rgbs = self.collect_files(data_directory, 'rgb*.png', group_size=24)
        self.flows = self.collect_files(data_directory, 'flow*.png', group_size=24)
        self.coords = self.collect_files(data_directory, 'coords*.pt')
        self.masks = self.collect_files(data_directory, 'mask*.pt')
        
        # Store transforms
        self.transforms = transforms

        assert len(self.rgbs) == len(self.flows) == len(self.coords) == len(self.masks), "Data and annotations need to be of the same size"

        logging.info(f"{split.upper()} Data Loaded: Coordinates: {len(self.coords)}, Masks: {len(self.masks)}, RGB videos:  {len(self.rgbs)}, Flows:  {len(self.flows)}")

    def __getitem__(self, idx):
        # return self.coord[idx], self.mask[idx], self.rgb[idx], self.flow[idx]
 
        rgb_paths = self.rgbs[idx]  # list of frame paths for the video frames = idx
        flow_paths = self.flows[idx]
          
        rgbs =  torch.stack([io.read_image(p).to(torch.float32) for p in rgb_paths])
        flows =  torch.stack([io.read_image(p).to(torch.float32) for p in flow_paths])
        
        coords = torch.load(self.coords[idx], map_location="cpu")  # loaded lazily
        masks = torch.load(self.masks[idx], map_location="cpu")
        
        # Apply transforms if provided
        if self.transforms:
            
            # reset for new sequence
            self.transforms.reset_sequence()
            
            # Apply transforms to each frame
            transformed_frames = []
            for i in range(rgbs.shape[0]):  # iterate over frames (24 frames)
                # Extract center of mass and bounding boxes for this frame
                com_frame = coords['com'][i] if 'com' in coords else torch.zeros(11, 2)
                bbox_frame = coords['bbox'][i] if 'bbox' in coords else torch.zeros(11, 4)
                mask_frame = masks['masks'][i]
                rgb_frame = rgbs[i]
                flow_frame = flows[i]
                
                # Prepare data in new format for transforms
                masks_frame = {'mask': mask_frame}
                coord_frame = {'com': com_frame, 'bbox': bbox_frame}
                
                # Apply transforms with new signature
                rgb_t, masks_t, flow_t, coord_t = self.transforms(
                    rgb_frame, masks_frame, flow_frame, coord_frame
                )
                
                transformed_frames.append((coord_t['com'], coord_t['bbox'], masks_t['mask'], rgb_t, flow_t))
            
            # Reconstruct the tensors
            coms = torch.stack([frame[0] for frame in transformed_frames])
            bboxes = torch.stack([frame[1] for frame in transformed_frames])
            masks_dict = {'masks': torch.stack([frame[2] for frame in transformed_frames])}
            rgbs = torch.stack([frame[3] for frame in transformed_frames])
            flows = torch.stack([frame[4] for frame in transformed_frames])
            
            # Update coords dictionary and masks
            coords = {'com': coms, 'bbox': bboxes}
            masks = masks_dict
        
        return rgbs, masks, flows, coords
    
    def collect_files(self, data_directory, condition, group_size=None):
        """
        Collect files matching a pattern and optionally group them by video.
        """
        files = sorted(glob.glob(os.path.join(data_directory, condition)))
        
        if group_size:  # group into videos
            files = [files[i:i+group_size] for i in range(0, len(files), group_size)]
        
        return files


    def __len__(self):
        return len(self.masks)


    def get_video_frame_labels(self, com, bbox, masks, rgbs, flows):
        '''
        outputs 24 entries, each of which corresponds to data of a frame data.
        '''
        output=[]
        for i in range(com.shape[0]):
            output.append((com[i], bbox[i], masks[i], rgbs[i], flows[i]))
        
        return output 
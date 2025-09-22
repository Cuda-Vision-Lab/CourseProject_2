from .MoviC import MOVIC
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from CONFIG import config
from .transforms import get_train_transforms, get_validation_transforms

def load_data(path, split="train", use_transforms=True):
    """
    Loading a dataset given the parameters

    Args:
    -----
    path: string
        path to the dataset to load
    split: string
        Split from the dataset to obtain (e.g., 'train' or 'test')
    use_transforms: bool
        Whether to apply transforms (includes sequence-consistent augmentations for training)

    Returns:
    --------
    dataset: torch dataset
        Dataset loaded given specifications from exp_params
    """
    # Get appropriate transforms based on split
    num_epochs = config['training']['num_epochs']
    
    if use_transforms:
        if split == "train":
            transforms = get_train_transforms(base_seed=42)
        else:
            transforms = get_validation_transforms(base_seed=42)
    else:
        transforms = None
        
    #  dataset = MOVIC(path, split=split, transforms=None)
    dataset = MOVIC(path, split=split, transforms=transforms, num_epochs=num_epochs)
    return dataset

# @log_function
def build_data_loader(dataset, split='train'):
    """
    Fitting a data loader for the given dataset

    Args:
    -----
    dataset: torch dataset
        Dataset (or dataset split) to fit to the DataLoader
    batch_size: integer
        number of elements per mini-batch
    shuffle: boolean
        If True, mini-batches are sampled randomly from the database
    """

    batch_size = config['data']['batch_size']
    shuffle = config['training'][split]['shuffle']
    num_workers = config['data']["num_workers"]

    data_loader = DataLoader(
                             dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers
                            )

    return data_loader
import os
import numpy as np
import torch.nn as nn
import torch
import random
import json
from matplotlib import pyplot as plt


from torch.utils.tensorboard import SummaryWriter

    
def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = 42
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return

def init_weights(m):
    # initialize nn.Linear and nn.LayerNorm
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
            


def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f


def save_model(model, optimizer, epoch, stats, experiment_name):
    """ Saving model checkpoint """
    
    if(not os.path.exists("checkpoints")):
        os.makedirs("checkpoints")
    savepath = f"checkpoints/checkpoint_{experiment_name}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    return

def save_config(config, exp_path):
    """
    Saves the config (dict or config.py module) as a JSON file in the given path.

    Args:
        config: Either a config dictionary or a config.py module (with attributes).
        path: The file path where the JSON should be saved.
    """

    # Ensure the directory exists
    dir_name = os.path.join(exp_path,'config')
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    with open(dir_name, "w") as f:
        json.dump(config, f, indent=4)
    return

def load_model(model, optimizer, savepath):
    """ Loading pretrained checkpoint """
    
    checkpoint = torch.load(savepath, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    
    return model, optimizer, epoch, stats


def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def create_directory(exp_path):
    """
    Creating a folder in given path.
    """
    # if(not os.path.exists(exp_path)):
    os.makedirs(exp_path,exist_ok=True)
    # if(dir_name is not None):
    #     dir_path = os.path.join(dir_path, dir_name)
    return

class TensorboardWriter:
    """
    Class for handling the tensorboard logger

    Args:
    -----
    logdir: string
        path where the tensorboard logs will be stored
    """

    def __init__(self, logdir):
        """ Initializing tensorboard writer """
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        return

    def add_scalar(self, name, val, step):
        """ Adding a scalar for plot """
        self.writer.add_scalar(name, val, step)
        return

    def add_scalars(self, plot_name, val_names, vals, step):
        """ Adding several values in one plot """
        val_dict = {val_name: val for (val_name, val) in zip(val_names, vals)}
        self.writer.add_scalars(plot_name, val_dict, step)
        return

    def add_image(self, fig_name, img_grid, step):
        """ Adding a new step image to a figure """
        self.writer.add_image(fig_name, img_grid, global_step=step)
        return

    def add_figure(self, tag, figure, step):
        """ Adding a whole new figure to the tensorboard """
        self.writer.add_figure(tag=tag, figure=figure, global_step=step)
        return

    def add_graph(self, model, input):
        """ Logging model graph to tensorboard """
        self.writer.add_graph(model, input_to_model=input)
        return

    def log_full_dictionary(self, dict, step, plot_name="Losses", dir=None):
        """
        Logging a bunch of losses into the Tensorboard. Logging each of them into
        its independent plot and into a joined plot
        """
        if dir is not None:
            dict = {f"{dir}/{key}": val for key, val in dict.items()}
        else:
            dict = {key: val for key, val in dict.items()}

        for key, val in dict.items():
            self.add_scalar(name=key, val=val, step=step)

        plot_name = f"{dir}/{plot_name}" if dir is not None else key
        self.add_scalars(plot_name=plot_name, val_names=dict.keys(), vals=dict.values(), step=step)
        return


def plot_sequence_comparison(
    rgbs_orig, masks_orig, flows_orig, coords_orig,
    rgbs_trans, masks_trans, flows_trans, coords_trans,
    n_rows=6, figsize=(16, 12), sequence_idx=0):
    """
    Plots a sequence comparison: Original vs Transformed frames with RGB, Flow, Mask, and RGB+BBox.
    """
    title = f'Sequence Comparison: Original (Left) vs Transformed (Right)'
    fig, axes = plt.subplots(n_rows, 8, figsize=figsize)  # 8 columns: 4 orig + 4 trans

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, y=0.98)
    
    # Add column headers
    col_headers = ['RGB (Orig)', 'Flow (Orig)', 'Mask (Orig)', 'RGB+BBox (Orig)',
                   'RGB (Trans)', 'Flow (Trans)', 'Mask (Trans)', 'RGB+BBox (Trans)']
    
    for col, header in enumerate(col_headers):
        axes[0, col].set_title(header, fontsize=10, pad=10)

    # Plot every 4th frame to show sequence progression
    for i in range(n_rows):
        frame_idx = i * 4  # Show frames 0, 4, 8, 12, 16, 20
        
        if frame_idx >= len(rgbs_orig):
            break

        # === ORIGINAL DATA (Left 4 columns) ===
        rgb_orig = rgbs_orig[frame_idx]
        flow_orig = flows_orig[frame_idx]
        mask_orig = masks_orig['masks'][frame_idx]
        bbox_orig = coords_orig['bbox'][frame_idx]

        # Original RGB
        rgb_display_orig = rgb_orig.clamp(0, 255) / 255.0
        axes[i, 0].imshow(rgb_display_orig.permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_aspect('auto')

        # Original Flow
        flow_display_orig = flow_orig.clamp(0, 255) / 255.0
        axes[i, 1].imshow(flow_display_orig.permute(1, 2, 0).cpu().numpy())
        axes[i, 1].set_aspect('auto')

        # Original Mask
        axes[i, 2].imshow(mask_orig.cpu().numpy(), cmap='gray')
        axes[i, 2].set_aspect('auto')

        # Original RGB + BBox
        rgb_np_orig = rgb_orig.clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy().copy()
        for b in bbox_orig:
            h, w = rgb_np_orig.shape[:2]
            x0, y0, x1, y1 = map(int, [max(0, min(w-1, b[0])), max(0, min(h-1, b[1])),
                                       max(0, min(w-1, b[2])), max(0, min(h-1, b[3]))])
            if x1 > x0 and y1 > y0:
                rgb_np_orig[y0:y1, x0:min(x0+2, w)] = [255, 0, 0]
                rgb_np_orig[y0:y1, max(x1-2, 0):x1] = [255, 0, 0]
                rgb_np_orig[y0:min(y0+2, h), x0:x1] = [255, 0, 0]
                rgb_np_orig[max(y1-2, 0):y1, x0:x1] = [255, 0, 0]
        axes[i, 3].imshow(rgb_np_orig)
        axes[i, 3].set_aspect('auto')

        # === TRANSFORMED DATA (Right 4 columns) ===
        rgb_trans = rgbs_trans[frame_idx]
        flow_trans = flows_trans[frame_idx]
        mask_trans = masks_trans['masks'][frame_idx]
        bbox_trans = coords_trans['bbox'][frame_idx]

        # Transformed RGB (handle normalization)
        if rgb_trans.max() <= 1.0:
            rgb_display_trans = rgb_trans.clamp(0, 1)
        else:
            rgb_display_trans = rgb_trans.clamp(0, 255) / 255.0
        axes[i, 4].imshow(rgb_display_trans.permute(1, 2, 0).cpu().numpy())
        axes[i, 4].set_aspect('auto')

        # Transformed Flow (handle normalization)
        if flow_trans.max() <= 1.0:
            flow_display_trans = flow_trans.clamp(0, 1)
        else:
            flow_display_trans = flow_trans.clamp(0, 255) / 255.0
        axes[i, 5].imshow(flow_display_trans.permute(1, 2, 0).cpu().numpy())
        axes[i, 5].set_aspect('auto')

        # Transformed Mask
        axes[i, 6].imshow(mask_trans.cpu().numpy(), cmap='gray')
        axes[i, 6].set_aspect('auto')

        # Transformed RGB + BBox
        if rgb_trans.max() <= 1.0:
            rgb_np_trans = (rgb_trans.clamp(0, 1) * 255).permute(1, 2, 0).byte().cpu().numpy().copy()
        else:
            rgb_np_trans = rgb_trans.clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy().copy()

        for b in bbox_trans:
            h, w = rgb_np_trans.shape[:2]
            x0, y0, x1, y1 = map(int, [max(0, min(w-1, b[0])), max(0, min(h-1, b[1])),
                                       max(0, min(w-1, b[2])), max(0, min(h-1, b[3]))])
            if x1 > x0 and y1 > y0:
                rgb_np_trans[y0:y1, x0:min(x0+2, w)] = [255, 0, 0]
                rgb_np_trans[y0:y1, max(x1-2, 0):x1] = [255, 0, 0]
                rgb_np_trans[y0:min(y0+2, h), x0:x1] = [255, 0, 0]
                rgb_np_trans[max(y1-2, 0):y1, x0:x1] = [255, 0, 0]
        axes[i, 7].imshow(rgb_np_trans)
        axes[i, 7].set_aspect('auto')

    plt.tight_layout()
    plt.show()
    


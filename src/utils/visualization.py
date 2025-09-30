import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

def visualize_attention(image, attention_maps, patch_size=16, img_size=64):
    """ Overlaying the attention maps on the image """
    num_layers = len(attention_maps)
    num_heads, num_tokens = attention_maps[0].shape
    patches_per_side = img_size // patch_size
    num_patches = patches_per_side * patches_per_side
    
    # first displaying raw image
    fig, ax = plt.subplots(1, num_layers + 1)
    fig.set_size_inches(30, 5)
    ax[0].imshow(image, cmap='gray')
    ax[0].axis("off")
    ax[0].set_title("Image", fontsize=20)

    # displaying attention from each layer
    image_unnorm = (image * 255).astype(np.uint8)
    H, W = image.shape[:2]
    for i in range(num_layers):
        cur_attn = attention_maps[i][:, 1:]  # current attn and removing [CLS] token

        attn = cur_attn.mean(axis=0)  # average across heads 
        attn = attn / attn.max()  # renormalization
        attn_grid = attn.reshape(patches_per_side, patches_per_side)  # mapping back to image

        # Resize to image resolution        
        attn_up = cv2.resize(attn_grid, (W, H), interpolation=cv2.INTER_CUBIC)
        # attn_up = cv2.resize(attn_grid, (W, H), interpolation=cv2.INTER_NEAREST)

        # cmap = "jet"
        cmap = "coolwarm"
        
        im = ax[i+1].imshow(image, cmap='gray')
        ax[i+1].imshow(attn_up, cmap='gray', alpha=0.01, extent=(0, W, H, 0))
        cbar = plt.colorbar(ax[i+1].imshow(attn_up, cmap=cmap, alpha=0.8, extent=(0, W, H, 0), vmin=0, vmax=1), ax=ax[i+1], fraction=0.046, pad=0.04, cmap=cmap)
        cbar.set_label('Attention Intensity', fontsize=15)
        ax[i+1].axis('off')
        ax[i+1].set_title(f"Attention Layer {i+1}/{num_layers}", fontsize=20)
        
    plt.show()

def visualize_progress(loss_iters, train_loss, val_loss, valid_acc, start=0):
    """ Visualizing loss and accuracy """
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(24,5)
    
    smooth_loss = smooth(loss_iters, 31)
    ax[0].plot(loss_iters, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_loss, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_title("Training Progress")
    
    epochs = np.arange(len(train_loss)) + 1
    ax[1].plot(epochs, train_loss, c="red", label="Train Loss", linewidth=3)
    ax[1].plot(epochs, val_loss, c="blue", label="Valid Loss", linewidth=3)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_title("Loss Curves")
    
    epochs = np.arange(len(val_loss)) + 1
    ax[2].plot(epochs, valid_acc, c="red", label="Valid accuracy", linewidth=3)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Accuracy (%)")
    ax[2].set_title(f"Valdiation Accuracy (max={round(np.max(valid_acc),2)}% @ epoch {np.argmax(valid_acc)+1})")
    
    plt.show()
    return


def visualize_masked_image(img, mask, patch_size=32):
    """
    Visualize original vs masked image.

    Args:
        img: [C, H, W] tensor (one image)
        mask: [N] tensor (0 = keep, 1 = masked)
        patch_size: size of each patch
    """
    C, H, W = img.shape
    num_patches_per_side = H // patch_size


    img = img.permute(1, 2, 0)/255.0

    # Make a copy for masked image
    masked_img = img.numpy().copy()

    # Apply mask by blacking out patches
    idx = 0
    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            if mask[idx] == 1:  # masked
                y0, y1 = i * patch_size, (i + 1) * patch_size
                x0, x1 = j * patch_size, (j + 1) * patch_size
                masked_img[y0:y1, x0:x1, :] = 0.0
            idx += 1

    # Plot side by side
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(masked_img)
    axs[1].set_title("Masked")
    axs[1].axis("off")

    plt.show()

def plot_images_(orig_imgs, recons_imgs, num_images=40):
    """
    Plots a 10x8 grid alternating between original and reconstructed images.
    Row 1: 8 originals, Row 2: 8 reconstructions, Row 3: 8 originals, etc.
    """
    # Move to CPU and clamp to [0,1]
    orig_imgs = torch.clamp(orig_imgs.detach().cpu(), 0, 1)
    recons_imgs = torch.clamp(recons_imgs.detach().cpu(), 0, 1)

    # Apply percentile normalization to both for consistent contrast
    for imgs in [orig_imgs, recons_imgs]:
        p1, p99 = torch.quantile(imgs, 0.01), torch.quantile(imgs, 0.99)
        imgs[:] = torch.clamp((imgs - p1) / (p99 - p1 + 1e-8), 0, 1)

    # We need 40 images total (5 pairs of orig/recon rows, 8 images each)
    n_rows, n_cols = 10, 8
    needed_images = 40  # 5 rows of originals + 5 rows of reconstructions
    orig_imgs = orig_imgs[:needed_images]
    recons_imgs = recons_imgs[:needed_images]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    
    for row in range(n_rows):
        for col in range(n_cols):
            if row % 2 == 0:  # Even rows: originals
                img_idx = (row // 2) * n_cols + col
                if img_idx < len(orig_imgs):
                    img = orig_imgs[img_idx]
                    img = img.permute(1, 2, 0) if img.shape[0] == 3 else img
                    axes[row, col].imshow(img.numpy())
                    axes[row, col].set_title(f"Orig {col+1}")
            else:  # Odd rows: reconstructions
                img_idx = (row // 2) * n_cols + col
                if img_idx < len(recons_imgs):
                    img = recons_imgs[img_idx]
                    img = img.permute(1, 2, 0) if img.shape[0] == 3 else img
                    axes[row, col].imshow(img.numpy())
                    axes[row, col].set_title(f"Recon {col+1}")
            
            axes[row, col].axis("off")
    
    plt.suptitle("Original V.S Reconstructed Images for Various Sequences", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_images_vs_recons(rgbs, recons, num_sequences=5, frames_per_sequence=8):
    """Plot 5 random sequences, each showing 8 non-sequential frames (every 2nd frame)"""
    batch_size = rgbs.shape[0]
    random_indices = torch.randperm(batch_size)[:num_sequences]
    
    orig_imgs_list = []
    recons_imgs_list = []
    
    for seq_idx, batch_idx in enumerate(random_indices):
        # Select non-sequential frames: 0, 2, 4, 6, 8, 10, 12, 14 (every 2nd frame)
        frame_indices = torch.arange(0, frames_per_sequence * 2, 2)  # [0, 2, 4, 6, 8, 10, 12, 14]
        
        # Make sure we don't exceed the sequence length
        max_frames = rgbs.shape[1]  # Total frames in sequence
        frame_indices = frame_indices[frame_indices < max_frames]
        
        # Select frames from this sequence
        orig_imgs = rgbs[batch_idx, frame_indices]  # [selected_frames, C, H, W]
        recons_imgs = recons[batch_idx, frame_indices]  # [selected_frames, C, H, W]
        
        orig_imgs_list.append(orig_imgs)
        recons_imgs_list.append(recons_imgs)
    
    # Concatenate all images for plotting
    all_orig_imgs = torch.cat(orig_imgs_list, dim=0)  # [total_selected_frames, C, H, W]
    all_recons_imgs = torch.cat(recons_imgs_list, dim=0)
    total_images = all_orig_imgs.shape[0]
    plot_images_(all_orig_imgs, all_recons_imgs, num_images=total_images)
    
    
def plot_predictor_images(input_images, target_images, recons):
    
    # Plot input images
    fig, axes = plt.subplots(1, input_images.shape[1], figsize=(input_images.shape[1]*3, 3))
    for i in range(input_images.shape[1]):
        ax = axes[i] if input_images.shape[1] > 1 else axes
        img = input_images[0, i].detach().cpu().permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"Input {i+1}")
        ax.axis('off')
    plt.suptitle("Input Images")
    plt.show()

    # Plot target images
    fig, axes = plt.subplots(1, target_images.shape[1], figsize=(target_images.shape[1]*3, 3))
    for i in range(target_images.shape[1]):
        ax = axes[i] if target_images.shape[1] > 1 else axes
        img = target_images[0, i].detach().cpu().permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"Target {i+1}")
        ax.axis('off')
    plt.suptitle("Target Images")
    plt.show()

    # Plot the reconstructed images from the holistic predictor
    def quantile_0_1(img):
        # img: numpy array, shape [H, W, C] or [C, H, W]
        q_min = np.quantile(img, 0.0)
        q_max = np.quantile(img, 1.0)
        if q_max > q_min:
            img = (img - q_min) / (q_max - q_min)
        else:
            img = np.zeros_like(img)
        img = np.clip(img, 0, 1)
        return img

    if recons.ndim == 5:
        fig, axes = plt.subplots(1, recons.shape[1], figsize=(recons.shape[1]*3, 3))
        for i in range(recons.shape[1]):
            ax = axes[i] if recons.shape[1] > 1 else axes
            img = recons[0, i].detach().cpu().permute(1, 2, 0).numpy()
            img = quantile_0_1(img)
            ax.imshow(img)
            ax.set_title(f"Recon {i+1}")
            ax.axis('off')
        plt.suptitle("Reconstructed Images")
        plt.show()
    else:
        print("Reconstructed output is not in image format and cannot be plotted directly.")

 
    
def plot_transform_comparison(
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
    
def plot_object_frames(rgbs, object_frames, batch_idx, time_idx):
    """
    Plots the original image and the extracted object frames for a given batch and time index.

    Args:
        rgbs: Tensor of shape [B, T, C, H, W]
        object_frames: Tensor of shape [B, T, num_objects, C, H, W]
        batch_idx: int, index of the batch to plot
        time_idx: int, index of the time step to plot
    """
    num_objects = object_frames.shape[2]

    # Plot the original image
    plt.figure(figsize=(3, 3))
    plt.imshow(rgbs[batch_idx, time_idx].permute(1, 2, 0).cpu().numpy())
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(3 * num_objects, 3))
    for obj_id in range(num_objects):
        obj_img = object_frames[batch_idx, time_idx, obj_id]  # shape: [C, H, W]
        plt.subplot(1, num_objects, obj_id + 1)
        # If the object frame is all zeros, skip displaying
        if obj_img.abs().sum() == 0:
            plt.title(f"Object {obj_id}\n(empty)")
            plt.axis("off")
            continue
        plt.imshow(obj_img.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Object {obj_id+1}")
        plt.axis("off")
    plt.show()
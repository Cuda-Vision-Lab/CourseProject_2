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
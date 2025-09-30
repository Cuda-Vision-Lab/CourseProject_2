"""
This file contains the base class for metrics and the implementation of the PSNR, SSIM, and LPIPS metrics. 
Code adapted from https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction
"""

import torch
import piqa


# try:
#     from pytorch_fvd import fvd
#     FVD_AVAILABLE = True
# except ImportError:
#     FVD_AVAILABLE = False
#     print("Warning: pytorch_fvd not available. FVD metric will not work.")


class Metric:
    """
    Base class for metrics
    """

    def __init__(self):
        """ Metric initializer """
        self.results = None
        self.reset()

    def reset(self):
        """ Reseting precomputed metric """
        raise NotImplementedError("Base class does not implement 'reset' functionality")

    def accumulate(self):
        """ """
        raise NotImplementedError("Base class does not implement 'accumulate' functionality")

    def aggregate(self):
        """ """
        raise NotImplementedError("Base class does not implement 'aggregate' functionality")

    def _shape_check(self, tensor, name="Preds"):
        """ """
        if len(tensor.shape) not in [3, 4, 5]:
            raise ValueError(f"{name} has shape {tensor.shape}, but it must have one of the folling shapes\n"
                             " - (B, F, C, H, W) for frame or heatmap prediction.\n"
                             " - (B, F, D) or (B, F, N_joints, N_coords) for pose skeleton prediction")


class PSNR(Metric):
    """ Peak Signal-to-Noise ratio computer """

    LOWER_BETTER = False

    def __init__(self):
        """ """
        super().__init__()
        self.values = []

    def reset(self):
        """ Reseting counters """
        self.values = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_psnr = piqa.psnr.psnr(preds, targets)
        cur_psnr = cur_psnr.view(B, F)
        self.values.append(cur_psnr)
        return cur_psnr.mean()

    def aggregate(self):
        """ Computing average metric, both global and framewise"""
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)
        return float(mean_values), frame_values


class SSIM(Metric):
    """ Structural Similarity computer """

    LOWER_BETTER = False

    def __init__(self, window_size=11, sigma=1.5, n_channels=3):
        """ """
        self.ssim = piqa.ssim.SSIM(
                window_size=window_size,
                sigma=sigma,
                n_channels=n_channels,
                reduction=None
            )
        super().__init__()
        self.values = []

    def reset(self):
        """ Reseting counters """
        self.values = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        if self.ssim.kernel.device != preds.device:
            self.ssim = self.ssim.to(preds.device)

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_ssim = self.ssim(preds, targets)
        cur_ssim = cur_ssim.view(B, F)
        self.values.append(cur_ssim)
        return cur_ssim.mean()

    def aggregate(self):
        """ Computing average metric, both global and framewise"""
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)
        return float(mean_values), frame_values


class LPIPS(Metric):
    """ Learned Perceptual Image Patch Similarity computers """

    LOWER_BETTER = True

    def __init__(self, network="alex", pretrained=True, reduction=None):
        """ """
        # piqa.lpips.LPIPS doesn't support 'pretrained' parameter in current version
        self.lpips = piqa.lpips.LPIPS(
                network=network,
                reduction=reduction
            )
        super().__init__()
        self.values = []

    def reset(self):
        """ Reseting counters """
        self.values = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        if not hasattr(self.lpips, "device"):
            self.lpips = self.lpips.to(preds.device)
            self.lpips.device = preds.device

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_lpips = self.lpips(preds, targets)
        cur_lpips = cur_lpips.view(B, F)
        self.values.append(cur_lpips)
        return cur_lpips.mean()

    def aggregate(self):
        """ Computing average metric, both global and framewise"""
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)
        return float(mean_values), frame_values


class FVD(Metric):
    """
    Fréchet Video Distance (FVD) computer using pytorch-fvd package.
    """

    LOWER_BETTER = True

    def __init__(self):
        if not FVD_AVAILABLE:
            raise ImportError("pytorch-fvd is required for FVD metric. Install with `pip install pytorch-fvd`.")
        super().__init__()
        self.values = []

    def reset(self):
        """ Resetting counters """
        self.values = []

    def accumulate(self, preds, targets):
        """
        Computes the FVD for a batch of generated and real videos.
        Args:
        -----
        preds: torch.Tensor, shape (B, F, C, H, W), values in [0, 1]
        targets: torch.Tensor, shape (B, F, C, H, W), values in [0, 1]
        """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        # pytorch-fvd expects (B, T, C, H, W) with values in [0, 255] (uint8) or [0, 1] (float32)
        # We'll assume input is [0, 1] float32.

        # Move to same device
        device = preds.device
        _preds = preds.to(device)
        _targets = targets.to(device)

        # FVD computes distance between two sets of videos
        if not FVD_AVAILABLE:
            raise ImportError("pytorch-fvd is required for FVD metric. Install with `pip install pytorch-fvd`.")
        score = fvd(_preds, _targets)
        self.values.append(score.detach())
        return score.item()

    def aggregate(self):
        """Computing average FVD over all accumulated batches"""
        all_values = torch.stack(self.values, dim=0)  # shape: (num_batches,)
        mean_value = all_values.mean()
        return float(mean_value), all_values  # no framewise for FVD

    # Add to _get_metric
    def _get_metric(self, metric):
        """ """
        if metric == "psnr":
            metric_computer = PSNR()
        elif metric == "ssim":
            metric_computer = SSIM()
        elif metric == "lpips":
            metric_computer = LPIPS()
        elif metric == "fvd":
            if not FVD_AVAILABLE:
                raise ImportError("pytorch-fvd is required for FVD metric. Install with `pip install pytorch-fvd`.")
            metric_computer = FVD()
        else:
            raise NotImplementedError(f"Unknown metric {metric}...")
        return metric_computer
    
    
    
def evaluate_metrics(recons, rgbs):

# Applying percentile normalization to rgbs and recons for consistent contrast before metric evaluation

    def percentile_normalize(imgs):
        # imgs: torch.Tensor, shape [..., C, H, W]
        p1 = torch.quantile(imgs, 0.01)
        p99 = torch.quantile(imgs, 0.99)
        imgs = torch.clamp((imgs - p1) / (p99 - p1 + 1e-8), 0, 1)
        return imgs

    rgbs = percentile_normalize(rgbs)
    recons = percentile_normalize(recons)

    psnr_metric = PSNR()
    ssim_metric = SSIM(n_channels=3)  
    lpips_metric = LPIPS(network="alex")  # network can be "alex", "vgg", etc.

    psnr_metric.reset()
    ssim_metric.reset()
    lpips_metric.reset()

    # preds and targets: shape [B, T, C, H, W]
    psnr_metric.accumulate(recons, rgbs)
    ssim_metric.accumulate(recons, rgbs)
    lpips_metric.accumulate(recons, rgbs)

    psnr_mean, psnr_framewise = psnr_metric.aggregate()
    ssim_mean, ssim_framewise = ssim_metric.aggregate()
    lpips_mean, lpips_framewise = lpips_metric.aggregate()

    print("PSNR Mean:", psnr_mean)
    print("PSNR Framewise:", psnr_framewise)
    print("SSIM Mean:", ssim_mean)
    print("SSIM Framewise:", ssim_framewise)
    print("LPIPS Mean:", lpips_mean)
    print("LPIPS Framewise:", lpips_framewise)

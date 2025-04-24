"""
this file is modified from pytorch-ssim
https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(
        img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size //
                       2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.shape[-3]
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

'''
def gau_loss(image, gt_image, loss_lambda=0.2, chroma_weight=0.5):
    """
    image:      (3, H, W) predicted image in normalized Lab
    gt_image:   (3, H, W) ground truth image in normalized Lab
    loss_lambda: weight for SSIM loss on L channel
    chroma_weight: weight for chroma (a/b) loss
    """
    delta_e_loss = torch.norm(image - gt_image, dim=0).mean()
    #delta_e_squared = ((image - gt_image)**2).sum(dim=0).mean()
    # --- L1 loss on full Lab ---
    loss_l1 = torch.abs(image - gt_image).mean()
    
    # --- Chroma (a and b channels) L1 loss ---
    #chroma_loss = torch.abs(image[1:] - gt_image[1:]).mean()
    
    image_L = image[:, 0:1]  # L channel only
    gt_L = gt_image[:, 0:1]
    ssim_L = 1.0 - ssim(image_L, gt_L)
    
    # --- Final loss ---
    #total_loss =  loss_l1 #+ chroma_weight * chroma_loss
    #return 0.8 * delta_e_loss + 0.1 * loss_l1 + 0.1 * ssim_L    
    return 0.7 * delta_e_loss + 0.1 * loss_l1 + 0.1 * ssim_L
'''
    
    
# Original approach
def gau_loss(image, gt_image, loss_lambda=0.2):
    loss_l1 = torch.abs((image - gt_image)).mean()
    loss_ssim = 1.0 - ssim(image, gt_image)
    return (1.0 - loss_lambda) * loss_l1 + loss_lambda * loss_ssim


'''
def gau_loss(image, gt_image, loss_lambda=0.2):
    # image: (3, H, W), normalized Lab
    loss_l1 = torch.abs(image - gt_image).mean()

    # SSIM only on L channel
    image_L = image[0:1, :, :]      # (1, H, W)
    gt_L = gt_image[0:1, :, :]      # (1, H, W)
    
    loss_ssim = 1.0 - ssim(image_L, gt_L)
    
    return (1.0 - loss_lambda) * loss_l1 + loss_lambda * loss_ssim
'''

### customized loss
def sparse_weight_loss(params, eps=1e-8):
    """
    Encourage sparsity (non-uniformity) across rows of W (shape Nx6).
    Assumes each row is positive and sums to 1.
    """
    w = params["weights"]
    w = F.softmax(w, dim=1)
    entropy = -torch.sum(w * torch.log(w + eps), dim=1)  # shape: (N,)
    return torch.mean(entropy)  # minimize this to encourage sparsity

def simple_gamut_loss(sh, L=3):
    # Compute L1 norm per row
    rowsum = sh.sum(dim=1, keepdim=True)  # shape: (6, (L+1)^2)
    # Multiply with sigmoid bound
    one_bound = 0.25 * rowsum   # 1-bound
    # Penalize over gamut bounds
    violation_one = torch.relu(one_bound-1)
    violation_pos = torch.relu(-sh)
    return 0.5*torch.mean(violation_one) + 0.5*torch.mean(violation_pos)

def _hard_sh_gamut_channel_loss(sh, L):
    """
    Computes hard gamut constraint for a single color channel's SH coefficients.

    Args:
        sh (Tensor): shape (N, (L+1)^2)
        L (int): SH order
        eps (float): Stability term

    Returns:
        Tensor: scalar loss
    """
    f00 = sh[:, 0]  # Zeroth-order term
    sh_rest = sh[:, 1:]  # Higher-order terms
    
    norm_sq = torch.sum(sh_rest**2, dim=1)
    
    # Bound 1: Ensure function ≤ 1
    C = 1
    bound1 = C*(4 * torch.pi - f00**2 * (L + 1)**2) / (L + 1)**2
    # Bound 2: Ensure function ≥ 0
    bound2 = C*f00**2 / ((L + 1)**2 - 1)
    
    bound = torch.min(bound1, bound2)
    violation = torch.relu(norm_sq - bound)
    return torch.mean(violation)

def _hard_sh_gamut_channel_loss_w_measurement(sh, L, eps=1e-6):
    """
    Computes hard gamut constraint for a single color channel's SH coefficients.

    Args:
        sh (Tensor): shape (N, (L+1)^2)
        L (int): SH order
        eps (float): Stability term

    Returns:
        Tensor: scalar loss
    """
    f00 = sh[:, 0]  # Zeroth-order term
    sh_rest = sh[:, 1:]  # Higher-order SH terms (exclude f00)
    norm_sq = torch.sum(sh_rest**2, dim=1)
    
    # Spread factor s = ||f||_2 / ||f||_inf
    norm2_all = torch.norm(sh, dim=1)
    norm_inf_all = torch.amax(torch.abs(sh), dim=1) + eps
    norm2_higher = torch.norm(sh_rest, dim=1)
    norm_inf_higher = torch.amax(torch.abs(sh_rest), dim=1) + eps
    s_all = norm2_all / norm2_all
    s_higher = norm2_higher / norm2_higher

    # 1-bound using s: ||f||_2^2 <= (4π / (L+1)^2) * s^2
    bound1 = (4 * torch.pi / (L + 1)**2) * s_all**2 - f00**2
    # 0-bound using s: ||f (l>=1)||_2^2 <= f00^2 / ((L+1)^2-1) * s(l>=1)^2
    bound0 = f00**2 / ((L + 1)**2 - 1) * s_higher**2
    
    bound = torch.min(bound1, bound0)
    violation = torch.relu(norm_sq - bound)
    return torch.mean(violation)

def _soft_sh_gamut_channel_loss(sh, alpha=0.01, beta=0.01, eps=0.01):
    # Now: alpha_sq=0.01, beta_sq=0.01, eps=0.1 works
    """
    Computes soft gamut constraint for a single color channel's SH coefficients.

    Args:
        sh (Tensor): shape (N, (L+1)^2)
        alpha_sq (float): control how much percentage function outputs are bounded below zero 
        beta_sq (float): control how much percentage function outputs are bounded above one

    Returns:
        Tensor: scalar loss
    """
    FOUR_PI = 4 * torch.tensor(torch.pi)
    
    f00 = sh[:, 0]  # Zeroth-order term
    sh_rest = sh[:, 1:]  # Higher-order terms
    
    norm_sq = torch.sum(sh_rest**2, dim=1)
    
    # Bound 1: Ensure function ≥ 0 
    bound1 = FOUR_PI * (eps**2) / alpha - f00**2
    # Bound 2: Ensure function ≤ 1
    bound2 = FOUR_PI * ( (eps**2)/beta + 2*f00/torch.sqrt(FOUR_PI) - 1  ) - f00**2 
    
    bound = torch.min(bound1, bound2)
    violation = torch.relu(norm_sq - bound)
    return torch.mean(violation)

def sh_gamut_constraint_loss(params, L=3, eps=1e-8):
    """
    Applies the gamut constraint loss to RGB SH coefficients separately.
    
    Args:
        sh_coeffs (Tensor): shape (N, 3*(L+1)^2), SH coeffs for RGB
        L (int): SH order (default 3 → 16 coeffs per channel)
        lambda_reg (float): Regularization strength
        eps (float): Small value for stability
        
    Returns:
        Tensor: scalar loss
    """
    # reconstruct palette SH
    sh_coeffs = torch.cat([params['low_palette_shs'], params['high_palette_shs']], dim=1)  # shape: (palette_size, 48)
    
    ps, D = sh_coeffs.shape
    assert D == 3 * (L + 1)**2, "Expected 3 color channels with (L+1)^2 SH coeffs each"
    
    # Reshape to (palette_size, 3, 16): color_channel x SH basis
    sh_reshaped = sh_coeffs.view(ps, (L + 1)**2, 3).transpose(1, 2)

    # Split R, G, B
    sh_r = sh_reshaped[:, 0, :]  # shape (palette_size, 16)
    sh_g = sh_reshaped[:, 1, :]
    sh_b = sh_reshaped[:, 2, :]

    loss_r = simple_gamut_loss(sh_r, L)
    loss_g = simple_gamut_loss(sh_g, L)
    loss_b = simple_gamut_loss(sh_b, L)

    return (loss_r + loss_g + loss_b) / 3


if __name__ == "__main__":
    height, width = 100, 100
    image = torch.zeros([3, height, width], dtype=torch.float32).to('cuda')
    image_gt = torch.zeros([3, height, width], dtype=torch.float32).to('cuda')
    loss = gau_loss(image, image_gt)
    print(loss)
    
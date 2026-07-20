# niqe 实现代码（来自 scikit-image==0.19.3）

import numpy as np
import scipy
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.linalg import sqrtm
from skimage import color
from skimage.util import view_as_windows

def _as_row_vector(X):
    return X.reshape((1, X.size))

def _compute_mscn_transform(image, kernel_size=7, sigma=7/6):
    C = 1/255
    mu = gaussian_filter(image, sigma=sigma)
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(gaussian_filter(image * image, sigma=sigma) - mu_sq))
    mscn = (image - mu) / (sigma + C)
    return mscn

def _extract_subband_features(mscn_coefficients):
    shifted_left = np.roll(mscn_coefficients, 1, axis=1)
    shifted_up = np.roll(mscn_coefficients, 1, axis=0)
    shifted_up_left = np.roll(shifted_up, 1, axis=1)
    shifted_up_right = np.roll(np.roll(mscn_coefficients, -1, axis=1), 1, axis=0)

    H, W = mscn_coefficients.shape
    features = np.stack([
        mscn_coefficients[1:H-1, 1:W-1],
        shifted_left[1:H-1, 1:W-1],
        shifted_up[1:H-1, 1:W-1],
        shifted_up_left[1:H-1, 1:W-1],
        shifted_up_right[1:H-1, 1:W-1],
    ], axis=-1)

    return features

def _estimate_aggd_param(vec):
    gam = np.arange(0.2, 10, 0.001)
    r_gam = (scipy.special.gamma(2/gam) ** 2) / (scipy.special.gamma(1/gam) * scipy.special.gamma(3/gam))
    left_std = np.sqrt(np.mean(vec[vec < 0] ** 2))
    right_std = np.sqrt(np.mean(vec[vec > 0] ** 2))
    gamma_hat = left_std / right_std
    r_hat = (np.mean(np.abs(vec)))**2 / np.mean(vec**2)
    rhat_norm = r_hat * (gamma_hat**3 + 1) * (gamma_hat + 1) / ((gamma_hat**2 + 1)**2)
    idx = (np.abs(r_gam - rhat_norm)).argmin()
    return gam[idx], left_std, right_std

def _compute_feature_vector(image):
    image = image.astype(np.float32)
    if image.ndim == 3:
        image = color.rgb2gray(image)

    mscn = _compute_mscn_transform(image)
    features = _extract_subband_features(mscn)
    H, W, _ = features.shape
    features = features.reshape((H * W, 5))

    feat_vectors = []
    for i in range(5):
        alpha, left_std, right_std = _estimate_aggd_param(features[:, i])
        feat_vectors.extend([alpha, (left_std + right_std) / 2])

    return np.array(feat_vectors)

def niqe(image, pop_mu, pop_cov, patch_size=96):
    if image.ndim == 3:
        image = color.rgb2gray(image)
    image = image.astype(np.float32)

    h, w = image.shape
    if h < patch_size or w < patch_size:
        raise ValueError('Image too small for patch size')

    patches = view_as_windows(image, (patch_size, patch_size), step=patch_size // 2)
    n_h, n_w, _, _ = patches.shape
    feats = []
    for i in range(n_h):
        for j in range(n_w):
            patch = patches[i, j]
            feats.append(_compute_feature_vector(patch))
    feats = np.array(feats)

    mu = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False)
    inv_cov = np.linalg.pinv((cov + pop_cov) / 2)
    diff = mu - pop_mu
    niqe_score = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
    return float(niqe_score)


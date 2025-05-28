import torch
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np
from math import exp
import faiss
from skimage.metrics import structural_similarity as ssim

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def apply_mask(image, mask):
    masked = image.copy()
    masked[mask == 1] = 0  # 将掩膜区域置为黑色
    return masked

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    #return ssim_map

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim1(img1, img2, window_size=13, size_average=True):
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    img1 = torch.from_numpy(img1).permute(0, 3, 1, 2).float()
    img2 = torch.from_numpy(img2).permute(0, 3, 1, 2).float()

    _, channel, _, _ = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def Edge_select_faiss(image, mask, boundary_range=10):
    
    # 1. 检测 mask 的边缘
    mask_edges = cv2.Canny(mask, 100, 200)  # mask 的边缘
    mask_edge_coords = np.argwhere(mask_edges > 0)  # 提取边缘像素坐标

    # 2. 检测生成图片的边缘
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)  # 平滑处理，减少噪声
    image_edges = cv2.Canny(blurred_image, 50, 200)  # 生成图片的边缘
    image_edge_coords = np.argwhere(image_edges > 0)  # 提取边缘像素坐标

    if len(image_edge_coords) == 0 or len(mask_edge_coords) == 0:
        return 0  # 如果没有生成边缘或 mask 边缘，匹配分数为 0

    # 3. 使用 faiss 加速最近邻匹配
    image_edge_coords_float = np.ascontiguousarray(image_edge_coords.astype(np.float32))
    mask_edge_coords_float = np.ascontiguousarray(mask_edge_coords.astype(np.float32))

    # 构建 faiss 索引
    index = faiss.IndexFlatL2(image_edge_coords_float.shape[1])  # 欧几里得距离索引
    index.add(image_edge_coords_float)  # 加入生成图片边缘点

    # 查询最近邻
    distances, indices = index.search(mask_edge_coords_float, 1)  # 每个 mask 点查询 1 个最近邻

    # 4. 计算权重
    exact_matches = distances.flatten() == 0  # 精确匹配（距离为 0）
    weights = np.sum(exact_matches)  # 精确匹配的权重

    # remaining_mask_coords = mask_edge_coords_float[~exact_matches]
    remaining_distances = distances.flatten()[~exact_matches]

    valid_matches = remaining_distances <= boundary_range  # 距离在范围内的匹配
    weights += np.sum(np.maximum(0, 1 - remaining_distances[valid_matches] / boundary_range))  # 累加权重

    # 8. 归一化匹配分数
    total_mask_edges = len(mask_edge_coords)  # mask 的边缘像素总数
    normalized_score = weights / total_mask_edges if total_mask_edges > 0 else 0

    return normalized_score

def ssim_color(target, generated):
    # 确保图像是 3 通道
    if len(target.shape) != 3 or len(generated.shape) != 3:
        raise ValueError("图像必须是 3 通道")

    # 对每个通道计算 SSIM
    ssim_scores = []
    for channel in range(3):  # 遍历 R, G, B 通道
        target_channel = target[:, :, channel]
        generated_channel = generated[:, :, channel]
        score, _ = ssim(target_channel, generated_channel, full=True)
        ssim_scores.append(score)

    # 返回平均值
    return np.mean(ssim_scores)

def histogram_similarity(target, generated):
    # 计算直方图
    hist_target = cv2.calcHist([target], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_gen = cv2.calcHist([generated], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # 归一化直方图
    cv2.normalize(hist_target, hist_target, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_gen, hist_gen, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # 计算巴氏距离（值越小越相似）
    hist_score = cv2.compareHist(hist_target, hist_gen, cv2.HISTCMP_BHATTACHARYYA)
    
    # 转换为相似性得分（1 - 距离）
    return 1 - hist_score
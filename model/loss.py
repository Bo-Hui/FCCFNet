import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 1

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)

        return loss


def criterion(inputs, target):
    if isinstance(inputs, list):
        losses = [F.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]
        total_loss = sum(losses)
    else:
        total_loss = F.binary_cross_entropy_with_logits(inputs, target)

    return total_loss

def edgeLoss(inputs,target):
    target = sobel_edges_from_labels(target)
    if isinstance(inputs, list):
        losses = [F.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]
        total_loss = sum(losses)
    else:
        total_loss = F.binary_cross_entropy_with_logits(inputs, target)

    return total_loss

def sobel_edges_from_labels(labels: torch.Tensor, thresh: float = 0.2) -> torch.Tensor:
    """
    labels: [N, 1, H, W]，返回 [N, 1, H, W] 的边缘图（0/1浮点）
    thresh: 相对阈值（相对于梯度最大值）
    """
    assert labels.dim() == 4 and labels.size(1) == 1, "labels 需为 [N,1,H,W]"
    x = labels.float()
    if x.max() > 1.0:
        x = x / 255.0

    # Sobel 卷积核
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]], device=x.device).view(1, 1, 3, 3)

    gx = torch.nn.functional.conv2d(x, kx, padding=1)
    gy = torch.nn.functional.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy)  # 梯度幅值

    # 相对阈值二值化
    maxv = mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    edge = (mag >= (maxv * thresh)).float()
    return edge
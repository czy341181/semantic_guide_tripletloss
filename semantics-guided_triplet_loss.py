import torch
import torch.nn.functional as F
import torch.nn as nn


def compute_affinity(feature, kernel_size):
    pad = kernel_size // 2
    feature = F.normalize(feature, dim=1) #[2, 64, 24, 80]
    unfolded = F.pad(feature, [pad] * 4).unfold(2, kernel_size, 1).unfold(3, kernel_size, 1) #[2, 64, 24, 80]-->[2, 64, 28, 84]-->[2, 64, 24, 84, 5]-->[2, 64, 20, 84, 5, 5]
    feature = feature.unsqueeze(-1).unsqueeze(-1)
    similarity = (feature * unfolded).sum(dim=1, keepdim=True) #[2, 64, 24, 80, 1, 1]*[2, 64, 24, 80, 5, 5]-->[2, 1, 24, 80, 5, 5]

    eps = torch.zeros(similarity.shape) + 1e-9
    affinity = torch.max(eps, 2 - 2 * similarity).sqrt()
    return affinity



if __name__ == '__main__':

    # input = torch.randint(1, 8, (2, 1, 400, 1200))
    # input = torch.nn.functional.one_hot(input, num_classes=10)
    # input = input.permute(0, 4, 2, 3, 1)
    # seg_target = input.squeeze(-1).float()
    seg_target = torch.randint(1, 8, (2, 1, 192, 640)).float()
    _, _, h, w = seg_target.shape
    total_loss = 0

    sgt_layers = [3, 2, 1]
    sgt_kernel_size = [5, 5, 5]

    height = 192
    width = 640
    margin = 0.3

    feature_list = [torch.rand((2, 64, 24, 80)),torch.rand((2, 64, 48, 160)),torch.rand((2, 64, 96, 320))]

    for feature, s, kernel_size in zip(feature_list, sgt_layers, sgt_kernel_size):
        pad = kernel_size // 2
        h = height // 2 ** s
        w = width // 2 ** s
        seg = F.interpolate(seg_target, size=(h, w), mode='nearest') #[2, 1, 24, 80]
        print("seg:",seg.shape)
        center = seg
        padded = F.pad(center, [pad] * 4, value=-1) #[2, 1, 28, 84]
        aggregated_label = torch.zeros(*(center.shape + (kernel_size, kernel_size))) #[2, 1, 24, 80, 5, 5]

        for i in range(kernel_size):
            for j in range(kernel_size):
                shifted = padded[:, :, 0 + i: h + i, 0 + j:w + j] #[2, 1, 24, 80]
                label = center == shifted #[2, 1, 24, 80]

                aggregated_label[:, :, :, :, i, j] = label
        aggregated_label = aggregated_label.float()
        pos_idx = (aggregated_label == 1).float()
        neg_idx = (aggregated_label == 0).float()

        pos_idx_num = pos_idx.sum(dim=-1).sum(dim=-1) #[2, 1, 24, 80]
        neg_idx_num = neg_idx.sum(dim=-1).sum(dim=-1) #[2, 1, 24, 80]

        boundary_region = (pos_idx_num >= kernel_size - 1) & (
                neg_idx_num >= kernel_size - 1) #[2, 1, 24, 80]

        non_boundary_region = (pos_idx_num != 0) & (neg_idx_num == 0) #[2, 1, 24, 80]

        # if s == min(sgt_layers):
        #     outputs[('boundary', s)] = boundary_region.data
        #     outputs[('non_boundary', s)] = non_boundary_region.data
        affinity = compute_affinity(feature, kernel_size=kernel_size) #[2, 1, 24, 80, 5, 5]

        pos_dist = (pos_idx * affinity).sum(dim=-1).sum(dim=-1)[boundary_region] / \
                   pos_idx.sum(dim=-1).sum(dim=-1)[
                       boundary_region]   ##[2, 1, 24, 80, 5, 5]*#[2, 1, 24, 80, 5, 5] -->#[2, 1, 24, 80]-->只取boundary索引

        neg_dist = (neg_idx * affinity).sum(dim=-1).sum(dim=-1)[boundary_region] / \
                   neg_idx.sum(dim=-1).sum(dim=-1)[
                       boundary_region]

        zeros = torch.zeros(pos_dist.shape)
        loss = torch.max(zeros, pos_dist - neg_dist + margin)  #只对boundary计算triplet loss

        total_loss += loss.mean() / (2 ** s)
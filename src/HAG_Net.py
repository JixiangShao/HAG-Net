from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from torch_geometric.nn import GENConv, LayerNorm, Linear
from module.HLAEM import HLAEM
from module.AttentionGate import SimpleEnhancedAttentionGate

class DoubleConv(nn.Module):
    """双卷积块（避免过多计算）"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.double_conv(x)
        return x + residual


class Down(nn.Sequential):
    """下采样模块"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, up_in_channels, skip_in_channels, out_channels, bilinear=True, use_graph=False, use_hlaem=True):
        super(Up, self).__init__()
        self.bilinear = bilinear
        self.use_graph = use_graph
        self.use_hlaem = use_hlaem

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            decoder_channels = up_in_channels

        else:
            self.up = nn.ConvTranspose2d(up_in_channels, up_in_channels // 2, kernel_size=2, stride=2)
            decoder_channels = up_in_channels // 2

            # 初始化注意力门控（使用正确的参数名和通道数）
        self.att_gate = SimpleEnhancedAttentionGate(
            encoder_channels=skip_in_channels,  # 编码器跳跃特征通道
            decoder_channels=decoder_channels  # 解码器上采样后特征通道
        )
        # 卷积融合层（输入通道 = 编码器特征通道 + 解码器特征通道）
        self.conv = DoubleConv(skip_in_channels + decoder_channels, out_channels)


        # 仅在关键解码阶段使用HLAEM
        self.hlaem_fusion = HLAEM(dim=out_channels) if use_hlaem else nn.Identity()

        # 增强的空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )

        if use_graph:
            self.graph_att = nn.Sequential(
                nn.Conv2d(64, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid()
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, graph_feat=None) -> torch.Tensor:
        x1 = self.up(x1)
        # 尺寸对齐
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        # 注意力门控融合（替代原有的直接拼接）
        x = self.att_gate(encoder_feat=x2, decoder_feat=x1)  # x2是编码器特征，x1是解码器特征

        x = self.conv(x)
        x = self.hlaem_fusion(x)  # 关键位置HLAEM融合

        # 应用GNN注意力（若启用）
        if self.use_graph and graph_feat is not None:
            graph_att = self.graph_att(F.interpolate(graph_feat, size=x.shape[2:], mode="bilinear", align_corners=True))
            x = x * graph_att

        # 空间注意力
        spatial_att = self.spatial_att(x)
        x = x * spatial_att + x  # 残差连接
        return x


class OutConv(nn.Sequential):
    """输出层"""

    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )


class VesselGraphBranch(nn.Module):
    """血管图神经网络分支"""

    def __init__(self, in_channels=320, hidden_channels=64, edge_dim=128):
        super().__init__()
        self.convs = nn.ModuleList([
            GENConv(in_channels, hidden_channels, edge_dim=edge_dim),
            GENConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        ])
        self.norms = nn.ModuleList([LayerNorm(hidden_channels) for _ in range(2)])
        self.node_proj = Linear(hidden_channels, 64)

    def forward(self, x, edge_index, edge_attr):
        # 计算节点特征相似度作为动态边权重
        src, dst = edge_index
        node_sim = F.cosine_similarity(x[src], x[dst], dim=1).unsqueeze(1)
        edge_attr = edge_attr * node_sim

        # 第一层图卷积
        x = self.convs[0](x, edge_index, edge_attr)
        x = self.norms[0](x)
        x = F.relu(x)

        # 第二层图卷积（残差连接）
        residual = x
        x = self.convs[1](x, edge_index, edge_attr)
        x = self.norms[1](x)
        x = F.relu(x + residual)

        return self.node_proj(x)


class HAG_Net(nn.Module):
    """融合GNN的UNet_HLAEM网络（优化HLAEM位置）"""

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 32,
                 use_graph: bool = True,
                 k_neighbors: int = 6):
        super(HAG_Net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.use_graph = use_graph
        self.k_neighbors = k_neighbors

        # 编码器通道数
        self.c1 = base_c
        self.c2 = base_c * 2
        self.c3 = base_c * 4
        self.c4 = base_c * 8

        # 输入预处理
        self.input_preprocess = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1)
        )

        # 编码器
        self.in_conv = DoubleConv(in_channels, self.c1)
        self.down1 = Down(self.c1, self.c2)
        self.down2 = Down(self.c2, self.c3)
        self.down3 = Down(self.c3, self.c4)

        # 在编码器与瓶颈过渡处添加HLAEM（关键位置1）
        self.hlaem_bottleneck = HLAEM(dim=self.c4)

        # 解码器（仅在关键阶段使用HLAEM）
        # 高层解码阶段使用HLAEM（关键位置2）
        self.up1 = Up(up_in_channels=self.c4, skip_in_channels=self.c3, out_channels=self.c3,
                      bilinear=bilinear, use_hlaem=True)
        # 中层解码阶段不使用HLAEM
        self.up2 = Up(up_in_channels=self.c3, skip_in_channels=self.c2, out_channels=self.c2,
                      bilinear=bilinear, use_hlaem=False)
        # 最后解码阶段使用HLAEM（关键位置3）
        self.up3 = Up(up_in_channels=self.c2, skip_in_channels=self.c1, out_channels=self.c1,
                      bilinear=bilinear, use_graph=use_graph, use_hlaem=True)

        # 跳跃连接融合节点添加HLAEM（关键位置4）
        self.hlaem_skip3 = HLAEM(dim=self.c3)
        self.hlaem_skip1 = HLAEM(dim=self.c1)

        # 输出层
        self.out_conv = OutConv(self.c1, num_classes)

        # GNN分支
        if self.use_graph:
            self.graph_branch = VesselGraphBranch(
                in_channels=self.c3 + 64,  # c3特征 + 位置编码
                hidden_channels=64,
                edge_dim=self.c3
            )
            self.gaussian_kernel = self._create_gaussian_kernel(kernel_size=3, sigma=1.0)

    def _create_gaussian_kernel(self, kernel_size=3, sigma=1.0):
        kernel_1d = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        kernel_1d = torch.exp(-0.5 * (kernel_1d ** 2) / (sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d.view(1, 1, kernel_size, kernel_size)

    def extract_vessel_centers(self, enc_feat):
        batch_size = enc_feat.shape[0]
        all_centers = []

        for b in range(batch_size):
            # 计算特征强度
            feat_norm = torch.norm(enc_feat[b], dim=0).cpu().detach().numpy()
            feat_norm = (feat_norm - feat_norm.min()) / (feat_norm.max() - feat_norm.min() + 1e-8)

            threshold = threshold_otsu(feat_norm) if feat_norm.max() > 0 else 0.3
            threshold *= 0.8
            vessel_mask = (feat_norm > threshold).astype(np.uint8)
            vessel_skeleton = skeletonize(vessel_mask)
            # 过滤过小区域
            if vessel_mask.sum() < 5:
                all_centers.append(torch.empty(0, 2, device=enc_feat.device))
                continue

            # 提取血管坐标并降采样
            vessel_coords = torch.nonzero(torch.from_numpy(vessel_skeleton).to(enc_feat.device), as_tuple=False)
            step = max(1, vessel_coords.shape[0] // 200)
            vessel_coords = vessel_coords[::step]

            all_centers.append(vessel_coords)

        return all_centers

    def build_knn_graph(self, centers,features= None):
        if centers.shape[0] < 2:
            return torch.empty(2, 0, dtype=torch.long, device=centers.device)

        centers_np = centers.cpu().numpy()
        # 动态调整邻居数量
        k_neighbors = min(self.k_neighbors + 2, centers_np.shape[0] - 1)  # 增加邻居数

        # 使用空间距离构建图
        nbrs = NearestNeighbors(n_neighbors=k_neighbors)
        nbrs.fit(centers_np)
        distances, indices = nbrs.kneighbors(centers_np)

        # 构建无向边
        edges = []
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                if i != indices[i, j]:
                    edges.append([i, indices[i, j]])
                    edges.append([indices[i, j], i])

        return torch.tensor(edges, dtype=torch.long, device=centers.device).t().contiguous() if edges else torch.empty(
            2, 0, device=centers.device)

    def get_sinusoidal_encoding(self, coords, num_freqs=16):
        device = coords.device
        freqs = torch.arange(num_freqs, dtype=torch.float32, device=device)
        freqs = 1.0 / (10000 ** (freqs / num_freqs))
        pos_y = coords[:, 0].unsqueeze(1) * freqs
        pos_x = coords[:, 1].unsqueeze(1) * freqs
        return torch.cat([pos_y.sin(), pos_y.cos(), pos_x.sin(), pos_x.cos()], dim=1)

    def scatter_graph_feats(self, feat_shape, node_feats, node_coords, batch_idx):
        b, c, h, w = feat_shape
        device = node_feats.device
        feat_map = torch.zeros((b, c, h, w), device=device)

        if node_feats.shape[0] == 0:
            return feat_map

        # 坐标裁剪避免越界
        y = node_coords[:, 0].long().clamp(0, h - 1)
        x = node_coords[:, 1].long().clamp(0, w - 1)
        batch_idx = batch_idx.clamp(0, b - 1)

        # 散射节点特征
        feat_map[batch_idx, :, y, x] = node_feats

        # 高斯模糊平滑
        kernel = self.gaussian_kernel.repeat(c, 1, 1, 1).to(device)
        feat_map = F.conv2d(
            feat_map,
            weight=kernel,
            stride=1,
            padding=1,
            groups=c
        )

        return feat_map

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.input_preprocess(x)

        # 编码器特征提取
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # 瓶颈处HLAEM增强
        x4 = self.hlaem_bottleneck(x4)

        # 跳跃连接处HLAEM增强
        x3 = self.hlaem_skip3(x3)
        x1 = self.hlaem_skip1(x1)

        # GNN特征处理
        graph_feat_map = None
        if self.use_graph:
            batch_centers = self.extract_vessel_centers(x3)
            batch_size = x.shape[0]
            h3, w3 = x3.shape[2], x3.shape[3]
            graph_feat_map = torch.zeros((batch_size, 64, h3, w3), device=x.device)

            for b in range(batch_size):
                centers = batch_centers[b]
                if centers.shape[0] < 2:
                    continue

                # 节点特征：编码器特征 + 位置编码
                norm_coords = (centers / torch.tensor([h3 - 1, w3 - 1], device=x.device)) * 2 - 1
                norm_coords = norm_coords.unsqueeze(0).unsqueeze(0)
                enc_feats = F.grid_sample(x3[b:b + 1], norm_coords, mode="bilinear", align_corners=False)
                enc_feats = enc_feats.squeeze(0).squeeze(1).permute(1, 0)
                pos_enc = self.get_sinusoidal_encoding(centers / torch.tensor([h3, w3], device=x.device))
                node_features = torch.cat([enc_feats, pos_enc], dim=1)

                # 构建图和边特征
                edge_index = self.build_knn_graph(centers)
                if edge_index.numel() == 0:
                    continue

                # 简化的边特征提取：只使用边的中点
                src_nodes = centers[edge_index[0]]
                dst_nodes = centers[edge_index[1]]
                mid_points = (src_nodes + dst_nodes) / 2

                # 提取边特征
                norm_mid_points = (mid_points / torch.tensor([h3 - 1, w3 - 1], device=x.device)) * 2 - 1
                norm_mid_points = norm_mid_points.unsqueeze(0).unsqueeze(0)
                edge_attr = F.grid_sample(x3[b:b + 1], norm_mid_points, mode="bilinear", align_corners=False)
                edge_attr = edge_attr.squeeze(0).squeeze(1).permute(1, 0)

                # GNN推理
                node_feats = self.graph_branch(node_features, edge_index, edge_attr)

                # 散射到特征图
                batch_idx_tensor = torch.full((node_feats.shape[0],), fill_value=b, device=x.device, dtype=torch.long)
                batch_graph_feat = self.scatter_graph_feats(
                    feat_shape=(1, 64, h3, w3),
                    node_feats=node_feats,
                    node_coords=centers,
                    batch_idx=batch_idx_tensor
                )
                graph_feat_map[b] = batch_graph_feat[0]

        # 解码器融合
        x = self.up1(x4, x3)  # 带HLAEM的高层解码
        x = self.up2(x, x2)  # 不带HLAEM的中层解码
        x = self.up3(x, x1, graph_feat=graph_feat_map)  # 带HLAEM的最后解码

        # 输出层
        logits = self.out_conv(x)
        return {"out": logits}


if __name__ == "__main__":
    # 测试网络输出尺寸
    model = HAG_Net(in_channels=3, num_classes=1, use_graph=True)
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(f"输入尺寸: {input_tensor.shape}")
    print(f"输出尺寸: {output['out'].shape}")  # 应保持 [1, 1, 256, 256]
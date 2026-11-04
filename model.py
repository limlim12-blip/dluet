import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from base.base_model import BaseModel


# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
class Chomp1d(BaseModel):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x shape: (Batch, Channels, len)
        if self.chomp_size > 0:
            return x[:, :, : -self.chomp_size].contiguous()
        return x


def weight_norm_conv(n_in, n_out, kernel_size, stride, padding, dilation):
    conv = nn.Conv1d(
        n_in, n_out, kernel_size, stride=stride, padding=padding, dilation=dilation
    )
    nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    return weight_norm(conv)


class TemporalBlock(BaseModel):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout
    ):
        super().__init__()

        # HACK: # 1602.07868v3.pdf
        # for short, weight have direction and length, normalize here mean break down that weight to g(length) and unit vector v/|v| (direction) so that g and v can grow independently
        # + it can self stabelize it self, g/|v| down if norm v up
        self.conv1 = weight_norm_conv(
            n_inputs, n_outputs, kernel_size, stride, padding, dilation
        )
        # cut future peek
        self.chomp1 = Chomp1d(padding)
        self.l_relu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm_conv(
            n_outputs, n_outputs, kernel_size, stride, padding, dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.l_relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        # (batch, chanels, len)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.l_relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.l_relu2,
            self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.l_relu = nn.PReLU()
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.l_relu(out + res)


class EmbedTCN(BaseModel):
    def __init__(
        self,
        kernel_size,
        dropout,
        embedding_dim,
        num_channels,
        num_inputs=3,
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim
        self.embedding_head = nn.Linear(num_channels[-1], embedding_dim)

    def forward(self, x):
        # (Batch, channels, len)
        tcn_out = self.tcn(x)
        tcn_out = tcn_out.transpose(1, 2)
        # (Batch, len, embed_dim)
        embeddings = self.embedding_head(tcn_out)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        assert torch.isfinite(embeddings).all(), "embed"

        return embeddings


class ArcFace(BaseModel):
    def __init__(
        self,
        kernel_size,
        dropout,
        num_channels,
        embedding_dim,
        num_classes,
        num_sub_classes,
        s,
        m,
        drop_center_prob,
        num_inputs=3,
    ):
        super().__init__()
        self.drop_center_prob = drop_center_prob
        self.num_classes = num_classes
        self.num_sub_classes = num_sub_classes
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes * num_sub_classes, embedding_dim)
        )
        self.embed_tcn = EmbedTCN(
            kernel_size, dropout, embedding_dim, num_channels, num_inputs
        )
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("m", torch.tensor(m))
        self.register_buffer("s", torch.tensor(s))

    def _head(self, embed, labels, seq_len):
        embedding = embed.reshape(-1, self.embedding_dim).float()
        weight = self.weight.float()

        # (B,L,N*S)
        cos = F.linear(F.normalize(embedding), F.normalize(weight))
        # B*L,N,S
        cos = cos.view(-1, self.num_classes, self.num_sub_classes)
        if self.training and self.drop_center_prob > 0:
            mask = torch.rand(cos.shape, device=cos.device) > self.drop_center_prob
            fallback = torch.randint(
                0, self.num_sub_classes, size=cos.shape[:-1], device=cos.device
            ).unsqueeze(-1)
            mask.scatter_(dim=-1, index=fallback, value=True)
            cos = torch.where(mask, cos, -1e9)
        # B*L,N
        cos_s, _ = torch.max(cos, dim=-1)
        plain = self.s * cos_s
        if not self.training or labels is None:
            return plain, plain, embed

        # B*S,1
        labels = labels.repeat_interleave(seq_len).view(-1, 1)
        cos_c = cos_s.clamp(-1.0 + 1e-3, 1.0 - 1e-3).float()
        # B*S
        gt_cos = cos_c.gather(1, labels).squeeze(1)
        # gt_theta = torch.arccos(gt_cos)
        margin = self.m.to(gt_cos.dtype)
        cos_m = torch.cos(margin)
        sin_m = torch.sin(margin)
        th = torch.cos(torch.pi - margin)
        mm = torch.sin(torch.pi - margin) * margin

        sin_theta = torch.sqrt((1.0 - gt_cos.pow(2)).clamp(min=1e-7))
        cos_theta_m = gt_cos * cos_m - sin_theta * sin_m

        gt_phi = torch.where(gt_cos > th, cos_theta_m, gt_cos - mm).float()
        is_gt = F.one_hot(labels.squeeze(1), self.num_classes).bool()  # (B*L, N)
        logits = torch.where(is_gt, gt_phi.unsqueeze(1), cos_c)
        logits.clamp_(min=-100, max=100)
        return self.s * logits, plain, embed

    def forward(self, data, labels=None):
        # data (B, C, L)
        # label (B,)
        # weight (N*S,D)
        seq_len = data.shape[-1]
        # (B,L,D)
        embed = self.embed_tcn(data)
        # with torch.autocast("cuda", enabled=False):
        return self._head(embed, labels, seq_len)


def cross_entropy(output, target):
    return F.cross_entropy(output, target)


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = torch.sum(pred == target.view(-1, 1)).item()
    return correct / len(target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets, reduction="none"):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss


# class TemporalAttention(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.attention_weights = nn.Sequential(
#             nn.Conv1d(channels, channels // 2, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv1d(channels // 2, 1, kernel_size=1),
#             nn.Softmax(dim=-1),
#         )
#
#     def forward(self, x):
#         weights = self.attention_weights(x)  # Shape: (Batch, 1, Length)
#
#         out = x * weights  # Shape: (Batch, Channels, Length)
#         return out, weights

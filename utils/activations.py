# Activation functions

import torch
import torch.nn as nn
import torch.nn.functional as F


# Swish https://arxiv.org/pdf/1905.02244.pdf ---------------------------------------------------------------------------
class Swish(nn.Module):  #
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


class MemoryEfficientSwish(nn.Module):
    # 节省内存的Swish 不采用自动求导(自己写前向传播和反向传播) 更高效
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            # save_for_backward会保留x的全部信息(一个完整的Autograd Function的Variable),
            # 并提供避免in-place操作导致的input在backward被修改的情况.
            # in-place操作指不通过中间变量计算的变量间的操作。
            ctx.save_for_backward(x)
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            # 此处saved_tensors[0] 作用同上面save_for_backward
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            # 返回该激活函数求导之后的结果
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x):  # 应用前向传播方法
        return self.F.apply(x)


# Mish https://github.com/digantamisra98/Mish --------------------------------------------------------------------------
class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    """
        一种高效的Mish激活函数，不采用自动求导(自己写前向传播和反向传播)，更高效
    """
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            # 前向传播
            # save_for_backward函数可以将对象保存起来，用于后续的backward函数
            # 会保留此input的全部信息，并提供避免in_place操作导致的input在backward中被修改的情况
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            # 反向传播
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


# FReLU https://arxiv.org/abs/2007.11824 -------------------------------------------------------------------------------
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        # 定义漏斗条件T(x)，参数池窗口（Parametric Pooling Window ）来创建空间依赖
        # nn.Con2d(in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=True)
        # 使用，深度可分离卷积 DepthWise Separable Conv + BN 实现T(x)
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))

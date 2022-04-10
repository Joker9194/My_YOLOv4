# PyTorch utils

import logging
import math
import os
import time
from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

logger = logging.getLogger(__name__)


# 定义的用于同步不同进程对数据读取的上下文管理器
@contextmanager     # 装饰器 为上下文管理模块
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    让所有进程在分布式训练中等待每一个local_master做事情的修饰器
    """
    if local_rank not in [-1, 0]:   # 判断local进程是否是主进程
        # 如果不是主进程 该函数会产生一个阻挡 限制这个进程的进行 直到所有进程进行同步
        torch.distributed.barrier()
    yield   # 中断后执行上下文代码，然后返回到此处继续往下执行
    if local_rank == 0:
        """
        如果是主进程 则设置阻挡 上述if中非主进程均已阻挡 
        等待主进程完成后 此时所有进程完成同步 并可以同时完成释放
        """
        torch.distributed.barrier()


# 初始化相关种子并确定训练模式。
def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html 速度与可重复性之间的权衡
    torch.manual_seed(seed)     # 为CPU设置随机种子
    """
        benchmark模式会自动寻找最优配置 但由于计算的随机性 每次网络进行前向反馈时会有差异
        避免这样差异的方式就是将deterministic设置为True（该设置表明每次卷积的高效算法均相同）
    """
    if seed == 0:  # slower, more reproducible 慢，高可重复性
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3' 可输入的设备形式
    cpu_request = device.lower() == 'cpu'   # 如果device输入为CPU 则cpu_request为True
    if device and not cpu_request:  # if device requested other than 'cpu'  如果设备请求的不是cpu
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable 设置环境变量
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()  # 当使用CPU时cuda被设置为False
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()      # 返回GPU数量
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count 检查batch_size是否能被显卡数整除
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]    # x为每个可用显卡相关属性
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            logger.info("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        logger.info(f'Using torch {torch.__version__} CPU')

    logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    """
        基于pytorch的精准时间测量
        如果cuda可用则执行synchronize函数
        该函数会等待当前设备上的流中的所有核心全部完成 这样测量时间便会准确 因为pytorch中程序为异步执行
    """
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def is_parallel(model):
    """
        判断模型是否并行
        返回值为True/False 代表model的类型 是否在后续tuple内
    """
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    # 返回字典 da 中的键值对，要求键k在字典db中且全部都不在exclude中，同时da中值的shape对应db中值的shape 需相同
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


# 模型初始化权重
def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:  # 如果是2维卷积层则跳过
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:   # 如果是2维BN层 则设置相关参数如下
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    # 找到model各个层里匹配mclass种类的层的索引
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    # 返回模型整体0参数占所有参数的比例
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()          # a为模型的总参数量
        b += (p == 0).sum()     # b为模型参数为0的数量
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    # 对模型进行剪枝，以增加稀疏性
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # 此处为非结构化剪枝操作 将计算不重要的参数规为0
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent 彻底移除被剪掉的权重
    print(' %.3g global sparsity' % sparsity(model)) # 返回模型的稀疏度


# conv代表torch支持的卷积层 bn代表torch支持的卷积层
def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    # 融合卷积与BN层 https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    # 将BN层写为1*1卷积层的形式 能够节省计算资源并简化网络结构
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters 准备滤波器（权重） 并进行替换
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

    # prepare spatial bias  准备空间偏置 并进行替换
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    # 综合上述两步 该卷积层在数学表达上等价于BN操作
    # 返回fuseconv相关配置
    return fusedconv


def model_info(model, verbose=False, img_size=640):     # verbose意为冗长的
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    # 模型信息. img_size 可能为 int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters  模型总参数量
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients 需要求梯度的参数量
    if verbose:
        # 格式化输出字符串
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            # 输出模型参数相关信息
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        flops = profile(deepcopy(model), inputs=(torch.zeros(1, 3, img_size, img_size),), verbose=False)[0] / 1E9 * 2
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.9f GFLOPS' % (flops)  # 640x640 FLOPS
    except (ImportError, Exception):
        fs = ''

    logger.info(f"\nModel Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


# 通过改变预训练的backbone 并重置全连接层来构造分类器
def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    # 加载torchvision中pretrained模型 reshape为n类输出
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    # reshape的过程是将fc层的权重和偏置清0 并将输出改为类别个数
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


# 实现对图片的缩放
def scale_img(img, ratio=1.0, same_shape=False):  # img(16,3,256,416), r=ratio
    # scales img(bs,3,y,x) by ratio
    # 对img进行缩放 gs代表最终图片的元素点数目必须被32整除以满足浮点数计算要求
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize 用torch自带的插值函数进行resize
        if not same_shape:  # pad/crop img 如果不保持相同 则将按比例放缩的h和w当做输出图片尺度
            gs = 32  # (pixels) grid size
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        # 将放缩的部分和要求的图片尺寸部分 不相交的部分 用imagenet均值填充
        # Q:如果ratio大于1 且same_shape=False 时 w-s[1]<0 此时是否会报错？ A:math.ceil()确保不会出现这种情况
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


# 复制实例对象的属性
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    # 复制属从b到a, options to only include [...] and to exclude [...]
    # .__dict__返回一个类的实例的属性和对应取值的字典
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)    # 将对象a的属性k赋值v


# 为模型的指数加权平均方法 使模型更具有鲁棒性
class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    在模型state_dict中保留所有内容的移动平均值（参数和缓冲区）。
    权重的平滑版本对于某些训练方案表现好十分必要.
    这个类对按照模型初始化顺序进行初始化十分敏感.
    GPU分配与分布式训练的包装器。
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        parameters:
        @model: 模型
        @decay: 衰减函数参数，默认0.9999，考虑过去10000次的真实值
        @updates: ema更新次数
        """
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates 更新次数
        # self.decay为一个衰减函数 输入变量为x
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():     # 参数取消设置梯度
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)    # 随着更新的次数设置衰减
            # msd为模型配置的字典
            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d  # 对浮点数进行衰减
                    # .detach函数 使对应Variables与网络隔开进而不参与梯度更新
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

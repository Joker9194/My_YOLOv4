# General utils

import glob
import logging
import math
import os
import platform
import random
import re
import subprocess
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import yaml

from utils.google_utils import gsutil_getsize
from utils.metrics import fitness, fitness_p, fitness_r, fitness_ap50, fitness_ap75, fitness_ap, fitness_f
from utils.torch_utils import init_torch_seeds

# Set printoptions
# 设置运行相关的一些基本的配置
# 控制print打印torch.tensor格式设置: tensor精度为5(小数点后5位)，每行字符数为320个，显示方法为long
torch.set_printoptions(linewidth=320, precision=5, profile='long')
# 控制print打印np.array格式设：精度为5，每行字符数为320个
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
# 阻止opencv参与多线程(与Pytorch的Dataloader不兼容)
cv2.setNumThreads(0)


def set_logging(rank=-1):
    """
    对日志的设置（format、level等）进行初始化
    """
    logging.basicConfig(
        # 设置日志输出的格式和内容，只打印日志消息
        format="%(message)s",
        # 设置日志级别
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def init_seeds(seed=0):
    """设置一系列随机数种子"""
    # 设置随机数，针对使用random.random()生成随机数的时候相同
    random.seed(seed)
    # 设置随机数，针对使用np.random.rand()生成随机数的时候相同
    np.random.seed(seed)
    # 为CPU设置种子，用于生成随机数时相同，并确定训练模式
    init_torch_seeds(seed)


def get_latest_run(search_dir='.'):
    """
    用于返回该项目中最近的模型，'last.pt'对应的路径
    @param search_dir: 要搜索的文件的根目录，默认是'.'，即当前路径
    @return:
    """
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    # 从Python版本3.5开始, glob模块支持该"**"指令（仅当传递recursive标志时才会解析该指令)
    # glob.glob函数匹配所有的符合条件的文件, 并将其以list的形式返回
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    # os.path.getctime 返回路径对应文件的创建时间
    # 故这里是返回路径列表中创建时间最近(最近的last文件)的路径
    return max(last_list, key=os.path.getctime) if last_list else ''


def check_git_status():
    """检查当前代码版本是否是最新的   如果不是最新的 会提示使用git pull命令进行升级"""
    # Suggest 'git pull' if repo is out of date
    # 判断系统平台是否是linux或者darwin，以及是否存在docker环境
    if platform.system() in ['Linux', 'Darwin'] and not os.path.isfile('/.dockerenv'):
        # 并创建子进程进行执行cmd命令，返回执行结果
        s = subprocess.check_output('if [ -d .git ]; then git fetch && git status -uno; fi', shell=True).decode('utf-8')
        if 'Your branch is behind' in s:
            print(s[s.find('Your branch is behind'):s.find('\n\n')] + '\n')


def check_img_size(img_size, s=32):
    """
    这个函数主要用于train.py中和detect.py中，用来检查图片的长宽是否符合规定
    检查img_size是否能被s整除，这里默认s为32，train.py test.py中设置64
    返回大于等于img_size且是s的最小倍数
    """
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_file(file):
    """
    Search for file if not found
    检查相关文件路径是否找到文件，并返回文件名
    @param file:
    @return:
    """
    # 如果file是文件或file为''，直接返回文件名
    if os.path.isfile(file) or file == '':
        return file
    else:
        # 否则，传进来的就是当前项目下的一个全局路径，查找匹配的文件名，返回第一个
        # glob.glob: 匹配当前项目下的所有项目 返回所有符合条件的文件files
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)  # assert unique
        # 返回第一个匹配到的文件名
        return files[0]  # return file


def check_dataset(dict):
    """
    用在train.py和detect.py中 检查本地有没有数据集
    检查数据集 如果本地没有则从torch库中下载并解压数据集
    :params data: 是一个解析过的data_dict   len=7
        例如: ['path'='../datasets/coco128', 'train', 'val', 'test', 'nc', 'names', 'download']
    """
    # Download dataset if not found locally
    val, s = dict.get('val'), dict.get('download')
    if val and len(val):
        # path.resolve() 该方法将一些的 路径/路径段 解析为绝对路径
        # docker环境下 val = [PosixPath('/yolo/breast/val.txt')]
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        # 如果val不存在 说明本地不存在数据集
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            # 如果下载地址s不为空, 就直接下载
            if s and len(s):  # download script
                print('Downloading %s ...' % s)
                # 如果下载地址s是http开头就从url中下载数据集
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    # 开始下载 利用torch.hub.download_url_to_file函数从s路径中下载文件名为f的文件
                    torch.hub.download_url_to_file(s, f)
                    # 执行解压命名 将文件f解压到root地址 解压后文件名为f
                    r = os.system('unzip -q %s -d ../ && rm %s' % (f, f))  # unzip
                # 否则执行bash指令下载数据集
                else:  # bash script
                    r = os.system(s)
                print('Dataset autodownload %s\n' % ('success' if r == 0 else 'failure'))  # analyze return value
            else:
                raise Exception('Dataset not found.')


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    # 取大于等于x，且是divisor的倍数的最小数
    return math.ceil(x / divisor) * divisor


def labels_to_class_weights(labels, nc=80):
    """用在train.py中，得到每个类别的权重，标签频率高的类权重低
    从训练(gt)标签获得每个类的权重，标签频率高的类权重低
    Get class weights (inverse frequency) from training labels
    @param labels: gt框的所有真实标签labels
    @param nc:  数据集的类别数
    @return: torch.from_numpy(weights): 每一个类别根据labels得到的占比(次数越多权重越小) tensor
    """
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    # classes: 所有标签对应的类别labels   labels[:, 0]: 类别   .astype(np.int): 取整
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    # weightss: 返回每个类别出现的次数，(1, nc)
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    # 将出现0次的类别权重全部取1
    weights[weights == 0] = 1  # replace empty bins with 1
    # 其他所有类别的权重全部取次数的倒数
    weights = 1 / weights  # number of targets per class
    # 求出每一类别的占比
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    """用在train.py中 利用上面得到的每个类别的权重得到每一张图片的权重  再对图片进行按权重进行采样
    通过每张图片真实gt框的真实标签labels和上一步labels_to_class_weights得到的每个类别的权重进行采样
    Produces image weights based on class mAPs
    @param labels: 每张图片真实gt框的真实标签
    @param nc: 数据集的类别数，默认80
    @param class_weights: labels_to_class_weights得到的每个类别的权重
    @return:
    """
    n = len(labels)
    # class_counts: 每个类别出现的次数。[num_labels, nc]: 每一行是当前这张图片每个类别出现的次数, num_labels=图片数量=label数量
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    # [80] -> [1, 80]
    # 整个数据集的每个类别权重[1, 80] * 每张图片的每个类别出现的次数[num_labels, 80] = 得到每一张图片每个类对应的权重[128, 80]
    # 另外注意: 这里不是矩阵相乘, 是元素相乘. [1, 80]和每一行图片的每个类别出现的次数[1, 80]分别按元素相乘
    # 再sum(1): 按行相加, 得到最终image_weights: 得到每一张图片对应的采样权重
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    """将预测信息从xyxy格式转为xywh格式, 再保存
    Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    @param x: [n, x1y1x2y2]
    @return: y: [n, xywh]
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """ Rescale coords (xyxy) from img1_shape to img0_shape
    将预测坐标从feature map映射回原图
    @param img1_shape: coords相对于的shape大小
    @param coords: 要进行缩放的box坐标信息
    @param img0_shape: 要将coords缩放到相对的目标shape大小
    @param ratio_pad: 缩放比例gain和pad值，None就是先计算gain和pad再pad+scale，不为空就直接pad+scale
    @return:
    """
    # 先计算缩放比例gain和pad值
    if ratio_pad is None:  # calculate from img0_shape
        # 取高宽缩放比例中较小的，之后还可以再pad，即先对宽高较长部分进行缩放，再对短边进行填充
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        # wh中有一个为0，pad另一个
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]  # 制定比例
        pad = ratio_pad[1]  # 制定pad值
    # 因为pad = img1_shape - img0_shape, 所以要把尺寸从img1 -> img0 就同样也需要减去pad
    # 如果img1_shape>img0_shape  pad>0   coords从大尺寸缩放到小尺寸 减去pad 符合
    # 如果img1_shape<img0_shape  pad<0   coords从小尺寸缩放到大尺寸 减去pad 符合
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    # 缩放scale
    coords[:, :4] /= gain
    # 防止放缩后的坐标过界 边界处直接剪切
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # 将boxes的坐标(x1y1x2y2 左上角右下角)限定在图像的尺寸(img_shape hw)内
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, ECIoU=False, eps=1e-9):
    """
    ciou = iou - p2 / c2 - av
    @param box1:    pred_xywh  [4, 186]
    @param box2:    label_xywh  [186, 4]
    @param x1y1x2y2: box1和box2的格式是否为左上角点和右下角点
    @param GIoU:    GIoU Loss
    @param DIoU:    DIoU Loss
    @param CIoU:    CIoU Loss
    @param EIoU:    EIoU Loss
    @param ECIoU:   ECIoU Loss
    @param eps:
    @return:
    """
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1 (box里面的数字存的都是左上右下xy的坐标）
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy 将中心点+宽高变成 左上右下两点
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area 交集
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area 并集
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU or EIoU or ECIoU:
        # 最小包围框的宽高
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU or EIoU or ECIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # 最小包围框的对角线长度
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            # 两个框的中心点距离
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIoU: # Efficient IoU https://arxiv.org/abs/2101.08158
                rho3 = (w1-w2) **2
                c3 = cw ** 2 + eps
                rho4 = (h1-h2) **2
                c4 = ch ** 2 + eps
                return iou - rho2 / c2 - rho3 / c3 - rho4 / c4  # EIoU
            elif ECIoU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                rho3 = (w1-w2) **2
                c3 = cw ** 2 + eps
                rho4 = (h1-h2) **2
                c4 = ch ** 2 + eps
                return iou - v * alpha - rho2 / c2 - rho3 / c3 - rho4 / c4  # ECIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """ Performs Non-Maximum Suppression (NMS) on inference results
    @param prediction: [batch, num_anchors(3个yolo预测层), (x+y+w+h+1+num_classes)] = [1, 18900, 25]  3个anchor的预测结果总和
    @param conf_thres: conf_thres: 先进行一轮筛选，将分数过低的预测框（<conf_thres）删除（分数置0）
    @param iou_thres: iou阈值, 如果其余预测框与target的iou>iou_thres, 就将那个预测框置0
    @param merge: use merge-NMS，多个bounding box给它们一个权重进行融合  默认False
    @param classes: 是否nms后只保留特定的类别 默认为None
    @param agnostic: 进行nms是否也去除不同类别之间的框 默认False
    @return: detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections 是否需要冗余的detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()  # 记录当前时刻时间
    # batch-size个output，存放最终筛选后的预测结果
    output = [torch.zeros(0, 6)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # 根据conf-thres滤除背景目标（置信度极低的目标）
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        # 经过上面过滤后如果该feature map没有目标框了，就结束这轮直接进行下一张图
        if not x.shape[0]:
            continue

        # Compute conf 计算conf_score
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        # 将box中xywh变成xyxy的形式
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            # 针对每个类别score(obj_conf * cls_conf) > conf_thres
            # 这里一个框是有可能有多个物体的，所以要筛选
            # nonzero: 获得矩阵中的非0(True)数据的下标  a.t(): 将a矩阵拆开
            # i: 下标 [43]   j: 类别index [43] 过滤了两个score太低的
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            # conf: 每个类别的最大置信度分数  j: 对应类的下标
            conf, j = x[:, 5:].max(1, keepdim=True)
            # 每个类别的最大置信度分数得超过conf_thres阈值
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class 是否只保留特定的类别，默认None
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:   # 如果经过前面过滤该feature map没有目标框了，就结束这轮直接进行下一张图
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # 做个切片 得到boxes和scores   不同类别的box位置信息加上一个很大的数但又不同的数c
        # 这样作非极大抑制的时候不同类别的框就不会掺和到一块了  这是一个作nms挺巧妙的技巧
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # 返回nms过滤后的bounding box(boxes)的索引（降序排列）
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def strip_optimizer(f='weights/best.pt', s=''):  # from utils.general import *; strip_optimizer()
    """
    用在train.py模型训练完后
    将optimizer、training_results、updates...从保存的模型文件f中删除
    Strip optimizer from 'f' to finalize training, optionally save as 's'
    @param f: 传入的原始保存的模型文件
    @param s: 删除optimizer等变量后的模型保存的地址 dir
    """
    # x: 为加载训练的模型
    x = torch.load(f, map_location=torch.device('cpu'))
    # 以下模型训练涉及到的若干个指定变量置空
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1  # 模型epoch恢复初始值-1
    #x['model'].half()  # to FP16
    #for p in x['model'].parameters():
    #    p.requires_grad = False
    # 保存模型 x -> s/f
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print('Optimizer stripped from %s,%s %.1fMB' % (f, (' saved as %s,' % s) if s else '', mb))


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    """ 用在train.py的进化超参结束后
    打印进化后的超参结果和results到evolve.txt和hyp_evolved.yaml中
    Print mutation results to evolve.txt (for use with train.py --evolve)
    @param hyp: 进化后的超参 dict {28对 key:value}
    @param results: tuple(7)   (mp, mr, map50, map50:95, box_loss, obj_loss, cls_loss)
    @param yaml_file: 要保存的进化后的超参文件名  runs\train\evolve\hyp_evolved.yaml
    @param bucket:
    @return:
    """
    # 定义相关变量 并赋值 按指定格式输出
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        url = 'gs://%s/evolve.txt' % bucket
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
            os.system('gsutil cp %s .' % url)  # download evolve.txt if larger than local

    # 将结果c(results)和b(得到所有超参的value)写入evolve.txt中
    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml 保存yaml配置文件 为'hyp_evolved.yaml'
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])  # 将hyp保存到数组hyp[]中
    with open(yaml_file, 'w') as f:  # 将hyp写入yaml_file
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)

    if bucket: # 如果需要存到谷歌云盘, 就上传。默认是不需要的
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))  # upload


def apply_classifier(x, model, img, im0):
    """ applies a second stage classifier to yolo outputs
    用在detect.py文件的nms后继续对feature map送入model2 进行二次分类
    定义了一个二级分类器来处理yolo的输出, 当前实现本质上是一个参考起点，您可以使用它自行实现此项
    比如你有照片与汽车与车牌, 你第一次剪切车牌, 并将其发送到第二阶段分类器, 以检测其中的字符
    @param x: yolo层的输出
    @param model: 分类模型
    @param img: 进行resize + pad之后的图片
    @param im0: 原尺寸的图片
    @return:
    """
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def increment_path(path, exist_ok=True, sep=''):
    """ 这是个用处特别广泛的函数
    递增路径 如 run/train/exp --> runs/train/exp{sep}0, runs/exp{sep}1 etc.
    @param path: 路径
    @param exist_ok: True
    @param sep: 文件名的后缀  默认''
    @return:
    """
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    # 如果文件路径存在且有参数exist_ok，或者文件路径不存在，直接返回str
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        # 模糊搜索和path\sep相似的路径, 存在一个list列表中 如['runs\\train\\exp', 'runs\\train\\exp1']
        # f开头表示在字符串内支持大括号内的python表达式
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # r的作用是去除转义字符       path.stem: 没有后缀的文件名 exp
        # re 模糊查询模块  re.search: 查找dir中有字符串'exp/数字'的d   \d匹配数字
        # matches [None, <re.Match object; span=(11, 15), match='exp1'>]  可以看到返回span(匹配的位置) match(匹配的对象)
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        # 生成需要生成文件的exp后面的数字 n = max(i) + 1 = 2
        n = max(i) + 1 if i else 2  # increment number
        # 返回path, 如runs/train/exp2
        return f"{path}{sep}{n}"  # update path

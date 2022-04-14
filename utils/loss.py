# Loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """
    用在ComputeLoss类中
    标签平滑操作  [1, 0]  =>  [0.95, 0.05]
    @param eps: 平滑参数
    @return: return positive, negative label smoothing BCE targets 两个值分别代表正样本和负样本的标签取值
            原先的正样本=1 负样本=0 改为 正样本=1.0 - 0.5 * eps  负样本=0.5 * eps
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """用在ComputeLoss类的__init__函数中
    BCEwithLogitLoss() with reduced missing label effects.
    https://github.com/ultralytics/yolov5/issues/1030
    The idea was to reduce the effects of false positive (missing labels) 就是检测成正样本了 但是检测错了
    """
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """
    用在代替原本的BCEcls（分类损失）和BCEobj（置信度损失）
    Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    论文: https://arxiv.org/abs/1708.02002
    https://blog.csdn.net/qq_38253797/article/details/116292496
    TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
    """
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()=Sigmoid+BCELoss  定义为多分类交叉熵损失函数
        self.gamma = gamma      # 参数gamma  用于削弱简单样本对loss的贡献程度
        self.alpha = alpha      # 参数alpha  用于平衡正负样本个数不均衡的问题
        self.reduction = loss_fcn.reduction     # 控制FocalLoss损失输出模式 sum/mean/none   默认是Mean
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element 需要将Focal loss应用于每一个样本之中

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)    # 正常BCE的loss:   loss = -log(p_t)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        pred_prob = torch.sigmoid(pred)  # prob from logits
        # true=1: p_t=pred_prob    true=0: p_t=1-pred_prob
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        # true=1: alpha_factor=self.alpha    true=0: alpha_factor=1-self.alpha
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma   # 这里代表Focal loss中的指数项
        # 返回最终的loss=BCE * 两个参数  (看看公式就行了 和公式一模一样)
        loss *= alpha_factor * modulating_factor

        # 最后选择focalloss返回的类型 默认是mean
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def compute_loss(p, targets, model):  # predictions, targets, model
    """
    @param p: 预测框，由模型构建中的三个检测头Detector返回的三个yolo层的输出
              tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
               如: [16, 3, 80, 80, 6]、[16, 3, 40, 40, 6]、[16, 3, 20, 20, 6]
               [bs, anchor_num， grid_h, grid_w, xywh+class+classes]
               可以看出来这里的预测值p是三个yolo层batch里面每个图像每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
    @param targets: 数据增强后的真实框 [29, 6] [num_target,  image_index+class+xywh] xywh为归一化后的框
    @param model: 模型
    @return: loss 总损失 (lbox, lobj, lcls, loss) box损失， 置信度损失， 分类损失， 单个图像损失
    """
    device = targets.device
    # print(device)
    # 用来保存三层特征图的损失
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    # 获取对应的gt和anchor
    # tcls: 表示这个target所属的class index
    # tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
    # indices: b: 表示这个target属于的image index
    #          a: 表示这个target使用的anchor index
    #          gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
    #          gi: 表示这个网格的左上角x坐标
    # anchors: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets

    # hyper parameters
    h = model.hyp

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    # 标签平滑  eps=0代表不做标签平滑-> cp=1 cn=0  eps!=0代表做标签平滑 cp代表positive的标签值 cn代表negative的标签值
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss g=0 代表不用Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    no = len(p)  # number of outputs
    # balance用来设置三个feature map对应输出的置信度损失系数(平衡三个feature map的置信度损失)
    # 从左到右分别对应大feature map(检测小目标)到小feature map(检测大目标)
    # 思路:  It seems that larger output layers may overfit earlier, so those numbers may need a bit of adjustment
    #       一般来说，检测小物体的难度大一点，所以会增加大特征图的损失系数，让模型更加侧重小物体的检测
    # 如果no=3就返回[4.0, 1.0, 0.4]否则返回[4.0, 1.0, 0.4, 0.1]
    # 如果no=5就返回[4.0, 1.0, 0.5, 0.4, 0.1]， 不然就返回[4.0, 1.0, 0.4]
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    balance = [4.0, 1.0, 0.5, 0.4, 0.1] if no == 5 else balance
    # i: 第几个检测头 pi： 对应的预测框， 对每个特征图计算损失
    for i, pi in enumerate(p):  # layer index, layer predictions
        # 获取该层特征图上的gt信息: 图像序号, anchor序号, 位于特征图上的格网坐标
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        # tboj: [16, 3, grid_h, grid_w] 存储gt中的置信度真值
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:   # 有gt才计算分类和回归损失，否则只计算置信度损失
            nt += n  # cumulative targets 目标个数的累计
            # 获取真值（targets 本身正样本+附近4个grid的正样本）对应的预测框box信息
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets 与目标对应的预测子集

            # Regression 对预测值进行预处理（用的yolov5的策略）
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            # 计算ciou
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)

            #
            # d focal loss
            # if h['dfl']:
            #     DFLbox = DistributionFocalLoss(loss_weight=0.25)
            #     gain = torch.tensor(p[i].shape)[[3, 2, 3, 2]].to(device)
            #     tbox_centers = (tbox[i][:, :2] + tbox[i][:, 2:] / 2.).to(device)
            #     tbox_centers = torch.cat([tbox_centers, tbox[i][:, :2]], 1) * gain
            #     tbox_centers = tbox_centers.reshape(-1)
            #     pbox = pbox.reshape(-1, 1)
            #     ldfl = DFLbox(pbox, tbox_centers)

            # box坐标回归损失
            lbox += (1.0 - iou).mean()  # iou loss

            # Objectness
            # 利用IoU对gt中的置信度进行加权（对应与build_targets中的gt扩充）
            # model.gr=1，完全使用标签框与预测框的iou值来作为该预测框的objectness标签
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

            # Classification
            # 计算分类损失
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
        # 置信度损失
        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / no  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if no >= 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model):
    """
    @param p: 预测框，有模型构建中的三个检测头Detector返回的三个yolo层的输出
              tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
               如: [16, 3, 80, 80, 6]、[16, 3, 40, 40, 6]、[16, 3, 20, 20, 6]
               [bs, anchor_num， grid_h, grid_w, xywh+class+classes]
               可以看出来这里的预测值p是三个yolo层batch里面每个图像每个grid_cell(每个grid_cell有三个预测值)的预测值,之后进行正样本筛选
    @param targets: 数据增强后的真实框 [29, 6] [num_target,  image_index+class+xywh] xywh为归一化后的框
    @param model: 模型
    @return:    tcls: 表示这个target所属的class index
                tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
                indices: b: 表示这个target属于的image index
                         a: 表示这个target使用的anchor index
                        gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
                        gi: 表示这个网格的左上角x坐标
                anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算

    """
    nt = targets.shape[0]  # number of targets
    tcls, tbox, indices, anch = [], [], [], []
    # gain是为了后面将targets=[nt, 6]中的归一化了的xywh映射到相对feature map尺度上
    # 6: image_index + class + xywh
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    # 这两个变量是用来扩展正样本的 因为预测框预测到target有可能不止当前的格子预测到了
    # 可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
    # 以自身 + 周围左上右下4个网格 = 5个网格  用来计算offsets
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets
    g = 0.5  # offset 中心偏移  用来衡量target中心点离哪个格子更近
    multi_gpu = is_parallel(model)
    # i：第i个yolo层(0-4)       jj：yolo所在的层数
    for i, jj in enumerate(model.module.yolo_layers if multi_gpu else model.yolo_layers):
        # get number of grid points and anchor vec for this yolo layer
        # anchor_vec = anchor / stride，即cfg文件中anchors的数/步长 shape: [anchor number, 2]
        anchors = model.module.module_list[jj].anchor_vec if multi_gpu else model.module_list[jj].anchor_vec
        # gain: 保存每个输出feature map的宽高 -> gain[2:6]=gain[]
        # [1, 1, 1, 1, 1, 1] -> [1, 1, 80, 80, 80, 80] = image_index + class + xywh
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        # t = [nt, 6]  将target中的xywh的归一化尺度放缩到相对当前feature map的坐标尺度
        #     [nt, image_index + class + xywh]
        a, t, offsets = [], targets * gain, 0
        if nt:  # 开始匹配
            na = anchors.shape[0]  # number of anchors
            # torch.arange(na): tensor([0,1,2]) shape=3
            # .view shape=[3,1]  repeat: 重复nt次，shape为[3， nt]
            # at: 每个anchor有29个target
            at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)
            # t=[nt, 6]   t[None, :, 4:6] = [1, nt, 2]=[1, 29, 2]
            # anchors[:, None]=[na, 1, 2] = [3, 1, 2]
            # r=[na, nt, 2]=[3, 29, 2], 先将t复制两份再做除法
            # 当前feature map的所有的targets与三个anchor的宽高比(w/w  h/h)
            r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
            # 筛选条件  GT与anchor的宽比或高比超过一定的阈值 就当作负样本
            # torch.max(r, 1. / r)=[3, 29, 2] 筛选出宽比w1/w2 w2/w1 高比h1/h2 h2/h1中最大的那个
            # .max(2)返回宽比 高比两者中较大的一个值和它的索引  [0]返回较大的一个值
            # j: [3, nt]  False: 当前gt是当前anchor的负样本  True: 当前gt是当前anchor的正样本
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare

            # at: 每个anchor有nt个target  a:正样本所在的anchor索引
            # t: [正样本数, 6]  t.repeat(na, 1, 1): 将t复制na份， [j]找出其中的正样本
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

            # overlaps
            # Offsets 筛选当前格子周围格子 找到2个离target中心最近的两个格子 可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
            # 除了target所在的当前格子外, 还有2个格子对目标进行检测(计算损失) 也就是说一个目标需要3个格子去预测(计算损失)
            # 首先当前格子是其中1个 再从当前格子的上下左右四个格子中选择2个 用这三个格子去预测这个目标(计算损失)
            # feature map上的原点在左上角 向右为x轴正坐标 向下为y轴正坐标
            # gxy: [正样本数, 2]  z: [正样本数, 2]
            gxy = t[:, 2:4]  # grid xy  取target中心的坐标xy(相对feature map左上角的坐标)
            z = torch.zeros_like(gxy)
            # 筛选中心坐标 距离当前grid_cell的左、上方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
            # ((gxy % 1. < g) & (gxy > 1.)).shape： [62, 2]
            # j: [62] bool 如果是True表示当前target中心点所在的格子的左边格子也对该target进行回归(后续进行计算损失)
            # k: [62] bool 如果是True表示当前target中心点所在的格子的上边格子也对该target进行回归(后续进行计算损失)
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            # 筛选中心坐标 距离当前grid_cell的右、下方偏移小于g=0.5 且 中心坐标必须小于宽高-1(坐标不能在边上 此时就没有4个格子了)
            # l: [62] bool 如果是True表示当前target中心点所在的格子的右边格子也对该target进行回归(后续进行计算损失)
            # m: [62] bool 如果是True表示当前target中心点所在的格子的下边格子也对该target进行回归(后续进行计算损失)
            l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
            # a: [186]， 本身正样本+周围左上右下四个网格的正样本对应的anchor索引
            # t: [186, 6], 本身正样本+周围左上右下四个网格的正样本的targets * gain
            a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
            # offsets: [186， 2]， 得到所有筛选后的网格的中心相对于这个要预测的真实框所在网格边界（左右上下边框）的偏移量
            offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

        # Define
        # 对应的图片、类别索引（编号）
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy 中心
        gwh = t[:, 4:6]  # grid wh 宽高
        gij = (gxy - offsets).long()
        # gj: 网格的左上角y坐标  gi: 网格的左上角x坐标
        gi, gj = gij.T  # grid xy indices

        # Append
        # indices.append((b, a, gj, gi))  # image, anchor, grid indices
        # b: image index  a: anchor index  gj: 网格的左上角y坐标  gi: 网格的左上角x坐标 clamp_(): 将gi的值限制在0到gain[3]-1之间
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch

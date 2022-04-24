# Model validation metrics

import matplotlib.pyplot as plt
import numpy as np


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def fitness_p(x):
    # Model fitness as a weighted combination of metrics
    w = [1.0, 0.0, 0.0, 0.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def fitness_r(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 1.0, 0.0, 0.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def fitness_ap50(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 1.0, 0.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def fitness_ap(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.0, 1.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def fitness_f(x):
    # Model fitness as a weighted combination of metrics
    #w = [0.0, 0.0, 0.0, 1.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return ((x[:, 0]*x[:, 1])/(x[:, 0]+x[:, 1]))


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, fname='precision-recall_curve.png'):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).     目标的置信度取值0-1
        pred_cls:  Predicted object classes (nparray).  预测目标类别
        target_cls:  True object classes (nparray).     真实目标类别
        plot:  Plot precision-recall curve at mAP@0.5   是否绘制P-R曲线 在mAP@0.5的情况下
        fname:  Plot filename   P-R曲线图的保存名称
    # Returns
        像faster-rcnn那种方式计算AP （这里涉及计算AP的两种不同方式 建议查询）
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness 将目标进行排序
    # np.argsort(-conf)函数返回一个索引数组 其中每一个数按照conf中元素从大到小 置为 0,1...n
    i = np.argsort(-conf)
    # tp conf pred_cls 三个矩阵均按照置信度从大到小进行排列
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes 找到各个独立的类别
    # np.unique()会返回输入array中出现至少一次的变量 这里返回所有独立的类别
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class 创建P-R曲线并计算每一个类别的AP
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    # 第一个为类别数目, 第二为IOU loss阈值的类别的 (i.e. 10 for mAP0.5...0.95)
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    # 初始化 对每一个类别在每一个IOU阈值下面 计算P R AP参数
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):  # ci为类别对应索引 c为具体的类别
        # i为一个包含True/False 的列表 代表 pred_cls array 各元素是否与 类别c 相同
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels 类别c 的个数 all_results
        n_p = i.sum()  # number of predictions 预测类别中为 类别c 的个数

        if n_p == 0 or n_l == 0:  # 如果没有预测到 或者 ground truth没有标注 则略过类别c
            continue
        else:
            # Accumulate FPs and TPs
            """ 
            计算 FP（False Positive） 和 TP(Ture Positive)
            tp[i] 会根据i中对应位置是否为False来决定是否删除这一位的内容，如下所示：
            a = np.array([0,1,0,1]) i = np.array([True,False,False,True]) b = a[i]
            则b为：[0 1]
            而.cumsum(0)函数会 按照对象进行累加操作，如下所示：
            a = np.array([0,1,0,1]) b = a.cumsum(0)
            则b为：[0,1,1,2]
            （FP + TP = all_detections 所以有 fp[i] = 1 - tp[i]）
            所以fpc为 类别c 按照置信度从大到小排列 截止到每一位的FP数目
                tpc为 类别c 按照置信度从大到小排列 截止到每一位的TP数目
            recall 和 precision 均按照元素从小到大排列
            """
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall = TP / (TP + FN) = TP / all_results = TP / n_l
            recall = tpc / (n_l + 1e-16)  # recall curve
            """
            np.interp() 函数第一个输入值为数值 第二第三个变量为一组x y坐标 返回结果为一个数值
            这个数值为 找寻该数值左右两边的x值 并将两者对应的y值取平均 如果在左侧或右侧 则取 边界值
            如果第一个输入为数组 则返回一个数组 其中每一个元素按照上述计算规则产生
            """
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision = TP / TP + FP = TP / all_detections
            precision = tpc / (tpc + fpc)  # precision curve
            # print("tpc: {}, fpc: {}".format(tpc, fpc))
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 score (harmonic mean of precision and recall P和R的调和平均值)
    f1 = 2 * p * r / (p + r + 1e-16)

    if plot:
        py = np.stack(py, axis=1)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(px, py, linewidth=0.5, color='grey')  # plot(recall, precision)
        ax.plot(px, py.mean(1), linewidth=2, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.legend()
        fig.tight_layout()
        fig.savefig(fname, dpi=200)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = recall  # np.concatenate(([0.], recall, [recall[-1] + 1E-3]))
    mpre = precision  # np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

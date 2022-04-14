import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.google_utils import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, clip_coords, set_logging, increment_path
from utils.loss import compute_loss
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized

from models.models import *


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


# 测试函数， 输入为测试过程中需要的各种参数
def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_conf=False,
         plots=True,
         log_imgs=0):  # number of logged images

    # Initialize/load model and set device
    # 初始化/加载模型，并设置设备
    training = model is not None  # 有模型则training为True
    if training:  # called by train.py 调用train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        # 调用general.py文件中的函数 设置日志 opt对象main中解析传入变量的对象
        set_logging()
        # 调用torch_utils中select_device来选择执行程序时的设备
        device = select_device(opt.device, batch_size=batch_size)
        # 获取保存测试之后的label文件路径，格式为txt
        save_txt = opt.save_txt  # save *.txt labels

        # Directories
        # 调用genera.py中的increment_path函数来设置保存文件的路径
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # 加载模型
        model = Darknet(opt.cfg).to(device)

        # load model
        try:
            ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except:
            load_darknet_weights(model, weights[0])
        # 调用general.py中的check_img_size函数来检查图像分辨率能否被64整除
        imgsz = check_img_size(imgsz, s=64)  # check img_size

    # Half
    # 如果设备类型不是cpu，则将模型由32位浮点数转换为16位浮点数
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    # 将模型转换为测试模式，固定住dropout层和BN层
    model.eval()
    # 判断输入的数据yaml是不是coco.yaml文件
    is_coco = data.endswith('coco.yaml')  # is COCO dataset
    # 加载数据集配置yaml文件，获取训练集、测试集、验证集和nc等相关信息
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.3, 0.75, 10).to(device)  # iou vector for mAP@0.3:0.75  mAP@0.3:0.75 的iou向量
    niou = iouv.numel()  # numel为pytorch预置函数 用来获取张量中的元素个数

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases wandb为可视化权重和各种指标的库
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        # 利用空图片对模型进行测试（只在运行设备不是CPU时进行）
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        # 如果任务为test，则调用yaml中的测试路径，否则调用验证路径
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, 64, opt, pad=0.5, rect=True)[0]

    seen = 0  # 初始化已完成测试的图片数量
    try:
        # 获取模型训练中存储的类别名字数据
        names = model.names if hasattr(model, 'names') else model.module.names
    except:
        names = load_classes(opt.names)
    coco91class = coco80_to_coco91_class()  # 调用general.py中的函数 来转换coco的类
    # 为后续设置基于tqdm的进度条作基础
    s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.75', 'mAP@.3:.75')
    # 初始化各种指标的值，t0, t1为时间
    p, r, f1, mp, mr, map50, map75, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    # 初始化网络的loss
    loss = torch.zeros(3, device=device)
    # 初始化json文件涉及到的字典、统计信息、AP、每一类的AP、图片汇总
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        # 将图片数据拷贝到device(GPU)上面
        img = img.to(device, non_blocking=True)
        # 将图片从64位精度转换为32位精度
        img = img.half() if half else img.float()  # uint8 to fp16/32
        # 将图片像素值0-255的范围归一化到0-1的范围
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 对targets也做同样的拷贝操作
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        # 之后的过程不生成计算图，开始运行模型
        with torch.no_grad():
            # Run model
            # 调用torch_utils中的函数，开始计时
            t = time_synchronized()
            # 输入图像进行模型推断，返回推断结果及训练结果
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            # t0为累计的各个推断所用时间
            t0 += time_synchronized() - t

            # Compute loss 计算损失
            if training:  # if model has loss hyperparameters 训练时是否进行test
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # box, obj, cls

            # Run NMS
            t = time_synchronized()
            # 调用general.py中的函数，进行非极大值抑制操作
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
            t1 += time_synchronized() - t

        # Statistics per image
        # si代表第si张图像，pred是对应图像预测的label信息
        for si, pred in enumerate(output):
            # 读取第si张图像的label信息
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            # 检测到的目标的类别 label矩阵的第一列
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1  # 处理的图像加1

            if len(pred) == 0:  # 如果没有预测到目标
                if nl:  # 同时有label信息
                    # stats初始化为一个空列表[] 此处添加一个空信息
                    # 添加的每一个元素均为tuple 其中第二第三个变量为一个空的tensor
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            path = Path(paths[si])
            # 将结果保存到文本文档中
            if save_txt:
                # shapes具体变量设置应看dataloader 此处应为提取长和宽并构建新tensor gn
                # 对torch.tensor()[[1,0]]操作可构建一个新tensor其中第一行为内层列表中第一个索引对应的行
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                x = pred.clone()  # 对pred进行深复制
                x[:, :4] = scale_coords(img[si].shape[1:], x[:, :4], shapes[si][0], shapes[si][1])  # to original
                for *xyxy, conf, cls in x:
                    """
                    将xyxy格式的坐标转换成xywh坐标 调用general.py中的函数
                    xyxy格式为记录bounding box 的左上角和右下角坐标
                    xywh格式为记录中心点坐标和bounding box的宽和高
                    """
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # line为按照YOLO格式输出的测试结果[类别 x y w h]
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    # 将上述test得到的信息输出保存 输出为xywh格式 coco数据格式也为xywh格式
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging 记录图片测试结果相关信息 并 存储在日志当中
            if plots and len(wandb_images) < log_imgs:
                # 一个包含嵌套字典的列表的数据结构，存储一个box对应的数据信息
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[int(cls)], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
                # 记录每一张图片 每一个box的相关信息 wandb_images 初始化为一个空列表
                # wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Clip boxes to image bounds
            # 将boxes的坐标(x1y1x2y2 左上角右下角)限定在图像的尺寸(height, width)内
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary 将信息添加到JSON字典当中
            if save_json:
                # 储存的格式
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                # 记录的信息有id box xy to top-left 得分等 如下所示
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            # 初始化时将所有的预测都当做错误
            # niou为iou阈值的个数
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:  # 当检测到的个数不为0时 nl为图片检测到的目标个数
                detected = []  # target indices
                tcls_tensor = labels[:, 0]  # 获得类别的tensor

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class 对于每一个目标类
                for cls in torch.unique(tcls_tensor):
                    # numpy.nonzero()函数为获得非零元素的所以 返回值为长为numpy.dim长度的tuple
                    # 因此ti为标签box对应的索引 ti = test indices
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    # 因此pi为预测box对应的索引 pi = prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections 对于每个单独的类别寻找检测结果
                    if pi.shape[0]:
                        # Prediction to target ious
                        # 调用general.py中的box_iou函数 返回最大的iou对应目标及相关指标
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections 将检测到目标统一添加到 detected_set集合当中
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target 检测到的目标
                            if d.item() not in detected_set:
                                # 将不在检测集合中的目标添加到集合里面
                                detected_set.add(d.item())
                                detected.append(d)
                                # 只有iou大于阈值的才会被认为是正确的目标
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            # # 向stats（list）中添加统计指标 格式为：(correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images 画出前三个图片的ground truth和 对应的预测框
        # if plots and batch_i < 3:
        #     f = save_dir / f'test_batch{batch_i}_labels.jpg'  # filename
        #     plot_images(img, targets, paths, f, names)  # labels
        #     f = save_dir / f'test_batch{batch_i}_pred.jpg'
            # plot_images(img, output_to_target(output, width, height), paths, f, names)  # predictions

    # Compute statistics
    # 计算上述测试过程中的各种性能指标
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, fname=save_dir / 'precision-recall_curve.png')
        p, r, ap50, ap75, ap = p[:, 0], r[:, 0], ap[:, 4], ap[:, -1], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # W&B logging
    # 绘图 wandb日志文件中对测试图片进行可视化
    if plots and wandb:
        wandb.log({"Images": wandb_images})
        wandb.log({"Validation": [wandb.Image(str(x), caption=x.name) for x in sorted(save_dir.glob('test*.jpg'))]})

    # Print results
    # 按照以下格式来打印测试过程的指标
    pf = '%20s' + '%12.3g' * 7  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map75, map))

    # Print results per class
    # 打印每一个类别对应的性能指标
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap75[i], ap[i]))

    # Print speeds
    # 打印 推断/NMS过程/总过程 的在每一个batch上面的时间消耗
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    # 保存之前json格式的预测结果，并利用coco的api进行评估
    # 因为COCO测试集的标签是给出的 因此此评估过程结合了测试集标签
    # 在更多的目标检测场合下 为保证公正测试集标签不会给出 因此以下过程应进行修改
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = glob.glob('../coco/annotations/instances_val*.json')[0]  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            # 以下过程为利用官方coco工具进行结果的评测
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    # Return results
    # 返回结果
    if not training:  # 如果不是训练过程则将结果保存到对应的路径
        print('Results saved to %s' % save_dir)
    model.float()  # for training 将模型转换为适用于训练的状态
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    # 返回对应的测试结果
    return (mp, mr, map50, map75, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    """
    opt参数详解：
        weights: 模型的权重文件地址
        data: 数据集配置文件地址，
        batch-size: 前向传播的批次大小，默认32
        img-size: 输入网络的图片分辨率，默认640
        conf-thres: object置信度阈值，默认0.25
        iou-thres: 进行NMS时IoU阈值，默认0.6
        task: 设置测试的类型，有val、test、study，默认val
        device: 测试的设备
        single-cls: 数据集是否只用一个类别，默认False
        augment: 测试时是否使用数据增强（TTA Test Time Augment），默认False
        verbose: 是否打印出每个类别的mAP，默认False
        下面两个参数是auto-labelling(有点像RNN中的teaching forcing)
        相关参数详见:https://github.com/ultralytics/yolov5/issues/1563 下面解释是作者原话
        save-txt: traditional auto-labelling
        save-conf: add confidences to any of the above commands
        save-json: 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签） 默认False
        project: 测试保存的源文件 默认runs/test
        name:  测试保存的文件地址 默认exp  保存在runs/test/exp下
        exist-ok: 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
        cfg: 网络配置文件
        names: 数据集里类别的名称
    """
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')  # |或 左右两个变量有一个为True 左边变量就为True
    opt.data = check_file(opt.data)  # check file
    print()
    print(opt)

    # 如果opt.task 为val 或者 test，就正常测试 验证集/测试集
    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt,
             save_conf=opt.save_conf,
             )

    # 如果opt.task = ['study']就评估yolov4-pacsp和yolov4-pacsp-x各个模型在各个尺度下的指标
    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolov4-pacsp.weights', 'yolov4-pacsp-x.weishts']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # utils.general.plot_study_txt(f, x)  # plot

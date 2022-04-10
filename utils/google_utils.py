# Google utils: https://cloud.google.com/storage/docs/reference/libraries

import os
import platform
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn


def gsutil_getsize(url=''):
    """
    计算某个url对应的文件大小，用于返回网站链接url对应的文件大小
    gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    """
    # 创建一个子进程在命令行执行，gsutil du url 命令(访问 Cloud Storage) 返回执行结果(文件)
    s = subprocess.check_output('gsutil du %s' % url, shell=True).decode('utf-8')
    # 返回文件的bytes大小
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes


def attempt_download(weights):
    """
    用在experimental中的attempt_load函数和train.py中下载预训练权重
    实现从github下载文件(预训练模型)
    目前现在下载不可用(链接失效)
    """
    # Attempt to download pretrained weights if not found locally
    # .strip()删除字符串前后空格 /n /t等  .replace将 ' 替换为空格  Path将str转换为Path对象
    weights = weights.strip().replace("'", '')
    file = Path(weights).name   # 权重名称

    msg = weights + ' missing, try downloading from https://github.com/WongKinYiu/ScaledYOLOv4/releases/'
    models = ['yolov4-csp.pt', 'yolov4-csp-x.pt']  # available models 目前支持的模型

    if file in models and not os.path.isfile(weights):
        try:  # GitHub
            url = 'https://github.com/WongKinYiu/ScaledYOLOv4/releases/download/v1.0/' + file
            print('Downloading %s to %s...' % (url, weights))
            torch.hub.download_url_to_file(url, weights)
            assert os.path.exists(weights) and os.path.getsize(weights) > 1E6  # check
        except Exception as e:  # GCP
            print('ERROR: Download failure.')
            print('')
            
            
def attempt_load(weights, map_location=None):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    加载模型权重文件并构建模型（可以构造普通模型或者集成模型）
    @param weights: 模型的权重文件地址
                    可以是[a]也可以是list格式[a, b]
    @param map_location: attempt_download函数参数  表示模型运行设备device
    """
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

    if len(model) == 1:  # 单个模型 正常返回
        return model[-1]  # return model
    else:
        # 多个模型 使用模型集成 并对模型先进行一些必要的设置
        print('Ensemble created with %s\n' % weights)
        # 给每个模型一个name属性
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


def gdrive_download(id='1n_oKgR81BJtqk75b00eAjdv03qVCQn2f', name='coco128.zip'):
    """
    实现从google drive上下载压缩文件并将其解压，再删除压缩文件
    @param id: url?后面的id参数的参数值
    @param name: 需要下载的压缩文件名
    """
    # Downloads a file from Google Drive. from utils.google_utils import *; gdrive_download()
    t = time.time()

    print('Downloading https://drive.google.com/uc?export=download&id=%s as %s... ' % (id, name), end='')
    # 如果文件存在则删除文件
    os.remove(name) if os.path.exists(name) else None  # remove existing
    # 如果cookie存在则删除cookie
    os.remove('cookie') if os.path.exists('cookie') else None

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    # 使用Terminal命令从google drive上下载文件
    os.system('curl -c ./cookie -s -L "drive.google.com/uc?export=download&id=%s" > %s ' % (id, out))
    if os.path.exists('cookie'):  # large file
        # 如果文件较大 就需要有令牌get_token(存在cookie才有令牌)的指令s才能下载
        # get_token()函数在下面定义了，用于获取当前cookie的令牌token
        s = 'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm=%s&id=%s" -o %s' % (get_token(), id, name)
    else:  # small file
        # 小文件就不需要带令牌的指令s，直接下载就行
        s = 'curl -s -L -o %s "drive.google.com/uc?export=download&id=%s"' % (name, id)
    # 执行下载指令s，并获得返回。如果Terminal命令执行成功，则os.system()命令会返回0
    r = os.system(s)  # execute, capture return
    # 再次移除已经存在的cookie
    os.remove('cookie') if os.path.exists('cookie') else None

    # Error check
    # 如果r != 0 则下载错误
    if r != 0:
        # 下载错误 移除下载的文件(可能不完全或者下载失败)
        os.remove(name) if os.path.exists(name) else None  # remove partial
        print('Download error ')  # raise Exception('Download error')
        return r

    # Unzip if archive
    # 如果是压缩文件，就解压
    if name.endswith('.zip'):
        print('unzipping... ', end='')
        os.system('unzip -q %s' % name)  # unzip
        os.remove(name)  # remove zip to free space

    print('Done (%.1fs)' % (time.time() - t))  # 打印下载解压过程中所需要的时间
    return r


def get_token(cookie="./cookie"):
    # 实现从cookie中获取令牌token
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""


class Ensemble(nn.ModuleList):
    """
    模型集成  Ensemble of models
    动机: 减少模型的泛化误差
    来源: https://www.sciencedirect.com/topics/computer-science/ensemble-modeling
    """
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        # 集成模型为多个模型时, 在每一层forward运算时, 都要运行多个模型在该层的结果append进y中
        for module in self:
            y.append(module(x, augment)[0])  # 添加module
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.cat(y, 1)  # nms ensemble
        y = torch.stack(y).mean(0)  # mean ensemble 求两个模型结果的均值
        return y, None  # inference, train output
    
    
# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     # Uploads a file to a bucket
#     # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
#
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#     print('File {} uploaded to {}.'.format(
#         source_file_name,
#         destination_blob_name))
#
#
# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     # Uploads a blob from a bucket
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#
#     blob.download_to_filename(destination_file_name)
#
#     print('Blob {} downloaded to {}.'.format(
#         source_blob_name,
#         destination_file_name))

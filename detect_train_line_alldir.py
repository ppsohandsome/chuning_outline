import matplotlib

import matplotlib.pyplot as plt
def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


import torch.jit
script_method1 = torch.jit.script_method
script1 = torch.jit.script
torch.jit.script_method = script_method
torch.jit.script = script
import torch
import cv2
import yaml
from torchvision import transforms
import numpy as np

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image
import os
import time
from torch.nn.functional import interpolate
start_time = time.time()  # 记录程序开始时间

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open('data/hyp.scratch.mask.yaml') as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)
weigths = torch.load('data/yolov7-mask.pt')
model = weigths['model']
model = model.half().to(device)
_ = model.eval()

input_folder = "zinput"
output_folder = "zoutput"

# 获取输入文件夹中所有图片文件名
image_files = os.listdir(input_folder)
# 定义dix从0开始
idx = 0

for image_file in image_files:
    # 构造输入文件路径
    input_path = os.path.join(input_folder, image_file)

    # 读入图片并进行尺寸变换
    image = cv2.imread(input_path)  # 原始大小的图片
    image = letterbox(image, 640, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    # 将图片放到GPU上
    image = image.to(device)
    image = image.half()

    # 使用模型进行目标检测和分割
    output = model(image)
    inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output['attn'], \
                                                            output[
                                                                'mask_iou'], output['bases'], output['sem']
    bases = torch.cat([bases, sem_output], dim=1)
    nb, _, height, width = image.shape
    names = model.names
    pooler_scale = model.pooler_scale
    pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1,
                       pooler_type='ROIAlignV2', canonical_level=2)
    output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases,
                                                                                                 pooler, hyp,
                                                                                                 conf_thres=0.25,
                                                                                                 iou_thres=0.65,
                                                                                                 merge=False,
                                                                                                 mask_iou=None)
    pred, pred_masks = output[0], output_mask[0]
    base = bases[0]
    bboxes = Boxes(pred[:, :4])
    original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
    pred_masks = retry_if_cuda_oom(paste_masks_in_image)(original_pred_masks, bboxes, (height, width), threshold=0.5)
    pred_masks_np = pred_masks.detach().cpu().numpy()
    pred_cls = pred[:, 5].detach().cpu().numpy()
    pred_conf = pred[:, 4].detach().cpu().numpy()
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int_)
    pnimg = nimg.copy()
    # cv2.imshow()
    # 只看cls id为6的目标，并且置信度 > 0.25
    target_cls_id = 6

    for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
        if cls != target_cls_id:
            continue

        if conf < 0.8:
            continue

        color1 = (0, 255, 0)
        pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color1, dtype=np.uint8) * 0.5

        # # 将目标掩膜保存为PNG文件
        mask_name = f"{target_cls_id}_{idx}_mask.png"
        cv2.imwrite(os.path.join(output_folder, mask_name), one_mask.astype(np.uint8) * 255)
        idx += 1
    # 得到输出路径并保存结果
    output_file = os.path.splitext(image_file)[0] + "_mask.jpg"
    output_path = os.path.join(output_folder, output_file)
    cv2.imwrite(output_path, pnimg)

    ############################################
    # # 定义黄色色彩空间范围
    # yellow_lower = (0, 80, 80)
    # yellow_upper = (10, 255, 255)
    # # 读取并原始图像
    # img = pnimg.copy()
    # # 将图像转换为HSV色彩空间
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # # 根据黄色色彩空间范围，创建掩码
    # mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    # # 对掩码进行形态学处理
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # # 查找并绘制轮廓
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # # 保存处理后的图像
    # mask_name = f"{image_file}_line.png"
    # cv2.imwrite(os.path.join(output_folder, mask_name),img)

# 记录程序结束时间
end_time = time.time()
# 计算程序总运行时间（单位：秒）
total_time = end_time - start_time
print("Total time:", total_time, "s")

import cv2
import numpy as np
from PIL import Image
import base64
import io

def tobase64(mask_contours):
    # 将图像转换为 Base64 字符串
    retval, buffer = cv2.imencode('.jpg', mask_contours)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    # print(base64_str)
    return base64_str

def toimg(base64_str):
    # 将 Base64 字符串转换为图像并保存
    img_data = base64.b64decode(base64_str)
    with open('zoutput/result.jpg', 'wb') as f:
        f.write(img_data)





from datetime import datetime

import cv2
import config

image_dict = {
    'Stream 0': (config.D25_line_y1, config.D25_line_y2, "D25", "10"),
    'Stream 1': (config.D26_line_y1, config.D26_line_y2, "D26", "14"),
    'Stream 2': (config.D27_line_y1, config.D27_line_y2, "D27", "32"),
    'Stream 3': (config.D28_line_y1, config.D28_line_y2, "D28", "50"),
    'Stream 4': (config.D29_line_y1, config.D29_line_y2, "D29", "64"),
    'Stream 5': (config.D30_line_y1, config.D30_line_y2, "D30", "76"),
}


def fence(mask, image_name):
    # 将图像进行二值化处理，黑色为0，白色为255
    _, img_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # 保存二值化结果
    cv2.imwrite('binary_image.jpg', img_binary)

    # 自下而上找到第一个白色像素所在的行
    y_coord = None
    for y in range(mask.shape[0]-1, -1, -1):
        if 255 in img_binary[y, :]:
            # 将找到的位置进行标记
            img_binary[y, :] = 128
            y_coord = y
            break

    # 保存标记后的图像
    cv2.imwrite('marked_image.jpg', img_binary)
    return image_dict[image_name][3], y_coord*3-35, image_dict[image_name][2], str(datetime.now())

if __name__ == '__main__':
    # 读取图像
    img = cv2.imread('6_mask.png', cv2.IMREAD_GRAYSCALE)
    fence(img,"Stream 5")


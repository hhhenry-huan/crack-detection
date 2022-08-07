import time
import copy
import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging, check_requirements
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_synchronized
from imgprocess2 import imageprocessing, Max_width, show


weights='runs/train/exp18/weights/best.pt'    # 训练好的模型路径   （必改）
device=''  # 设备
half=False  # 使用FP16半精度推理
imgsz=512   # 训练模型设置的尺寸 （必改）

# -----初始化-----
set_logging()
# 设置设备
device = select_device(device)
# CUDA仅支持半精度
half &= device.type != 'cpu'

# -----加载模型-----
# 加载FP32模型
model = attempt_load(weights, map_location=device)
# 模型步幅
stride = int(model.stride.max())
# 检查图像大小
imgsz = check_img_size(imgsz, s=stride)
# 获取类名
names = model.module.names if hasattr(model, 'module') else model.names
# toFP16
if half:
    model.half()

@torch.no_grad()
def detect(
        # --------------------这里更改配置--------------------
        # ---------------------------------------------------
        img0,  #待检测图片

        conf_thres=0.25,  # 置信度
        iou_thres=0.45,  # NMS IOU 阈值
        max_det=1000,  # 最大侦测的目标数

        crop=True,  # 显示预测框
        classes=None,  # 种类
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # 是否扩充推理

        hide_labels=False,  # 是否隐藏标签
        hide_conf=False,  # 是否隐藏置信度
        line_thickness=3  # 预测框的线宽
):
    # #--------------------这里更改配置--------------------
    # -----------------------------------------------------

    T1 = time.time()
        # ------运行推理------
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 跑一次

    # 设置labels--记录标签/概率/位置
    labels = []
    # 计时
    t0 = time.time()
    # 填充调整大小
    img = letterbox(img0, imgsz, stride=stride)[0]
    # 转换
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    # uint8 to fp16/32
    img = img.half() if half else img.float()
    # 0 - 255 to 0.0 - 1.0
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 推断
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]

    # 添加 NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    t2 = time_synchronized()

    # 目标进程
    box_info=[]
    for i, det in enumerate(pred):  # 每幅图像的检测率
        s, im0 = '', img0.copy()
        # 输出字符串
        s += '%gx%g ' % img.shape[2:]
        # 归一化增益
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        if len(det):
            # 将框从img_大小重新缩放为im0大小
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # 输出结果
            for c in det[:, -1].unique():
                # 每类检测数
                n = (det[:, -1] == c).sum()
                # 添加到字符串
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                # 结果输出
            for *xyxy, conf, cls in reversed(det):
                box_info_tmp=[]
                # 归一化xywh
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                # 标签格式
                line = (cls, *xywh, conf)
                # 整数类
                c = int(cls)
                # 建立标签
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                # 绘画预测框
                if crop:
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                # 记录标签/概率/位置
                labels.append([names[c], conf, xyxy])
                #print(xyxy)
                for i in range(4):
                    box_info_tmp.append(int(xyxy[i].cpu()))
                box_info.append(box_info_tmp)     #如果有多个检测框，按每四个一组放在列表中

    # 显示图片
    if box_info:
        for box in box_info:
            w1 = box[0]
            h1 = box[1]
            w2 = box[2]
            h2 = box[3]
            H = h2 - h1
            W = w2 - w1
            image = img0[h1:h2, w1:w2]

            if H<W:
                image2 = copy.deepcopy(image)
                image2 = np.rot90(image2)                          #np.flipud(np.rot90(image2))
                img_tmp = cv2.resize(image2,(40,120))
                image_binar = imageprocessing(img_tmp)
                image_binar = cv2.resize(image_binar,(H,W))
                # cv2.imshow("ss",image_binar)
                # cv2.waitKey(2000)
                po_row, po_col, width_max  = Max_width(image2,image_binar)
                # print(po_row,po_col,width_max)
                image_binar = np.rot90(image_binar,k=-1)

                show(image,image_binar)
                cv2.rectangle(image,(0,0),(W-1,H-1),(0,0,255))
                cv2.rectangle(image, (W-po_row- 6, po_col- 6), (W-po_row + 6, po_col + 6), (0, 255, 255))
                piInString = str(width_max)  # str(crack_width)
                cv2.putText(image, piInString, (W-po_row + 15, po_col), cv2.FONT_HERSHEY_PLAIN, 1,
                           (0, 255, 255), 1, cv2.LINE_AA)
            else:
                img_tmp = cv2.resize(image, (40, 120))
                image_binar = imageprocessing(img_tmp)
                image_binar = cv2.resize(image_binar, (W, H))
                po_row, po_col, width_max  = Max_width(image,image_binar)
                # cv2.imshow("image",image)
                # cv2.imshow("image_binar",image_binar)
                # cv2.waitKey(2000)
                show(image, image_binar)
                cv2.rectangle(image, (0, 0), (W - 1, H - 1), (0, 0, 255))

                cv2.rectangle(image, (po_col - 6, po_row - 6), (po_col + 6, po_row + 6), (0, 255, 255))
                piInString = str(width_max)  # str(crack_width)
                cv2.putText(image, piInString, (po_col + 15, po_row), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 255), 1, cv2.LINE_AA)
            # cv2.imshow(str(H),image_binar)
            # cv2.waitKey(2000)

        # h1, w1, h2, w2 =int(box_info[0][1]) ,int(box_info[0][0]),int(box_info[0][3]),int(box_info[0][2])
        # image = im0[h1:h2, w1:w2]

        # ##图像处理###
        # img_final, position_row, position_col, crack_width = imgproc.imageprocessing(image)
        #
        # # 加标注
        # if h1 + position_row > 20 and w1 + position_col + 100 < image.shape[1]:  # 右上标注
        #     cv2.putText(image, crack_width, (w1 + position_col + 10, h1 + position_row - 5), cv2.FONT_HERSHEY_PLAIN, 1,
        #                 (0, 255, 255), 1, cv2.LINE_AA)
        # elif h1 + position_row < 20 and w1 + position_col + 100 > image.shape[1]:  # 左下标注
        #     cv2.putText(image, crack_width, (w1 + position_col - 70, h1 + position_row + 5), cv2.FONT_HERSHEY_PLAIN, 1,
        #                 (0, 255, 255), 1, cv2.LINE_AA)
        # elif h1 + position_row < 20:  # 右下标注
        #     cv2.putText(image, crack_width, (w1 + position_col + 10, h1 + position_row + 5), cv2.FONT_HERSHEY_PLAIN, 1,
        #                 (0, 255, 255), 1, cv2.LINE_AA)
        # else:  # 左上标注
        #     cv2.putText(image, crack_width, (w1 + position_col - 70, h1 + position_row + 5), cv2.FONT_HERSHEY_PLAIN, 1,
        #                 (0, 255, 255), 1, cv2.LINE_AA)

        # crack_info = "裂纹信息：" + str(position_row) + "," + str(position_col) + "   宽度：" +str(crack_width)
        crack_info = "有裂纹"
    else:
        # image = cv2.resize(im0, (640, 480), interpolation=cv2.INTER_LINEAR)
        crack_info = "无裂纹"

    #cv2.imshow("666", im0)

    # 输出计算时间
    # print("裂纹区域大小：",image.shape)
    # print(f'检测消耗时间: ({time.time() - t0:.3f}s)')
    T2 = time.time()
    print("用时：",T2-T1)
    return img0,crack_info

if __name__ == "__main__":
    img0 = cv2.imread('C://Users//DELL//Desktop//torch_learning//yolov5//yolov5-master//my_dates//images//test//1316.jpg')
    image, crack_info = detect(img0)
    print("---------")
    # print(box_info)
    # cv2.imwrite("250.jpg",image)
    cv2.imshow("s", image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
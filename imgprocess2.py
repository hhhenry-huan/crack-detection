import sys
import time
import cv2 as cv
from copy import deepcopy, copy
import numpy as np
np.set_printoptions(threshold=np.inf)

def doublesobelthreshold(img,low_thresholdcoefficient,high_thresholdcoefficient):
    listofcrack = []
    height, width = img.shape
    num = np.array(img, dtype=np.int32)
    max1 = num.max(axis=1)
    min1 = num.min(axis=1)
    high_threshold=np.zeros((height),dtype=np.int32)

    for i in range(height):
        high_threshold[i] = high_thresholdcoefficient * (max1[i] - min1[i])
        low_threshold=low_thresholdcoefficient*(max1[i]-min1[i])
        for j in range(width):
            if (num[i][j] > max1[i] - low_threshold)or(num[i][j]<min1[i]+low_threshold):
                listofcrack.append([i, j])

    listofcrack1 = deepcopy(listofcrack)
    while (len(listofcrack) != 0):
        point = listofcrack[-1]
        row, col = point
        listofcrack.pop()
        # 上方 row-1 col不变
        if (row - 1 >= 0) and ([row - 1, col] not in listofcrack1):
            if (img[row - 1, col] > max1[row - 1] - high_threshold[row - 1])or(img[row - 1, col] < min1[row - 1] + high_threshold[row - 1]):
                listofcrack.append([row - 1, col])
                listofcrack1.append([row - 1, col])
        # 右上方  row-1 col+1
        if (row - 1 >= 0) and (col + 1 < width) and ([row - 1, col + 1] not in listofcrack1):
            if (img[row - 1, col + 1] > max1[row - 1] - high_threshold[row - 1])or(img[row - 1, col+1] < min1[row - 1] + high_threshold[row - 1]):
                listofcrack.append([row - 1, col + 1])
                listofcrack1.append([row - 1, col + 1])

        # 右下 row+1 col+1
        if (row + 1 < height) and (col + 1 < width) and ([row + 1, col + 1] not in listofcrack1):
            if (img[row + 1, col + 1] > max1[row + 1] -high_threshold[row+1])or(img[row + 1, col + 1] < min1[row + 1] + high_threshold[row+1]):
                listofcrack.append([row + 1, col + 1])
                listofcrack1.append([row + 1, col + 1])
        # 下方 row+1 col不变
        if (row + 1 < height) and ([row + 1, col] not in listofcrack1):
            if (img[row + 1, col] > max1[row + 1] -high_threshold[row+1])or(img[row + 1, col] < min1[row + 1] + high_threshold[row+1]):
                listofcrack.append([row + 1, col])
                listofcrack1.append([row + 1, col])
        # 左下方 row+1 col-1
        if (row + 1 < height) and (col - 1 > 0) and ([row + 1, col - 1] not in listofcrack1):
            if (img[row + 1, col - 1] > max1[row + 1] -high_threshold[row+1])or(img[row + 1, col-1] < min1[row + 1] + high_threshold[row+1]):
                listofcrack.append([row + 1, col - 1])
                listofcrack1.append([row + 1, col - 1])

        # 左上 row-1 col-1
        if (row - 1 >= 0) and (col - 1 >= 0) and ([row - 1, col - 1] not in listofcrack1):
            if (img[row - 1, col - 1] > max1[row - 1] - high_threshold[row-1])or(img[row - 1, col - 1] < min1[row - 1] + high_threshold[row-1]):
                listofcrack.append([row - 1, col - 1])
                listofcrack1.append([row - 1, col - 1])

    num=np.zeros((height,width),dtype=np.uint8)
    for i in range(len(listofcrack1)):
        num[listofcrack1[i][0], listofcrack1[i][1]] = 255
    return num


##基于连通域的过滤
def connections_filter(img1,n,img2,area_criterion,shape_criterion):#输入为 原图像，stats原图上连通域的信息，连通域检测之后的标签图
    h,w=img1.shape
    res=np.zeros((h,w),img1.dtype)
    for i in range(h):
        for j in range(w):
            item=img2[i][j]
            if item==0:
                pass
            elif (n[item,4]<area_criterion)or(n[item,3]/n[item,2]<shape_criterion):
                #根据面积和长宽比过滤
                res[i,j]=0
            else:res[i,j]=255
    return res

def binar_image(img):                ##改进
    h, w = img.shape
    num = np.array(img, dtype=np.uint8)
    min1 = num.min(axis=1)
    for i in range(h):
        for j in range(w):
            if num[i][j] < min1[i] +5:
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img

##输入细化骨架图和原图，获取骨架上的所有点，形成一个列表，获取每行的白点数
def geteverypiont(rowerzhiimg,img_thin,list,lenofrows):
    highth,width=rowerzhiimg.shape
    for i in range(highth):
        sum=0
        for j in range(width):
            if (img_thin[i][j]==255):
                list.append([i,j])
            if (rowerzhiimg[i][j]==255):
                sum=sum+1

        lenofrows.append(sum)
    for i in list:
        i.append(lenofrows[i[0]])



def show(row_img,erzhitu):
    highth=row_img.shape[0]
    width=row_img.shape[1]
    for i in range(highth):
        for j in range(width):
            if(erzhitu[i,j]>=120):
                row_img[i][j][0]=0
                row_img[i][j][1]=255
                row_img[i][j][2]=255




def imageprocessing(img):
    ini_img = img
    height=img.shape[0]
    width=img.shape[1]
    #灰度化
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #双边滤波
    img_bilateralF_gray = cv.bilateralFilter(img_gray,5,15,15)
    #cv.imshow('shaungb',img_bilateralF_row)
    img_bilateralFilter= binar_image(img_bilateralF_gray)
    #cv.imshow('2', img_bilateralFilter)

    # CLAHE图像增强
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    #高斯滤波
    img_Gaussion = cv.GaussianBlur(img_clahe, (3, 3), 0.2)

    #sobel算子边缘检测
    x = cv.Sobel(img_Gaussion, cv.CV_16S, 1, 0)
    y = cv.Sobel(img_Gaussion, cv.CV_16S, 0, 1)
    result = cv.addWeighted(x, 1, y, 0, 0)
    sobel_threshold=doublesobelthreshold(result,0.05,0.1)
    #cv.imshow('3', sobel_threshold)

    #形态学运算
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    result2 = cv.dilate(sobel_threshold, k)
    result3 = cv.morphologyEx(result2, cv.MORPH_CLOSE, k,3)
    img_morph= cv.erode(result3, k)
    #cv.imshow('4', img_morph)

    #连通域检测
    count, dst, stats, centroids = cv.connectedComponentsWithStats(img_morph,ltype=cv.CV_16U)
    img_connections = connections_filter(img_morph, stats, dst, 50,0.8)
    #cv.imshow('5', img_connections)

    #取交集
    img_final= cv.min(img_connections,img_bilateralFilter)
    #cv.imshow('6',img_final)
    return img_final

def Max_width(ini_img,img_final):
    # 细化，提取骨骼
    img_thin = cv.ximgproc.thinning(img_final, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)

    # 获取骨架上的所有点和每行的白点个数
    list = []
    lenofrows = []

    geteverypiont(img_final, img_thin, list, lenofrows)
    # 此时list记录了骨架上每个点的行，列，所在行的所有白点。

    # 寻找最大值                                                                  #改进
    # list_array = np.array(list)
    list_realwidth = np.zeros(len(list), dtype=float)
    for i in range(len(list)):
        list_realwidth[i] = list[i][2]

    # 从最上面开始冒泡得到最大值
    index_max = 0
    width_max = 0
    for i in range(len(list)):
        if list_realwidth[i] >= width_max:
            index_max = i
            width_max = list_realwidth[i]

    position_row = list[index_max][0]
    position_col = list[index_max][1]

    # print("最宽部分的行数和列数分别为", position_row, position_col)
    # print('最大宽度为：', width_max)

    # cv.rectangle(ini_img, (position_col - 6, position_row - 6), (position_col + 6, position_row + 6), (0, 255, 255))

    # realmaxwidth = width_max  # 改进
    # ratio = 0.08275
    # realmaxwidth = (realmaxwidth) * ratio
    # crack_width = '%.3f' % realmaxwidth

    #ini_img = show(ini_img, img_final)

    # piInString = str(width_max)  # str(crack_width)
    # cv.putText(ini_img, piInString, (position_col + 15, position_row), cv.FONT_HERSHEY_PLAIN, 1,
    #            (0, 255, 255), 1, cv.LINE_AA)


    return position_row, position_col, width_max

if __name__ == "__main__":
    img = cv.imread("C://Users//87441//Desktop//Mask_RCNN-master//result//4.jpg")
    # img_binar = imageprocessing(img)
    # image = Max_width(img,img_binar)
    # image = np.rot90(image)
    # img_binar = np.rot90(img_binar)
    # image = show(image,img_binar)
    # cv.imshow("2",image)
    image = img[0:20, 10:30]
    cv.rectangle(image, (5, 10), (8, 15), (0, 255, 255))
    cv.imshow("2", img)
    cv.waitKey(2000)
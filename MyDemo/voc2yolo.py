import xml.etree.ElementTree as ET

import os
from os import listdir, getcwd

classes = ["smoke", "fire"] # 换上你标签

def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(image_name,labelPath):
   # in_file = open(os.path.join(labelPath,image_name[:-3] + 'xml'))  # xml文件路径

    out_file = open(os.path.join(labelPath+'TXT',image_name[:-3] + 'txt'), 'w')  # 转换后的txt文件存放路径

    in_file = open(os.path.join(labelPath,image_name[:-3] + 'xml'))  # xml文件路径
    xml_text = in_file.read()
    root = ET.fromstring(xml_text)
    in_file.close()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
   # print(root.iter('object'))
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            print(cls)
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        #print(bb)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

if __name__ == '__main__':
    imgNmae=input('输入图像文件夹的绝对地址：')
    labelPath=input('输入xml标注的文件夹的绝对地址：')
    #imgNmae='E:\数据集\新建文件夹//bed//bed\data\img'
    #labelPath='E:\数据集\新建文件夹//bed/bed\data\labels'
    if not os.path.isdir(labelPath+'TXT'):
        os.mkdir(labelPath+'TXT')
    for image_path in os.listdir(imgNmae):  # 每一张图片都对应一个xml文件这里写xml对应的图片的路径
        #image_name = image_path.split('\\')[-1]
        print(image_path)
        convert_annotation(image_path,labelPath)

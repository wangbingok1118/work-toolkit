# -*- coding:utf-8 -*-
"""
这个脚本用于评估 检测；
--gtXmlBasePath 存放标注的xml文件的路径
--detResultFile 模型在测试数据集上的结果，暴恐检测回归测试 的格式
''' split by \t
    knives.jpg  [{"class": "knives", "index": 3, "pts": [[49, 69], [75, 69], [75, 108], [49, 108]], "score":0.9993000030517578}]
    ISIS.jpg    [{"class": "isis flag", "index": 1, "pts": [[157, 107], [515, 107], [515, 328], [157, 328]], "score":0.9998000264167786}]
    tibetan.jpg [{"class": "tibetan flag", "index": 5, "pts": [[13, 12], [215, 12], [215, 137], [13, 137]], "score":0.9998999834060669}]
    guns.jpg    [{"class": "guns", "index": 4, "pts": [[198, 325], [894, 325], [894, 571], [198, 571]], "score":0.9986000061035156}]
    islamic.jpg [{"class": "islamic flag", "index": 2, "pts": [[136, 147], [465, 147], [465, 425], [136, 425]], "score":0.9975000023841858}]
    weiboimg_wordlib_0-20170927_45944.jpg	[{"index": 2, "score": 0.997883141040802, "pts": [[9, 14], [568, 14], [568, 149], [9, 149]], "class": "guns"}, {"index": 2, "score": 0.9955806136131287, "pts": [[22, 112], [556, 112], [556, 302], [22, 302]], "class": "guns"}]
'''
--detClassLabelFile 需要评估的检测模型 类别文件
'''
    eg : className modelPreIndex flag 
        flag :1 then the class compute or flag : 0 then the class not compute
        {"class":"guns","index":1,"flag":1}
        {"class": "knives", "index": 2,"flag":1}
        {"class": "tibetan flag", "index": 3,"flag":1}
        {"class": "islamic flag", "index": 4,"flag":1}
        {"class": "isis flag", "index": 5,"flag":1}
        {"class": "not terror", "index": 6,"flag":0}
'''
"""

import os
import sys
import json
import argparse
import pprint
import xml.etree as etree
from compute_model_mAP import compute_model_mAP
from compute_each_class_ap import compute_each_class_ap
from compute_each_class_accuracy import compute_each_class_accuracy


def score_model_fun(detResultFile=None,detClassLabelFile=None,gtXmlBasePath=None):
    className_detfile_dict = convert_detResultFile_to_pascalEvalFile(
        detResultFile=detResultFile, detClassLabelFile=detClassLabelFile)
    # compute mAP IOU = 0.5
    print('*'*20+" mAP "+"*"*20)
    mAP = compute_model_mAP(className_detfile_dict=className_detfile_dict, gtXmlBasePath=gtXmlBasePath)

    # compute each class AP ,ingore bbox
    print('*'*20+" ap "+"*"*20)
    class_ap_dict = compute_each_class_ap(className_detfile_dict=className_detfile_dict, gtXmlBasePath=gtXmlBasePath)

    # compute each class Accuracy ,ingore bbox
    print('*'*20+" precision "+"*"*20)
    class_accuracy_dict = compute_each_class_accuracy(
        className_detfile_dict=className_detfile_dict, gtXmlBasePath=gtXmlBasePath)
    pass


# 这个函数用于 解析回归测试的结果文件，提取关于这个类的bbox
def get_class_preBbox_file(detResultFile=None, classNameStr=None):
    detResultFile_basePath = os.path.join(
        detResultFile[:detResultFile.rfind('.')], 'pascal_eval')
    if not os.path.exists(detResultFile_basePath):
        os.makedirs(detResultFile_basePath)
    detResultFile_basePath_classFile = os.path.join(
        detResultFile_basePath, classNameStr.replace(" ", '')+'_mAP.txt')
    detResultFile_basePath_classFile_imagelistFile = os.path.join(
        detResultFile_basePath, 'allTest_imagelist.txt')
    class_image_list = []
    with open(detResultFile_basePath_classFile, 'w') as w_f, open(detResultFile, 'r') as r_f:
        for line in r_f.readlines():
            if len(line) <= 0:
                continue
            image_name, bbox_lists = line.strip().split('\t')
            class_image_list.append(image_name[:image_name.rfind('.')])
            bbox_lists = eval(bbox_lists)
            for bbox in bbox_lists:
                line_list = []
                if (bbox.get('class') != None and bbox.get('class') == classNameStr):
                    line_list.append(image_name[:image_name.rfind('.')])
                    line_list.append(bbox.get('score'))
                    line_list.append(bbox.get('pts')[0][0])  # xmin
                    line_list.append(bbox.get('pts')[0][1])  # ymin
                    line_list.append(bbox.get('pts')[2][0])  # xmax
                    line_list.append(bbox.get('pts')[2][1])  # ymax
                    w_f.write(' '.join([str(i) for i in line_list]))
                    w_f.write('\n')
    if not os.path.exists(detResultFile_basePath_classFile_imagelistFile):
        with open(detResultFile_basePath_classFile_imagelistFile, 'w') as f:
            f.write('\n'.join(class_image_list))
            f.write('\n')
    return [detResultFile_basePath_classFile, detResultFile_basePath_classFile_imagelistFile]


def convert_detResultFile_to_pascalEvalFile(detResultFile=None, detClassLabelFile=None):
    """
        detResultFile is absolute path ,the detect result file 
        detClassLabelFile is score class label file
        这个函数的作用是，根据要计算的类的名称，将检测回归测试的结果文件，分开形成 单独类别的预测结果文件
        return dict: 
                    key is  className ,
                    value is list ; absolute path of  the pascal eval need class file , test imagelist file
    """
    class_label_dict = {}
    # key is class label name
    # value is the pascal evaluate file absolute path
    with open(detClassLabelFile, 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            if line_dict['flag'] == 0:  # the class not score
                continue
            else:
                resFile = get_class_preBbox_file(
                    detResultFile=detResultFile, classNameStr=line_dict['class'])
                class_label_dict[line_dict['class']] = resFile
                # print(resFile)
    return class_label_dict



def parse_args():
    parser = argparse.ArgumentParser(description='score detecr modle',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gtXmlBasePath',help='the path of groundtruth xml files',default=None,type=str)
    # parser.add_argument('--testImageNameFile',help='the absolute path of image name file',default=None,type=str)
    parser.add_argument('--detResultFile',help='the detect result file',default=None,type=str)
    parser.add_argument('--detClassLabelFile',help='detect class label', default=None, type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # print(__doc__)
    print(args)
    print('*'*20+'begin score model'+'*'*20)
    score_model_fun(detResultFile=args.detResultFile,
                    detClassLabelFile=args.detClassLabelFile, 
                    gtXmlBasePath=args.gtXmlBasePath)


if __name__ == '__main__':
    main()

"""
python det-model-score.py \
--gtXmlBasePath /workspace/data/BK/terror-dataSet-Dir/TERROR-DETECT-V0.9/Annotations \
--detResultFile /workspace/data/terror-det-Dir/rfcn-Dirs/rfcn-res152-v0.9-t3/TERROR-DETECT-V0.9-test-rf-output.txt \
--detClassLabelFile /workspace/data/BK/terror-dataSet-Dir/terror-6-class.txt 
"""

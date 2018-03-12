# -*- coding:utf-8 -*-
"""
这个脚本用于计算
每类 accuracy （无视bbox，只考虑类别命中）
"""
import os
import sys
import numpy as np


def parse_voc_rec_noBbox(filename):
    """
    parse pascal voc record into a dictionary
    :param filename: xml file path
    :return: list of dict
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        obj_dict['name'] = obj.find('name').text
        obj_dict['difficult'] = int(obj.find('difficult').text)
        objects.append(obj_dict)
    return objects


def class_bbox_2_image(detfile=None):
    basePath = detfile[:detfile.rfind('/')]
    fileName = detfile.split('/')[-1].replace('_mAP.txt', '_accuracy.txt')
    class_image_ap_file = os.path.join(basePath, fileName)
    image_score_dict = dict()
    # image_id(str):score(float)
    with open(detfile, 'r') as f:
        for line in f.readlines():
            line_list = line.split(' ')
            if line_list[0] in image_score_dict:
                if image_score_dict.get(line_list[0]) < float(line_list[1]):
                    image_score_dict[line_list[0]] = float(line_list[1])
            else:
                image_score_dict[line_list[0]] = float(line_list[1])
    with open(class_image_ap_file, 'w') as f:
        for key in image_score_dict.keys():
            line = "%s %f\n" % (key, image_score_dict.get(key))
            # line = key + " " + image_score_dict.get(key)+'\n'
            f.write(line)
    return class_image_ap_file


def voc_eval_noBbox_accuracy(detfile=None, annopath=None, imageset_file=None, classname=None):
    """
    pascal voc evaluation
    :param detresultFile: detection results file （mAP)
    :param annopath: annotations file base path
    :param imageset_file: text file containing list of images
    :param classname: category name
    :return: rec, prec, ap
    """
    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    image_filenames = [x.strip() for x in lines]

    # load annotations
    recs = {}
    for ind, image_filename in enumerate(image_filenames):
        xmlFile = os.path.join(annopath, image_filename+'.xml')
        recs[image_filename] = parse_voc_rec_noBbox(xmlFile)

    # extract objects in :param classname:
    class_recs = {}
    npods = 0
    for image_filename in image_filenames:
        objects = [obj for obj in recs[image_filename]
                   if obj['name'] == classname]
        if len(objects) > 0:
            npods += 1
        difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
        class_recs[image_filename] = {'difficult': difficult}

    # read detections
    # 这里会对 原来的 计算 mAP 格式的bbox 文件格式  转换为 image 格式的文件
    detfile = class_bbox_2_image(detfile=detfile)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]

    nd = len(image_ids)
    acc = 0
    err = 0
    for d in range(nd):
        r = class_recs[image_ids[d]]
        image_class_anno_list = r['difficult']
        if len(image_class_anno_list) > 0:
            if False in image_class_anno_list:
                acc += 1.
            else:
                err += 1.
        else:
            err += 1.
    assert acc+err == nd,"acc : %d , err: %d , nd : %d"%(acc,err,nd)

    accuracy = acc * 1.0 / nd
    recall = acc * 1.0 /npods

    return acc, err, accuracy, recall


def compute_each_class_accuracy(className_detfile_dict=None, gtXmlBasePath=None):
    class_accuracy_dict = {}
    # print("begin process model class accuracy")
    for key in className_detfile_dict.keys():
        acc, err, accuracy,recall=voc_eval_noBbox_accuracy(detfile = className_detfile_dict.get(key)[
                                        0], annopath=gtXmlBasePath, imageset_file=className_detfile_dict.get(key)[1], classname=key)
        class_accuracy_dict[key] = [acc, err, accuracy, recall]
    for key in sorted(class_accuracy_dict.keys()):
        print("%s  precision: %f " % (
            key, class_accuracy_dict.get(key)[2]))
    print('*'*20+" recall "+"*"*20)
    for key in class_accuracy_dict.keys():
        print("%s  recall: %f " % (
            key,class_accuracy_dict.get(key)[3]))
    return class_accuracy_dict


def main():
    pass


if __name__ == '__main__':
    main()


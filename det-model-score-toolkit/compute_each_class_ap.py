# -*- coding:utf-8 -*-
"""
这个脚本用于计算
每类标注数据AP（无视bbox，只考虑类别命中）
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


def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

#  将 pascal voc eval 格式的文件 转换为  以  image 为基本单位的预测结果文件
def class_bbox_2_image(detfile=None):
    basePath = detfile[:detfile.rfind('/')]
    fileName = detfile.split('/')[-1].replace('_mAP.txt', '_ap.txt')
    class_image_ap_file = os.path.join(basePath, fileName)
    image_score_dict=dict()
    # image_id(str):score(float)
    with open(detfile,'r') as f:
        for line in f.readlines():
            line_list = line.split(' ')
            if line_list[0] in image_score_dict:
                if image_score_dict.get(line_list[0]) < float(line_list[1]):
                    image_score_dict[line_list[0]] = float(line_list[1])
            else:
                image_score_dict[line_list[0]] = float(line_list[1])
    with open(class_image_ap_file,'w') as f:
        for key in image_score_dict.keys():
            line = "%s %f\n" % (key, image_score_dict.get(key))
            # line = key + " " + image_score_dict.get(key)+'\n'
            f.write(line)
    return class_image_ap_file


def voc_eval_noBbox(detfile=None, annopath=None, imageset_file=None, classname=None, use_07_metric=False):
    """
    pascal voc evaluation
    :param detresultFile: detection results file (mAP)
    :param annopath: annotations file base path
    :param imageset_file: text file containing list of images
    :param classname: category name
    :param ovthresh: overlap threshold
    :param use_07_metric: whether to use voc07's 11 point ap computation
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
    npods=0
    for image_filename in image_filenames:
        objects = [obj for obj in recs[image_filename]
                   if obj['name'] == classname]
        if len(objects)>0:
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
    confidence = np.array([float(x[1]) for x in splitlines])

    sorted_inds = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    image_ids = [image_ids[x] for x in sorted_inds]

    # go down detections and mark true positives and false positives
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        r = class_recs[image_ids[d]]
        image_class_anno_list = r['difficult']
        if len(image_class_anno_list) > 0:
            if False in image_class_anno_list:
                tp[d] = 1.
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npods)
    # avoid division by zero in case first detection matches a difficult ground ruth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def compute_each_class_ap(className_detfile_dict=None, gtXmlBasePath=None):
    class_ap_dict = {}
    for key in className_detfile_dict.keys():
        # print("AP process: begin  %s ap ,the detect result file is : %s" %(key, className_detfile_dict.get(key)[0]))
        rec, prec, ap = voc_eval_noBbox(detfile=className_detfile_dict.get(key)[
                                        0], annopath=gtXmlBasePath, imageset_file=className_detfile_dict.get(key)[1], classname=key,use_07_metric=True)
        class_ap_dict[key] = [rec, prec, ap]
    for key in class_ap_dict.keys():
        print("%s ap: %f" % (key, class_ap_dict.get(key)[2]))
    return class_ap_dict



def main():
    pass
if __name__ == '__main__':
    main()

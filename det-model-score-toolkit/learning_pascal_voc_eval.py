# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Haozhi Qi, from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------
"""
given a pascal voc imdb, compute mAP
"""

import numpy as np
import os
import cPickle
from mask.mask_transform import mask_overlap


def parse_voc_rec(filename):
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
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [int(float(bbox.find('xmin').text)),
                            int(float(bbox.find('ymin').text)),
                            int(float(bbox.find('xmax').text)),
                            int(float(bbox.find('ymax').text))]
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

# pascal_vol 数据格式的检测模型 计算 ap 
def voc_eval(detpath, annopath, imageset_file, classname, annocache, ovthresh=0.5, use_07_metric=False):
    """
    pascal voc evaluation
    :param detpath: detection results detpath.format(classname)
    :param annopath: annotations annopath.format(classname)
    :param imageset_file: text file containing list of images  # 这个是测试数据集的 图片列表文件
    :param classname: category name    # 这个是 检测的类别名称（计算这个类 的AP）
    :param annocache: caching annotations
    :param ovthresh: overlap threshold  # iou 的阈值
    :param use_07_metric: whether to use voc07's 11 point ap computation
    :return: rec, prec, ap
    """
    # 读取分析 测试图片列表文件  ，将图片的名称放到 image_filenames 这个list中（需要注意的是 图片名称 ？？？）
    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    image_filenames = [x.strip() for x in lines]

    # load annotations from xml file 
    recs = {}
    for ind, image_filename in enumerate(image_filenames):
        # annopath.format(image_filename) 这个换成该图片对应的 xml  文件路径 绝对路径
        recs[image_filename] = parse_voc_rec(annopath.format(image_filename))
        if ind % 100 == 0:
            print('reading annotations for {:d}/{:d}'.format(ind + 1, len(image_filenames)))

    # extract objects in :param classname:
    class_recs = {}
    npos = 0  # npos 存储是 真实的标注数据中，关于这个类有多少 标注信息
    for image_filename in image_filenames:
        objects = [obj for obj in recs[image_filename]
                   if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in objects])
        difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
        det = [False] * len(objects)  # stand for detected
        npos = npos + sum(~difficult) # ~ 这里是取反的意思呢
        class_recs[image_filename] = {'bbox': bbox,
                                      'difficult': difficult,
                                      'det': det}

    # read detections
    # detfile  是关于这个类的 模型检测的结果数据文件
    # 每行的格式是：以空格 分割开;每行一个预测为该类的 bbox
    # <image identifier> <confidence> <left> <top> <right> <bottom>
    """
        000004 0.702732 89 112 516 466
        000006 0.870849 373 168 488 229
        000006 0.852346 407 157 500 213
        000006 0.914587 2 161 55 221
        000008 0.532489 175 184 232 201
    """
    detfile = detpath.format(classname) 
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    bbox = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    if bbox.shape[0] > 0:
        sorted_inds = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        bbox = bbox[sorted_inds, :]
        image_ids = [image_ids[x] for x in sorted_inds]

    # go down detections and mark true positives and false positives
    nd = len(image_ids)  # 在这里 nd  指的是 预测为这个类的bbox 有多少（image_ids list 里有重复的）
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd): # 遍历分析 预测的bbox
        # image_ids[d] --- image_id; class_recs[image_id] -- 返回这个图片的标注信息
        r = class_recs[image_ids[d]]
        bb = bbox[d, :].astype(float)
        ovmax = -np.inf
        bbgt = r['bbox'].astype(float)

        if bbgt.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(bbgt[:, 0], bb[0])
            iymin = np.maximum(bbgt[:, 1], bb[1])
            ixmax = np.minimum(bbgt[:, 2], bb[2])
            iymax = np.minimum(bbgt[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                   (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not r['difficult'][jmax]:
                if not r['det'][jmax]:
                    tp[d] = 1.
                    r['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid division by zero in case first detection matches a difficult ground ruth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def voc_eval_sds(det_file, seg_file, devkit_path, image_list, cls_name, cache_dir,
                 class_names, mask_size, binary_thresh, ov_thresh=0.5):
    # 1. Check whether ground truth cache file exists
    with open(image_list, 'r') as f:
        lines = f.readlines()
    image_names = [x.strip() for x in lines]
    check_voc_sds_cache(cache_dir, devkit_path, image_names, class_names)
    gt_cache = cache_dir + '/' + cls_name + '_mask_gt.pkl'
    with open(gt_cache, 'rb') as f:
        gt_pkl = cPickle.load(f)

    # 2. Get predict pickle file for this class
    with open(det_file, 'rb') as f:
        boxes_pkl = cPickle.load(f)
    with open(seg_file, 'rb') as f:
        masks_pkl = cPickle.load(f)

    # 3. Pre-compute number of total instances to allocate memory
    num_image = len(image_names)
    box_num = 0
    for im_i in xrange(num_image):
        box_num += len(boxes_pkl[im_i])

    # 4. Re-organize all the predicted boxes
    new_boxes = np.zeros((box_num, 5))
    new_masks = np.zeros((box_num, mask_size, mask_size))
    new_image = []
    cnt = 0
    for image_ind in xrange(len(image_names)):
        boxes = boxes_pkl[image_ind]
        masks = masks_pkl[image_ind]
        num_instance = len(boxes)
        for box_ind in xrange(num_instance):
            new_boxes[cnt] = boxes[box_ind]
            new_masks[cnt] = masks[box_ind]
            new_image.append(image_names[image_ind])
            cnt += 1

    # 5. Rearrange boxes according to their scores
    seg_scores = new_boxes[:, -1]
    keep_inds = np.argsort(-seg_scores)
    new_boxes = new_boxes[keep_inds, :]
    new_masks = new_masks[keep_inds, :, :]
    num_pred = new_boxes.shape[0]
    import cv2
    # 6. Calculate t/f positive
    fp = np.zeros((num_pred, 1))
    tp = np.zeros((num_pred, 1))
    for i in xrange(num_pred):
        pred_box = np.round(new_boxes[i, :4]).astype(int)
        pred_mask = new_masks[i]
        pred_mask = cv2.resize(pred_mask.astype(
            np.float32), (pred_box[2] - pred_box[0] + 1, pred_box[3] - pred_box[1] + 1))
        pred_mask = pred_mask >= binary_thresh
        image_index = new_image[keep_inds[i]]

        if image_index not in gt_pkl:
            fp[i] = 1
            continue
        gt_dict_list = gt_pkl[image_index]
        # calculate max region overlap
        cur_overlap = -1000
        cur_overlap_ind = -1
        for ind2, gt_dict in enumerate(gt_dict_list):
            gt_mask_bound = np.round(gt_dict['mask_bound']).astype(int)
            pred_mask_bound = pred_box
            ov = mask_overlap(gt_mask_bound, pred_mask_bound,
                              gt_dict['mask'], pred_mask)
            if ov > cur_overlap:
                cur_overlap = ov
                cur_overlap_ind = ind2
        if cur_overlap >= ov_thresh:
            if gt_dict_list[cur_overlap_ind]['already_detect']:
                fp[i] = 1
            else:
                tp[i] = 1
                gt_dict_list[cur_overlap_ind]['already_detect'] = 1
        else:
            fp[i] = 1

    # 7. Calculate precision
    num_pos = 0
    for key, val in gt_pkl.iteritems():
        num_pos += len(val)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_pos)
    # avoid divide by zero in case the first matches a difficult gt
    prec = tp / np.maximum(fp+tp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, True)
    return ap


def parse_inst(image_name, devkit_path):
    """
    Get cooresponding masks, boxes, classes according to image name
    Args:
        image_name: input image name
        devkit_path: root dir for devkit SDS
    Returns:
        roi/mask dictionary of this image
    """
    import PIL
    seg_obj_name = os.path.join(
        devkit_path, 'SegmentationObject', image_name + '.png')
    seg_obj_data = PIL.Image.open(seg_obj_name)
    seg_obj_data = np.array(seg_obj_data.getdata(), np.uint8).reshape(
        seg_obj_data.size[1], seg_obj_data.size[0])

    seg_cls_name = os.path.join(
        devkit_path, 'SegmentationClass', image_name + '.png')
    seg_cls_data = PIL.Image.open(seg_cls_name)
    seg_cls_data = np.array(seg_cls_data.getdata(), np.uint8).reshape(
        seg_cls_data.size[1], seg_cls_data.size[0])

    unique_inst = np.unique(seg_obj_data)
    # delete background pixels
    background_ind = np.where(unique_inst == 0)[0]
    unique_inst = np.delete(unique_inst, background_ind)
    record = []
    for inst_ind in xrange(unique_inst.shape[0]):
        [r, c] = np.where(seg_obj_data == unique_inst[inst_ind])
        mask_bound = np.zeros(4, dtype=int)
        mask_bound[0] = np.min(c)
        mask_bound[1] = np.min(r)
        mask_bound[2] = np.max(c)
        mask_bound[3] = np.max(r)
        mask = seg_obj_data[mask_bound[1]:mask_bound[3] +
                            1, mask_bound[0]:mask_bound[2]+1]
        mask = (mask == unique_inst[inst_ind])
        mask_cls = seg_cls_data[mask_bound[1]:mask_bound[3]+1, mask_bound[0]:mask_bound[2]+1]
        mask_cls = mask_cls[mask]
        num_cls = np.unique(mask_cls)
        assert num_cls.shape[0] == 1
        cur_inst = num_cls[0]
        record.append({
            'mask': mask,
            'mask_cls': cur_inst,
            'mask_bound': mask_bound
        })

    return record


def check_voc_sds_cache(cache_dir, devkit_path, image_names, class_names):

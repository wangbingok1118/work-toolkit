# -*- coding:utf-8 -*-
import os
import sys
import json
import random
import numpy as np
import time

def getTimeFlag():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())

def delete_jsonList_line_labelInfo(flag=None, line=None):
    """
    用于去除 题库中的一行 jsonlist  包含的 label 信息
    flag == 0:
        class
    flag == 1:
        cluster
    flag == 2
        detect
    return : jsonlist line without label info
    """
    # print(line)
    resultLine = None
    if flag == 0:
        line_dict = json.loads(line)
        line_dict['label']['class'] = dict()
        resultLine = json.dumps(line_dict)
    return resultLine


def get_jsonList_line_labelInfo(flag=None, line=None):
    """
    用于获取打标过的一行jsonlist 包含的 label 信息
    flag == 0:
        class
    flag == 1:
        cluster
    flag == 2
        detect
    return key:value
            key is url
            value : label info ()
    """
    key = None
    value = None
    if flag == 0:
        line_dict = json.loads(line)
        key = line_dict['url']
        if 'class' not in line_dict['label']:
            return key, None
        if 'pulp' not in line_dict['label']['class']:
            return key,None
        else:
            value = line_dict['label']['class']['pulp']
    return key, value


def judge_labeled_sand_line(sandLine=None,labeledLine=None,flag=0):
    result = None
    if flag == 0:
        key_sand,sand_value = get_jsonList_line_labelInfo(flag=flag, line=sandLine)
        key_labeled, labeled_value = get_jsonList_line_labelInfo(
            flag=flag, line=labeledLine)
        assert key_sand == key_labeled, "judge_labeled_sand_line error %s , %s" % (key_sand, key_labeled)
        if sand_value == labeled_value:
            result = True
        else:
            result = False
    return result

    pass



def getSandFromLibrary(libraryFile=None, sandNum=None, sandFile=None, sandClsRatio=None, dataFlag=0):
    """
        libraryFile : library file name absolute path
        sandNum : get sand num ,type int
        sandFile : store sand file absolute path
        sandClsRatio : type list eg : ['pulp', 'sexy', 'normal', '2', '2', '1']
        dataFlag : class or cluster or detect
    """
    if len(sandClsRatio)>0 and dataFlag == 0:
        sandList = []
        library_dict = dict() # key--classLabel : value--list
        with open(libraryFile, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) <= 0:
                    continue
                img_url, class_label = get_jsonList_line_labelInfo(line=line, flag=dataFlag)
                if class_label == None:
                    continue
                if class_label in library_dict:
                    library_dict.get(class_label).append(line)
                else:
                    library_dict[class_label] = [line]
        class_label_length = int(len(sandClsRatio) / 2)
        class_label_ratio = [int(i) for i in sandClsRatio[class_label_length:]]
        class_label_Num = [i*(sandNum / np.sum(class_label_ratio)) for i in class_label_ratio]
        class_label_Num = [int(i) for i in class_label_Num]
        if np.sum(class_label_Num) < sandNum:
            class_label_Num[-1] = sandNum - np.sum(class_label_Num[:-1])
        for i in range(0, class_label_length):
            label = sandClsRatio[i]
            num = class_label_Num[i]
            if len(library_dict.get(label)) < num:
                print("library : class %s count %d < sand %d " %(label, len(library_dict.get(label)),num))
                return "Error"
            for line in random.sample(library_dict.get(label), num):
                sandList.append(line)
        with open(sandFile,'w') as f:
            # random.shuffle(sandList)
            f.write('\n'.join(sandList))
            f.write('\n')
        return ['success',sandFile]
    elif len(sandClsRatio) == 0 and dataFlag == 0:
        sandList = []
        line_list = []
        with open(libraryFile, 'r') as f:
            for line  in f.readlines():
                line = line.strip()
                if len(line) <= 0:
                    continue
                img_url, class_label = get_jsonList_line_labelInfo(
                    line=line, flag=dataFlag)
                if not class_label:
                    continue
                else:
                     line_list.append(line)
            sandList = random.sample(
                line_list, sandNum)  # random get sandNum
        with open(sandFile, 'w') as f:
            # random.shuffle(sandList)
            f.write('\n'.join(sandList))
            f.write('\n')
        return ['success', sandFile]



def addSandToLogFile(logFile=None,sandFile=None,resultFile=None,dataFlag=None):
    resultList=[]
    with open(logFile,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) < 0:
                continue
            resultList.append(line)
    with open(sandFile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) < 0:
                continue
            line = delete_jsonList_line_labelInfo(flag=dataFlag,line=line)
            resultList.append(line)
    random.shuffle(resultList)
    with open(resultFile,'w') as f:
        f.write('\n'.join(resultList))
        f.write('\n')
    return ['success', resultFile]


def computeAccuracy(sandFile=None, labeledFile=None, dataFlag=0, saveErrorFlag=False):
    # 拿到提交的标注结果，核算其中题目的正确率
    sand_dict = dict() # key:value -- url:line
    labeled_dict = dict()
    with open(sandFile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) <= 0:
                continue
            key, value = get_jsonList_line_labelInfo(line=line, flag=dataFlag)
            if key:
                sand_dict[key] = line
    withoutLabeledCount = 0
    with open(labeledFile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) <= 0:
                continue
            key, value = get_jsonList_line_labelInfo(line=line, flag=dataFlag)
            if not value:
                withoutLabeledCount += 1
            if key:
                labeled_dict[key] = line
    acc = 0.0
    allSandNum = 0
    accNum = 0
    errNum = 0
    label_error_list=[]
    for key in labeled_dict.keys():
        if key not in sand_dict:
            # the key is not sand
            continue
        allSandNum += 1
        sand_value_line= sand_dict.get(key)
        labeled_value_line = labeled_dict.get(key)
        res = judge_labeled_sand_line(sandLine=sand_value_line,
                                labeledLine=labeled_value_line,flag=dataFlag)
        if res == None:
            return "Error"
        if res:
            accNum += 1
        else:
            label_error_list.append(sand_value_line)
            # label_error_list.append(labeled_value_line)
            errNum += 1
    if saveErrorFlag:
        label_error_jsonlist_file = labeledFile[:labeledFile.rfind('.')]+'-SandGT.json'
        with open(label_error_jsonlist_file,'w') as f:
            f.write('\n'.join(label_error_list))
            f.write('\n')
        print("Label Error save file is : %s" % (label_error_jsonlist_file))
    print("sand number in the labeled file is %d"%(allSandNum))
    print("sand labeled acc num is %d"%(accNum))
    acc = accNum * 1.0 / allSandNum
    print("without label info count %d\tall label count %d" %
          (withoutLabeledCount, len(labeled_dict)))
    print("acc is : %f" % (acc))
    return acc


def getUnionInfoFromA_B_laneled(labeled_a_file=None, labeled_b_file=None, union_jsonlistFile=None, dataFlag=0):
    union_labeled_jsonlist = []
    a_dict = dict()
    b_dict = dict()
    with open(labeled_a_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) <=0 :
                continue
            key, value = get_jsonList_line_labelInfo(
                line=line, flag=dataFlag)
            if not value:
                continue
            a_dict[key] = [value, line]
    with open(labeled_b_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) <= 0:
                continue
            key, value = get_jsonList_line_labelInfo(
                line=line, flag=dataFlag)
            if not value:
                continue
            b_dict[key] = [value, line]

    for key in a_dict.keys():
        res = judge_labeled_sand_line(sandLine=a_dict.get(key)[
                                      1], labeledLine=b_dict.get(key)[1], flag=0)
        if res ==None:
            print("ERROR")
        if res:
            union_labeled_jsonlist.append(a_dict.get(key)[1])
    with open(union_jsonlistFile, 'w') as f:
        f.write('\n'.join(union_labeled_jsonlist))
        f.write('\n')
    return ['success',union_jsonlistFile]
    pass

# not used
ERROR_INFO_FLAG={
    1: {
        1 : "some params required"
    },
    2: {

    },
    3: {

    },
    4: {

    },
    5: {

    }

}

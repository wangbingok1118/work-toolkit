# -*- coding:utf-8 -*-
import os
import sys
import json
import random
import numpy as np
import time
import copy


def getTimeFlag():
    return time.strftime("%m%d%H", time.localtime())


def getFilePath_FileNameNotIncludePostfix(fileName=None):
    justFileName = os.path.split(fileName)[-1]
    filePath = os.path.split(fileName)[0]
    if '.' in justFileName:
        justFileName = justFileName[:justFileName.rfind('.')]
    return [filePath, justFileName, os.path.join(filePath, justFileName)]
    pass


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
    resultLine = None
    if flag == 0:
        line_dict = json.loads(line)
        for i_dict in line_dict['label']:
            i_dict['data'] = list()
        resultLine = json.dumps(line_dict)
    elif flag == 2:
        line_dict = json.loads(line)
        for i_dict in line_dict['label']:
            i_dict['data'] = list()
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
        value is list : 
            the element in the list is : 
                bbox list : 
                    [
                        {"class":"knives_true","bbox":[[43,105],[138,105],[138,269],[43,269]],"ground_truth":true},
                        {"class":"guns_true","bbox":[[62,33],[282,33],[282,450],[62,450]],"ground_truth":true},
                        {"class":"guns_true","bbox":[[210,5],[399,5],[399,487],[210,487]],"ground_truth":true}
                    ]
    return key:value 
            key is url
            value : label info ()
    """
    key = None
    value = None
    if flag == 0:
        line_dict = json.loads(line)
        key = line_dict['url']
        if line_dict['label'] == None or len(line_dict['label']) == 0:
            return key, None
        label_dict = line_dict['label'][0]
        if label_dict['data'] == None or len(label_dict['data']) == 0:
            return key, None
        data_dict = label_dict['data'][0]
        if 'class' not in data_dict or data_dict['class'] == None or len(data_dict['class']) == 0:
            return key, None
        value = data_dict['class']
    elif flag == 2:  # value is  all bbox info list , element is dict
        line_dict = json.loads(line)
        key = line_dict['url']
        if line_dict['label'] == None or len(line_dict['label']) == 0:
            return key, None
        label_dict = line_dict['label'][0]
        if label_dict['data'] == None or len(label_dict['data']) == 0:
            return key, None
        data_dict_list = label_dict['data']
        label_bbox_list_elementDict = []
        for bbox in data_dict_list:
            if 'class' not in bbox or bbox['class'] == None or len(bbox['class']) == 0:
                continue
            label_bbox_list_elementDict.append(bbox)
        if len(label_bbox_list_elementDict) == 0:
            value = None
        else:
            value = label_bbox_list_elementDict
    return key, value


def check_labelFile_urlList(labelxFormatFile=None,flag=None):
    # check url in labelxFormatFile 
    # input : labelx format file
    # return : [Flag,messageDict]
    #           Flag : True or False
    messageDict = dict() # key: url ,value : count
    url_count_dict = dict()
    with open(labelxFormatFile,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line is not None:
                url, _ = get_jsonList_line_labelInfo(flag=flag,line=line)
                if url in url_count_dict: # url already occur
                    url_count_dict[url] += 1
                else: # url first occur
                    url_count_dict[url] = 1
    for url_key in url_count_dict:
        if url_count_dict[url_key] > 1:
            messageDict[url_key] = url_count_dict[url_key]
    Flag = True
    if len(messageDict) > 0:
        Flag = False
    return [Flag, messageDict]


def get_IOU(bbox_a=None, bbox_b=None):
    """
    自定义函数，计算两矩形 IOU，传入为 [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
    """
    Reframe = [bbox_a[0][0], bbox_a[0][1], bbox_a[2][0], bbox_a[2][1]]
    GTframe = [bbox_b[0][0], bbox_b[0][1], bbox_b[2][0], bbox_b[2][1]]
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1, x2+width2)
    startx = min(x1, x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1, y2+height2)
    starty = min(y1, y2)
    height = height1+height2-(endy-starty)

    if width <= 0 or height <= 0:
        ratio = 0  # 重叠率为 0
    else:
        Area = width*height  # 两矩形相交面积
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
    # return IOU
    return ratio


def judge_labeled_sand_line(sandLine=None, labeledLine=None, flag=0, thresholdIou=0.7):
    """
        flag==0 : class 
            return [True] or [False] 
        flag==2 : detect
            return [accBboxNum,errorBboxNum,allSandNum,allLabeledNum,sand_and_error_bbox_list]
    """
    result = []

    def getBestMatchBbox(sand_bbox=None, labeled_value=None, labeled_bbox_flag=None):
        """
            这个函数的作用：根据沙子中的bbox ，遍历 labeled 的bbox , 找到一个可用的，且类别匹配的，最大iou 的 bbox
            return 
                index: -1 ,not found 
        """
        sand_bbox_name = sand_bbox['class']
        sand_bbox_bbox = sand_bbox['bbox']
        max_index = -1
        max_iou = 0
        for index, bbox_dict in enumerate(labeled_value):
            if bbox_dict['class'] != sand_bbox_name or labeled_bbox_flag[index] == False:
                continue
            temp_iou = get_IOU(bbox_a=sand_bbox_bbox, bbox_b=bbox_dict['bbox'])
            if temp_iou > max_iou:
                max_iou = temp_iou
                max_index = index
        return [max_index, max_iou]
    if flag == 0:
        key_sand, sand_value = get_jsonList_line_labelInfo(
            flag=flag, line=sandLine)
        key_labeled, labeled_value = get_jsonList_line_labelInfo(
            flag=flag, line=labeledLine)
        assert key_sand == key_labeled, "judge_labeled_sand_line error %s , %s" % (
            key_sand, key_labeled)
        if sand_value == labeled_value:
            result.append(True)
        else:
            result.append(False)
        result.append([sand_value, labeled_value])
        """
            if class:
            result : len(result) ==  2 ; first element is True or False ;second element is [sand_value, labeled_value]
            [
                True or False,
                [sand_value, labeled_value]
            ]
        """
    elif flag == 2:
        accBboxNum = 0
        errorBboxNum = 0
        sand_and_error_bbox_list = []
        key_sand, sand_value = get_jsonList_line_labelInfo(
            flag=flag, line=sandLine)
        key_labeled, labeled_value = get_jsonList_line_labelInfo(
            flag=flag, line=labeledLine)
        if sand_value == None:
            sand_value = []
        if labeled_value == None:
            labeled_value = []
        allSandNum = len(sand_value)
        allLabeledNum = len(labeled_value)
        labeled_bbox_flag = [True for i in range(allLabeledNum)]
        class_acc_err_dict = dict() # element : dict {class_name : {acc:num,err:num}}
        for sand_bbox_dict in sand_value:
            index, iou = getBestMatchBbox(
                sand_bbox=sand_bbox_dict, labeled_value=labeled_value, labeled_bbox_flag=labeled_bbox_flag)
            sand_bbox_dict_class_name = sand_bbox_dict['class']
            class_acc_err_dict_element_dict = dict()
            if sand_bbox_dict_class_name in class_acc_err_dict:
                class_acc_err_dict_element_dict = class_acc_err_dict.get(
                    sand_bbox_dict_class_name)
            else:
                class_acc_err_dict[sand_bbox_dict_class_name] = class_acc_err_dict_element_dict
                class_acc_err_dict_element_dict['acc'] = 0
                class_acc_err_dict_element_dict['err'] = 0
            if index != -1 and iou >= thresholdIou: # sand bbox matched
                labeled_bbox_flag[index] = False
                accBboxNum += 1
                class_acc_err_dict_element_dict['acc'] += 1
            else:  # sand bbox not match
                errorBboxNum += 1
                class_acc_err_dict_element_dict['err'] += 1
                error_bbox = copy.deepcopy(sand_bbox_dict)
                error_bbox['class'] = error_bbox['class']+'_GT'
                sand_and_error_bbox_list.append(error_bbox)
            pass
        for i, flag in enumerate(labeled_bbox_flag): # 由于只计算recall，所以标注的数据里，如果没有匹配上的框，不做处理
            if flag == False:                        # 只在保存的错误结果文件中存储。
                continue
            error_bbox = copy.deepcopy(labeled_value[i])
            error_bbox['class'] = error_bbox['class']+'_LB'
            sand_and_error_bbox_list.append(error_bbox)
        result = [accBboxNum, errorBboxNum, allSandNum,
                  allLabeledNum, sand_and_error_bbox_list,class_acc_err_dict]
    return result


def getSandFromLibrary(libraryFile=None, sandNum=None, sandFile=None, sandClsRatio=None, dataFlag=0):
    """
        libraryFile : library file name absolute path
        sandNum : get sand num ,type int
        sandFile : store sand file absolute path
        sandClsRatio : type list eg : ['pulp', 'sexy', 'normal', '2', '2', '1']
        dataFlag : class or cluster or detect
    """
    #check url
    res, message = check_labelFile_urlList(labelxFormatFile=libraryFile,flag=dataFlag)
    if res == False:
        print("url occur more than once : %s" % (libraryFile))
        print(message)
        exit()
    if len(sandClsRatio) > 0 and dataFlag == 0:
        sandList = []
        library_dict = dict()  # key--classLabel : value--list
        with open(libraryFile, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) <= 0:
                    continue
                img_url, class_label = get_jsonList_line_labelInfo(
                    line=line, flag=dataFlag)
                if class_label == None:
                    continue
                if class_label in library_dict:
                    library_dict.get(class_label).append(line)
                else:
                    library_dict[class_label] = [line]
        class_label_length = int(len(sandClsRatio) / 2)
        class_label_ratio = [int(i) for i in sandClsRatio[class_label_length:]]
        class_label_Num = [i*(sandNum / np.sum(class_label_ratio))
                           for i in class_label_ratio]
        class_label_Num = [int(i) for i in class_label_Num]
        if np.sum(class_label_Num) < sandNum:
            class_label_Num[-1] = sandNum - np.sum(class_label_Num[:-1])
        for i in range(0, class_label_length):
            label = sandClsRatio[i]
            num = class_label_Num[i]
            if len(library_dict.get(label)) < num:
                print("library : class %s count %d < sand %d " %
                      (label, len(library_dict.get(label)), num))
                return "Error"
            for line in random.sample(library_dict.get(label), num):
                sandList.append(line)
        with open(sandFile, 'w') as f:
            # random.shuffle(sandList)
            f.write('\n'.join(sandList))
            f.write('\n')
        return ['success', sandFile]
    elif len(sandClsRatio) == 0 and dataFlag == 0:
        sandList = []
        line_list = []
        with open(libraryFile, 'r') as f:
            for line in f.readlines():
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
    elif dataFlag == 2 and len(sandClsRatio) == 0:
        sandList = []
        line_list = []
        with open(libraryFile, 'r') as f:
            for line in f.readlines():
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


def addSandToLogFile(logFile=None, sandFile=None, resultFile=None, dataFlag=None):
    resultList = []
    with open(logFile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) <= 0:
                continue
            resultList.append(line)
    with open(sandFile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) <= 0:
                continue
            line = delete_jsonList_line_labelInfo(flag=dataFlag, line=line)
            resultList.append(line)
    random.shuffle(resultList)
    with open(resultFile, 'w') as f:
        f.write('\n'.join(resultList))
        f.write('\n')
    res,message = check_labelFile_urlList(labelxFormatFile=resultFile,flag=dataFlag)
    if res == False:
        print("url occur more than once : %s" % (resultFile))
        print(message)
        exit()
    return ['success', resultFile]


def addSandToLogFileDir(logFileDir=None, sandFile=None, resultFileDir=None, dataFlag=None):
    logFileList = sorted(os.listdir(logFileDir))
    logFileList = [fileName for fileName in logFileList if len(
        fileName) > 0 and fileName[0] != '.']
    for a_file in logFileList:
        logFile = os.path.join(logFileDir, a_file)
        if not os.path.exists(resultFileDir):
            os.makedirs(resultFileDir)
        resultFile = os.path.join(resultFileDir, a_file)
        return_resultFlag, return_resultFile = addSandToLogFile(logFile=logFile, sandFile=sandFile,
                                                                resultFile=resultFile, dataFlag=dataFlag)
        if return_resultFlag == "success":
            print("add sand to %s --- success --- %s" %
                  (logFile, return_resultFile))
    return ['success', resultFileDir]

    pass


def computeAccuracy(sandFile=None, labeledFile=None, dataFlag=0, saveErrorFlag=False, iou=0.7):
    res , message = check_labelFile_urlList(labelxFormatFile=sandFile,flag=dataFlag)
    if res == False:
        print("url occur more than once : %s" % (sandFile))
        print(message)
        exit()
    res, message = check_labelFile_urlList(
        labelxFormatFile=labeledFile, flag=dataFlag)
    if res == False:
        print("url occur more than once : %s" % (labeledFile))
        print(message)
        exit()
    # 拿到提交的标注结果，核算其中题目的正确率
    sand_dict = dict()  # key:value -- url:line
    labeled_dict = dict()
    with open(sandFile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) <= 0:
                continue
            key, value = get_jsonList_line_labelInfo(line=line, flag=dataFlag)
            if key:
                sand_dict[key] = line
            if value == None:
                print("WARNING : sand file : %s without class info" % (key))
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
                if key in labeled_dict.keys():
                    print("WARNING : label file : %s appear more than once " % (key))
                labeled_dict[key] = line
    if dataFlag == 0:  # class
        acc = 0.0
        allSandNum = 0
        accNum = 0
        errNum = 0
        sandIsPulp_labeledIsNotPulp_Num = 0
        sandIsNotPulp_labeledIsPulp_Num = 0
        label_error_sand_list = []
        label_error_labelxFile_list = []
        sand_every_class_accuracy_dict = dict()  
        for key in labeled_dict.keys():
            if key not in sand_dict:
                # the key is not sand
                continue
            allSandNum += 1
            sand_value_line = sand_dict.get(key)
            labeled_value_line = labeled_dict.get(key)
            res = judge_labeled_sand_line(sandLine=sand_value_line,
                                          labeledLine=labeled_value_line, flag=dataFlag)
            sand_class_name = res[1][0]
            temp_acc_err_dict = dict()
            if sand_class_name in sand_every_class_accuracy_dict:
                temp_acc_err_dict = sand_every_class_accuracy_dict[sand_class_name]
            else:
                sand_every_class_accuracy_dict[sand_class_name] = temp_acc_err_dict
                temp_acc_err_dict['acc'] = 0
                temp_acc_err_dict['err'] = 0
            if len(res) == 0:
                return "Error"
                exit() # if run this then the code run error ,so stop 
            if res[0]:
                accNum += 1
                temp_acc_err_dict['acc'] += 1
            else:
                label_error_sand_list.append(sand_value_line)
                label_error_labelxFile_list.append(labeled_value_line)
                errNum += 1
                temp_acc_err_dict['err'] += 1
            # get without labeled pulp
            sand_value = res[1][0]
            labeled_value = res[1][1]
            if sand_value == "pulp" and labeled_value != "pulp":
                sandIsPulp_labeledIsNotPulp_Num += 1
            elif sand_value != "pulp" and labeled_value == "pulp":
                sandIsNotPulp_labeledIsPulp_Num += 1
                pass
        if saveErrorFlag == True:
            label_error_sand_jsonlist_file = getFilePath_FileNameNotIncludePostfix(
                fileName=labeledFile)[2]+'-SandGT.json'  # sand
            with open(label_error_sand_jsonlist_file, 'w') as f:
                f.write('\n'.join(label_error_sand_list))
                f.write('\n')
            print("Label Error --sand Ground Truth save file is : %s" %
                  (label_error_sand_jsonlist_file))
            label_error_jsonlist_file = getFilePath_FileNameNotIncludePostfix(
                fileName=labeledFile)[2]+'-labeledError.json'  # sand
            with open(label_error_jsonlist_file, 'w') as f:
                f.write('\n'.join(label_error_labelxFile_list))
                f.write('\n')
            print("Label Error --labelX error save file is : %s" %
                  (label_error_jsonlist_file))
        print('*'*15+"  compute per class recall  "+'*'*15)
        for class_key in sand_every_class_accuracy_dict:
            temp_class_acc_err_dict = sand_every_class_accuracy_dict[class_key]
            sand_acc = temp_class_acc_err_dict['acc'] /(temp_class_acc_err_dict['acc']+temp_class_acc_err_dict['err'])
            print("class name : {class_name: >20s} , acc num : {acc: >5d} , err num : {err: >5d} , recall : {recall: >10f}".format(
                class_name=class_key, acc=temp_class_acc_err_dict['acc'], err=temp_class_acc_err_dict['err'], recall=sand_acc))
        print("sand number in the labeled file is %d" % (allSandNum))
        print("sand labeled acc num is %d" % (accNum))
        acc = accNum * 1.0 / allSandNum
        print("LABEL FILE : without labeledInfo count %d\tall Count %d" %
              (withoutLabeledCount, len(labeled_dict)))
        print("sand is pulp but labeled is not pulp num : %d" %
              (sandIsPulp_labeledIsNotPulp_Num))
        print("sand is not pulp but labeled is pulp num : %d" %
              (sandIsNotPulp_labeledIsPulp_Num))
        print("acc is : %.2f" % (acc*100))
        return acc
    elif dataFlag == 2:  # detect
        thresholdIou = float(iou)
        acc = 0.0
        allSandImageNum = 0
        allSandBboxNum = 0
        accSandBBoxNum = 0
        errSandBBoxNum = 0
        label_error_sand_list = []
        all_sand_bbox_acc_err_dict = dict() # 用来存储所有沙子的 每个类别的  ： acc & err num
        for key in labeled_dict.keys():
            if key not in sand_dict:
                # the key is not sand
                continue
            allSandImageNum += 1
            sand_value_line = sand_dict.get(key)
            labeled_value_line = labeled_dict.get(key)
            res = judge_labeled_sand_line(sandLine=sand_value_line,
                                          labeledLine=labeled_value_line, flag=dataFlag, thresholdIou=thresholdIou)
            if len(res) == 0:
                return "Error"
            else:
                accSandBBoxNum += res[0]
                allSandBboxNum += res[2]
                errSandBBoxNum += res[1]
                if len(res[4]) > 0:
                    sand_value_line_dict = json.loads(sand_value_line)
                    temp_sand_value_line_dict = copy.deepcopy(
                        sand_value_line_dict)
                    temp_sand_value_line_dict['label'][0]['data'] = res[4]
                    label_error_sand_list.append(
                        json.dumps(temp_sand_value_line_dict))
                # process per class acc num & err num
                the_key_class_acc_err_dict = res[5]
                for class_name_key in the_key_class_acc_err_dict: # 遍历这张图的 class : acc & err
                    class_name_acc_err_dict = the_key_class_acc_err_dict[class_name_key]
                    temp_acc_err_dict = dict()
                    if class_name_key in all_sand_bbox_acc_err_dict: # 先判断该类别，是否已有
                        temp_acc_err_dict = all_sand_bbox_acc_err_dict[class_name_key]
                    else:
                        all_sand_bbox_acc_err_dict[class_name_key] = temp_acc_err_dict
                    for i_key in class_name_acc_err_dict:  # acc , err 遍历统计 acc ,err
                        if i_key in temp_acc_err_dict:
                            temp_acc_err_dict[i_key] += class_name_acc_err_dict[i_key]
                        else:
                            temp_acc_err_dict[i_key] = class_name_acc_err_dict[i_key]
        if saveErrorFlag:
            label_error_sand_jsonlist_file = getFilePath_FileNameNotIncludePostfix(
                fileName=labeledFile)[2]+'-labeledError.json'
            with open(label_error_sand_jsonlist_file, 'w') as f:
                f.write('\n'.join(label_error_sand_list))
                f.write('\n')
            print("Label Error  save file is : %s" %
                  (label_error_sand_jsonlist_file))
        # compute per class : 
        print('*'*15+"  compute per class recall  "+'*'*15)
        for class_key in sorted(all_sand_bbox_acc_err_dict.keys()):
            temp_class_acc_err_dict = all_sand_bbox_acc_err_dict[class_key]
            recall = temp_class_acc_err_dict['acc']/(
                temp_class_acc_err_dict['acc']+temp_class_acc_err_dict['err'])
            print("class name : {class_name: >20s} , acc num : {acc: >5d} , err num : {err: >5d} , recall : {recall: >10f}".format(
                class_name=class_key, acc=temp_class_acc_err_dict['acc'], err=temp_class_acc_err_dict['err'], recall=recall))
        print("sand number in the labeled file is %d" % (allSandImageNum))
        print("sand bbox number is %d" % (allSandBboxNum))
        print("acc labeled sand bbox num is %d" % (accSandBBoxNum))
        acc = accSandBBoxNum * 1.0 / allSandBboxNum
        print("without label info count %d\tall label count %d" %
              (withoutLabeledCount, len(labeled_dict)))
        print("the sand recall is : %.2f" % (acc*100))
        return acc


def computeAccuracy_Floder(sandFile=None, labeledFile=None, dataFlag=0, saveErrorFlag=False, iou=0.7):
    labeledFileList = sorted(os.listdir(labeledFile))
    labeledFileList = [fileName for fileName in labeledFileList if len(
        fileName) > 0 and fileName[0] != '.']
    for a_file in labeledFileList:
        if "labeledError" in a_file or "SandGT" in a_file:
            continue
        a_file = os.path.join(labeledFile, a_file)
        print("*"*80)
        print("begin process file %s" % (a_file))
        computeAccuracy(sandFile=sandFile, labeledFile=a_file,
                        dataFlag=dataFlag, saveErrorFlag=saveErrorFlag, iou=iou)
    pass

def getUnionInfoFromA_B_laneled(labeled_a_file=None, labeled_b_file=None, union_jsonlistFile=None, sandFile=None, dataFlag=0):
    
    res, message = check_labelFile_urlList(
        labelxFormatFile=labeled_a_file, flag=dataFlag)
    if res == False:
        print("url occur more than once : %s" % (labeled_a_file))
        print(message)
        exit()
    res, message = check_labelFile_urlList(
        labelxFormatFile=labeled_b_file, flag=dataFlag)
    if res == False:
        print("url occur more than once : %s" % (labeled_b_file))
        print(message)
        exit()
    
    union_labeled_jsonlist = []
    a_dict = dict()
    b_dict = dict()
    sand_dict = dict()
    if sandFile:
        with open(sandFile, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) <= 0:
                    continue
                key, value = get_jsonList_line_labelInfo(
                    line=line, flag=dataFlag)
                if not value:
                    continue
                sand_dict[key] = [value, line]
    with open(labeled_a_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) <= 0:
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
        if sandFile and key in sand_dict:  # in sand file ,then the labeled info not put in the union file
            continue
        if a_dict.get(key) == None or b_dict.get(key) == None:
            continue
        res = judge_labeled_sand_line(sandLine=a_dict.get(
            key)[1], labeledLine=b_dict.get(key)[1], flag=0)
        if res == None:
            print("ERROR")
        if res[0]:
            union_labeled_jsonlist.append(a_dict.get(key)[1])
    with open(union_jsonlistFile, 'w') as f:
        f.write('\n'.join(union_labeled_jsonlist))
        f.write('\n')
    return ['success', union_jsonlistFile]
    pass

def excludeSand(sandFile=None, labeledFile=None, saveExcludeFile=None, dataFlag=2):
    res, message = check_labelFile_urlList(
        labelxFormatFile=sandFile, flag=dataFlag)
    if res == False:
        print("url occur more than once : %s" % (sandFile))
        print(message)
        exit()
    res, message = check_labelFile_urlList(
        labelxFormatFile=labeledFile, flag=dataFlag)
    if res == False:
        print("url occur more than once : %s" % (labeledFile))
        print(message)
        exit()
    labeledWithoutSand_list = []
    sand_dict = dict()  # key:value -- url:line
    with open(sandFile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) <= 0:
                continue
            key, value = get_jsonList_line_labelInfo(line=line, flag=dataFlag)
            if key:
                sand_dict[key] = line
    with open(labeledFile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) <= 0:
                continue
            key, value = get_jsonList_line_labelInfo(line=line, flag=dataFlag)
            if key and key not in sand_dict and value != None:
                labeledWithoutSand_list.append(line)
    with open(saveExcludeFile, 'w') as f:
        f.write('\n'.join(labeledWithoutSand_list))
        f.write('\n')
    pass


def excludeSand_Floder(sandFile=None, labeledFile=None, dataFlag=2):
    labeledFileList = sorted(os.listdir(labeledFile))
    saveExcludeFile_dir = labeledFile+'-excludeSand'
    if not os.path.exists(saveExcludeFile_dir):
        os.makedirs(saveExcludeFile_dir)
    labeledFileList = [fileName for fileName in labeledFileList if len(
        fileName) > 0 and fileName[0] != '.']
    for a_file in labeledFileList:
        a_labeledFile = os.path.join(labeledFile, a_file)
        a_saveExcludeFile = os.path.join(saveExcludeFile_dir, a_file)
        excludeSand(sandFile=sandFile, labeledFile=a_labeledFile,
                    saveExcludeFile=a_saveExcludeFile, dataFlag=dataFlag)
    pass


ERROR_INFO_FLAG = dict()

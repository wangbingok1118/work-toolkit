# -*- coding:utf-8 -*-

import labelX_helper as  labelX_helper
from labelX_helper import ERROR_INFO_FLAG
import os
import sys
import json
import argparse
import pprint
import random
helpInfoStr = """
    actionFlag : 功能flag
        1  : 从题库中抽取沙子
            --libraryJsonList required 指向题库文件的绝对路径
            --sandNum required 从题库中抽取沙子的量
            --sandJsonList required 抽取出的沙子保存到该文件
            --sandClsRatio optional 沙子类别比例 
                                    eg: --sandClsRatio pulp,sexy,normal,2,2,1
                                    如果没有指定这个参数，那么就随机抽取
        2  : 从题库中抽取沙子 并添加到 日志jsonlist文件中、shuffle 最终文件,生成 jsonlist 文件
            --logJsonList required 指向由日志生成的log jsonlist 文件
            --libraryJsonList required 指向题库文件的绝对路径
            --sandNum required 从题库中抽取沙子的量
            --sandClsRatio optional 沙子类别比例 eg: --sandClsRatio pulp,sexy,normal,2,2,1
            --sandJsonList optional 抽取出的沙子保存到该文件,如果不指定该参数，则在log文件名后添加 -sand******* ,作为沙子文件
            --addedSandLogJsonList optional 日志log文件添加沙子后形成的 jsonlist文件,如果没有指定 则 -addsand*******
        3  : 将指向的沙子文件 添加到日志log jsonlist 文件中，并 shuffle 生成 jsonlist 文件
            --logJsonList required 指向由日志生成的log jsonlist 文件
            --sandJsonList required 抽取出的沙子文件
            --addedSandLogJsonList optional 日志log文件添加沙子后形成的 jsonlist文件,如果没有指定 则 -addsand*******
        4  : 计算标注过的数据 正确率
            --labeledJsonList required 指向已经打标过的jsonlist 文件
            ### sandJsonList or libraryJsonList 这两个参数必须要指定一个
            --sandJsonList optional 抽取出的沙子文件
            --libraryJsonList optional 指向题库文件的绝对路径
            --iou optional 当检测数据的时候，计算正确率,默认是 0.7 (检测数据)
            --outputErrorFlag optional 是否输出打标错误的(bool类型),默认 False。
                如果 True: 
                    则将打标错误的记录 保存到： --labeledJsonList 这个指定的文件 + '-labeledError.json' 形成的文件
                    打标错误行--对应的沙子信息，保存到 --labeledJsonList 这个指定的文件 + '-SandGT.json' 形成的文件
        5  : 根据两份标注数据，取交集，保存
            --labeledJsonList_a required 指向已经打标过的jsonlist 文件
            --labeledJsonList_b required 指向已经打标过的jsonlist 文件
            --sandJsonList  optional 指向沙子文件，如果设定这个参数，那么交集结果文件中就不包含打标时候参加进入的沙子了
            --finalUnionJsonList optional 指向 a,b 打标一致的结果文件 ，如果不写的花 labeledJsonList_a-union-labeledJsonList_b*******
        6  : 将指向的沙子文件 添加到一个文件夹下的所有 jsonlist 文件中，
                并 shuffle 生成 jsonlist 文件 保存到新的文件夹中，如：folder : folder-addSand
                为了方便 直接使用原始的命令行参数了
            --logJsonList required 指向需要添加沙子的jsonlist文件夹 ;
            --sandJsonList required 抽取出的沙子文件
            --addedSandLogJsonList optional 指向添加沙子后形成的新的保存jsonlist文件夹,
                如果没有指定 则 ***-addsand-timeFlag 作为新的文件夹   
        7  : 计算指定文件夹下的所有labelx数据的正确率:
            --logJsonList required 指向需要计算机的jsonlist文件夹 ;
            --sandJsonList required 用于计算正确率的沙子文件   
            --iou optional 当检测数据的时候，计算正确率,默认是 0.7（检测数据）
            --outputErrorFlag optional 是否保存打标错误的记录(bool类型),默认 False。
                如果 True: 
                    则将打标错误的记录 保存到： --labeledJsonList 这个指定的文件 + '-labeledError.json' 形成的文件
                    打标错误行--对应的沙子信息，保存到 --labeledJsonList 这个指定的文件 + '-SandGT.json' 形成的文件   
    dataTypeFlag :
        0 : class
        1 : cluster
        2 : detect
"""
def parse_args():
    parser = argparse.ArgumentParser(description="labelx toolkit", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--actionFlag', required=True,type=int, help="0 or 1 or 2 or 3 or 4")
    parser.add_argument('--dataTypeFlag',default=0,type=int,help="0:class ; 1:cluster ; 2:decect")
    
    # libraryJsonList 指向 题库 jsonlist 文件
    parser.add_argument(
        '--libraryJsonList',help='library json list file absolute path', default=None, type=str)
    # 抽取的沙子保存到该文件
    parser.add_argument(
        '--sandJsonList', help='get sand file absolute path', default=None, type=str)
    parser.add_argument(
        '--sandNum', help='the number of sand', type=int, default=1000)
    # 抽出沙子的各类的比例：pulp,sexy,normal,2,2,1
    parser.add_argument(
        '--sandClsRatio', help='class ratio eg : pulp,sexy,normal,2,2,1',default=None, type=str)
    
    # logJsonList 指向 log jsonlist 文件
    parser.add_argument(
        '--logJsonList',help='log jsonlist file  absolute path', default=None, type=str)
    # addedSandLogJsonList 指向 added sand jsonlist 文件
    parser.add_argument(
        '--addedSandLogJsonList', help='added sand jsonlist file absolute path', default=None, type=str)
    parser.add_argument('--labeledJsonList', help='labeled jsonlist file absolute path', default=None, type=str)
    # outputErrorFlag 是否保存打标错误的信息 默认 False 不保存
    parser.add_argument(
        '--outputErrorFlag', help='if ture then save label error info', default=False, type=bool)
    # labeledJsonList_a 指向 labeled jsonlist 文件
    parser.add_argument(
        '--labeledJsonList_a', help='labeled jsonlist file  absolute path', default=None, type=str)
    # labeledJsonList_b 指向 labeled jsonlist 文件
    parser.add_argument(
        '--labeledJsonList_b', help='labeled jsonlist file  absolute path', default=None, type=str)
    # labeledJsfinalUnionJsonListonList_b 指向 union labeled jsonlist 文件
    parser.add_argument(
        '--finalUnionJsonList', help='union labeled jsonlist file  absolute path', default=None, type=str)
    parser.add_argument(
        '--iou', help='detect compute bbox iou', default='0.7', type=str)
    args = parser.parse_args()
    return args


args = parse_args()
def main():
    actionFlag = args.actionFlag
    dataTypeFlag = args.dataTypeFlag
    if actionFlag == 1:
        sandClsRatio_list=[]
        if (not args.libraryJsonList) or (not args.sandJsonList) or (not args.sandNum):
            return 1
        if (not args.sandClsRatio) and dataTypeFlag == 0:
            print("分类数据，但没有设置沙子中类别比例，所以会随机抽取沙子")  # 分类数据，但没有设置沙子中类别比例，所以会随机抽取沙子
        if args.sandClsRatio:
            clsList = args.sandClsRatio.split(',')
            if len(clsList) %2 != 0:
                print("sandClsRatio error ; split by , eg : pulp,sexy,normal,2,2,1")
                return 2
            sandClsRatio_list = args.sandClsRatio.split(',')
        getSand_result = labelX_helper.getSandFromLibrary(
            libraryFile=args.libraryJsonList, sandNum=args.sandNum, sandFile=args.sandJsonList, sandClsRatio=sandClsRatio_list, dataFlag=dataTypeFlag)
        print("generate sand file is %s" % (getSand_result[1]))
        return 0
        pass
    elif actionFlag == 2:
        sandClsRatio_list = []
        if (not args.libraryJsonList) or (not args.logJsonList) or (not args.sandNum):
            return 1
        if (not args.sandClsRatio) and dataTypeFlag == 0:
            print("分类数据，但没有设置沙子中类别比例，所以会随机抽取沙子")  # 分类数据，但没有设置沙子中类别比例，所以会随机抽取沙子
        if args.sandClsRatio:
            clsList = args.sandClsRatio.split(',')
            if len(clsList) % 2 != 0:
                print("sandClsRatio error ; split by , eg : pulp,sexy,normal,2,2,1")
                return 2
            sandClsRatio_list = args.sandClsRatio.split(',')
        temp_sandJsonList = args.sandJsonList
        filePath_Name_list = labelX_helper.getFilePath_FileNameNotIncludePostfix(
            fileName=args.logJsonList)
        if not args.sandJsonList:
            temp_sandJsonList = filePath_Name_list[2]+'-sand-'+labelX_helper.getTimeFlag() + '.json'

        addedSandLogFile = args.addedSandLogJsonList
        if not args.addedSandLogJsonList:
            addedSandLogFile = filePath_Name_list[2] + \
                '-addedSand-'+labelX_helper.getTimeFlag()+'.json'
        getSand_result = labelX_helper.getSandFromLibrary(
            libraryFile=args.libraryJsonList, sandNum=args.sandNum, sandFile=temp_sandJsonList, sandClsRatio=sandClsRatio_list, dataFlag=dataTypeFlag)
        print("generate temp sand file is %s" % (getSand_result[1]))
        getAddedSandLogJsonList_result = labelX_helper.addSandToLogFile(
            logFile=args.logJsonList, sandFile=getSand_result[1], resultFile=addedSandLogFile,dataFlag=dataTypeFlag)
        print("generate added sand log file is %s" %
              (getAddedSandLogJsonList_result[1]))
        return 0
        pass
    elif actionFlag == 3:
        if (not args.logJsonList) or (not args.sandJsonList):
            return 1
        addedSandLogFile = args.addedSandLogJsonList
        filePath_Name_list = labelX_helper.getFilePath_FileNameNotIncludePostfix(
            fileName=args.logJsonList)
        if not args.addedSandLogJsonList:
            addedSandLogFile = filePath_Name_list[2]+ \
                '-addedSand-'+labelX_helper.getTimeFlag()+'.json'
        getAddedSandLogJsonList_result = labelX_helper.addSandToLogFile(
            logFile=args.logJsonList, sandFile=args.sandJsonList, resultFile=addedSandLogFile, dataFlag=dataTypeFlag)
        if getAddedSandLogJsonList_result[0] == 'success':
            print("generate added sand log file is %s"%(getAddedSandLogJsonList_result[1]))
            return 0
        else :
            -4
        pass
    elif actionFlag == 4:
        if (not args.labeledJsonList) or ((not args.sandJsonList) and (not args.libraryJsonList)):
            return 1
        sandFile = args.sandJsonList
        if not sandFile :
            sandFile = args.libraryJsonList
        acc = labelX_helper.computeAccuracy(
            sandFile=sandFile, labeledFile=args.labeledJsonList, dataFlag=dataTypeFlag, saveErrorFlag=args.outputErrorFlag, iou=args.iou)
        return 0
        pass
    elif actionFlag == 5:
        if (not args.labeledJsonList_a) or (not args.labeledJsonList_b):
            return 1
        unionFile = args.finalUnionJsonList
        filePath_Name_list_a = labelX_helper.getFilePath_FileNameNotIncludePostfix(
            fileName=args.labeledJsonList_a)
        filePath_Name_list_b = labelX_helper.getFilePath_FileNameNotIncludePostfix(
            fileName=args.labeledJsonList_b)
        if not unionFile:
            b_file = filePath_Name_list_b[1]
            unionFile = filePath_Name_list_a[2]+'-union-'+b_file + \
                '-result-'+labelX_helper.getTimeFlag()+'.json'
        union_result = labelX_helper.getUnionInfoFromA_B_laneled(
            labeled_a_file=args.labeledJsonList_a, labeled_b_file=args.labeledJsonList_b, union_jsonlistFile=unionFile, sandFile=args.sandJsonList, dataFlag=dataTypeFlag)
        if union_result[0] == 'success':
            print("generate union file is %s"%(union_result[-1]))
            return 0
        else:
            return -5
        pass
    elif actionFlag == 6:
        # 和 3 的功能类似，只不过这里是处理的文件夹
        if (not args.logJsonList) or (not args.sandJsonList):
            return 1
        addedSandLogFile = args.addedSandLogJsonList
        if not args.addedSandLogJsonList:
            inputDir = args.logJsonList
            if args.logJsonList[-1] == '/':
                inputDir = inputDir[:-1]
            addedSandLogFile = inputDir+'-addedSand-'+labelX_helper.getTimeFlag()
        res_list_two = labelX_helper.addSandToLogFileDir(
            logFileDir=args.logJsonList, sandFile=args.sandJsonList, resultFileDir=addedSandLogFile, dataFlag=dataTypeFlag)
        if res_list_two[0] == 'success':
            print("generate added sand folder is %s" %
                  (res_list_two[1]))
            return 0
        else:
            return -6
        pass
    elif actionFlag == 7:
        if (not args.logJsonList) or ((not args.sandJsonList) and (not args.libraryJsonList)):
            return 1
        sandFile = args.sandJsonList
        if not sandFile:
            sandFile = args.libraryJsonList
        labelX_helper.computeAccuracy_Floder(
            sandFile=sandFile, labeledFile=args.logJsonList, dataFlag=dataTypeFlag, saveErrorFlag=args.outputErrorFlag, iou=args.iou)
        pass
    else:
        return 3
    
    pass
if __name__ == '__main__':
    # print(helpInfoStr)
    print('*'*50)
    print('*'*20+"  param is :::   "+'*'*20)
    pprint.pprint(args)
    print('*'*20+"  begin runing   "+'*'*20)
    res = main()
    if res == 1:
        print(helpInfoStr)
    elif res == 3:
        print("actionFlag must is 1 or 2 or 3 or 4 or 5 or 6")
    elif  res ==0:
        print("THE SCRIPT RUN SUCCESS")
        pass

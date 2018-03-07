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
        5  : 根据两份标注数据，取交集，保存
            --labeledJsonList_a required 指向已经打标过的jsonlist 文件
            --labeledJsonList_b required 指向已经打标过的jsonlist 文件
            --finalUnionJsonList optional 指向 a,b 打标一致的结果文件 ，如果不写的花 labeledJsonList_a-union-labeledJsonList_b*******
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
    # labeledJsonList_a 指向 labeled jsonlist 文件
    parser.add_argument(
        '--labeledJsonList_a', help='labeled jsonlist file  absolute path', default=None, type=str)
    # labeledJsonList_b 指向 labeled jsonlist 文件
    parser.add_argument(
        '--labeledJsonList_b', help='labeled jsonlist file  absolute path', default=None, type=str)
    # labeledJsfinalUnionJsonListonList_b 指向 union labeled jsonlist 文件
    parser.add_argument(
        '--finalUnionJsonList', help='union labeled jsonlist file  absolute path', default=None, type=str)
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
        if not args.sandJsonList:
            temp_sandJsonList = args.logJsonList[:args.logJsonList.rfind(
                '.')]+'-sand-'+labelX_helper.getTimeFlag()+'.json'

        addedSandLogFile = args.addedSandLogJsonList
        if not args.addedSandLogJsonList:
            addedSandLogFile = args.logJsonList[:args.logJsonList.rfind(
                '.')]+'-addedSand-'+labelX_helper.getTimeFlag()+'.json'
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
        if not args.addedSandLogJsonList:
            addedSandLogFile = args.logJsonList[:args.logJsonList.rfind(
                '.')]+'-addedSand-'+labelX_helper.getTimeFlag()+'.json'
        getAddedSandLogJsonList_result = labelX_helper.addSandToLogFile(
            logFile=args.logJsonList, sandFile=args.sandJsonList, resultFile=addedSandLogFile, dataFlag=dataTypeFlag)
        if getAddedSandLogJsonList_result[0] == 'success':
            print("generate added sand log file is %s"%(getAddedSandLogJsonList_result[1]))
            return 0
        pass
    elif actionFlag == 4:
        if (not args.labeledJsonList) or ((not args.sandJsonList) and (not args.libraryJsonList)):
            return 1
        sandFile = args.sandJsonList
        if not sandFile :
            sandFile = args.libraryJsonList
        acc = labelX_helper.computeAccuracy(
            sandFile=sandFile, labeledFile=args.labeledJsonList, dataFlag=dataTypeFlag)
        return 0
        pass
    elif actionFlag == 5:
        if (not args.labeledJsonList_a) or (not args.labeledJsonList_b):
            return 1
        unionFile = args.finalUnionJsonList
        if not unionFile:
            b_file = args.labeledJsonList_b.split('/')[-1]
            unionFile = args.labeledJsonList_a[:args.labeledJsonList_a.rfind(
                '.')]+'-union-'+b_file[:b_file.rfind('.')]+'-result-'+labelX_helper.getTimeFlag()+'.json'
        union_result = labelX_helper.getUnionInfoFromA_B_laneled(
            labeled_a_file=args.labeledJsonList_a, labeled_b_file=args.labeledJsonList_b, union_jsonlistFile=unionFile, dataFlag=dataTypeFlag)
        if union_result[0] == 'success':
            print("generate union file is %s"%(union_result[-1]))
            return 0
        else:
            return -5
        pass
    else:
        return 3
    
    pass
if __name__ == '__main__':
    # print(helpInfoStr)
    pprint.pprint(args)
    res = main()
    if res == 1:
        print(helpInfoStr)
    elif res == 3:
        print("actionFlag must is 1 or 2 or 3 or 4 or 5")
    elif  res ==0:
        print("THE SCRIPT RUN SUCCESS")
        pass
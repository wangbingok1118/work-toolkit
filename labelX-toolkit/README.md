# work-toolkit

## 使用说明

```
python labelX_main.py
--actionFlag   1    # 这个指向 具体什么任务的标记 必须要指定
--dataTypeFlag 0  # 这个指向 打标数据类型 ; 0 分类 1 聚类  2 检测 （默认 0）
```

##  参数说明
```
    使用具体的功能，对应的输入参数不同，required 表示必须有的，optional 表示可以指定，也可以不指定的
    actionFlag
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
            --outputErrorFlag optional 是否输出打标错误的(bool类型),对应的沙子信息，模型 False。保存到 --labeledJsonList 这个指定的文件 + '-SandGT.json' 形成的文件
        5  : 根据两份标注数据，取交集，保存
            --labeledJsonList_a required 指向已经打标过的jsonlist 文件
            --labeledJsonList_b required 指向已经打标过的jsonlist 文件
            --finalUnionJsonList optional 指向 a,b 打标一致的结果文件 ，如果不写的花 labeledJsonList_a-union-labeledJsonList_b*******
    dataTypeFlag :
        0 : class
        1 : cluster
        2 : detect
```

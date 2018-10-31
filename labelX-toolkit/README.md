# work-toolkit

## 使用说明
python 3 for reacall 

```
python labelX_main.py
--actionFlag   1    # 这个指向 具体什么任务的标记 必须要指定
--dataTypeFlag 0  # 这个指向 打标数据类型 ; 0 分类 1 聚类  2 检测 （默认 0）
```

##  参数说明
```
    查看main文件中的说明
```
# updata log
## 2018-10-31
    add : sand process flag
    --deleteLabeledData optional 对添加的沙子标注信息处理
        参数选择值 ： 0 ： 默认值 （去掉标注信息）
                    1 ： 保留标注信息
                    2 ： 只有检测沙子用到（对标注框进行随机处理）
    --bboxRandomShuffleRata optional 检测沙子处理（bbox 处理概率 取值范围 [0，1] float)
        只要当 dataTypeFlag==2 && deleteLabeledData == 2  检测沙子情况下，使用。

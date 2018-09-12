import pandas as pd
import numpy as np
from collections import Counter
import os
import config

def process_data(type_ = 'train'):
    df = pd.read_csv('data/train.csv')
    #print(df)
    #print((df == 0).astype(int).sum(axis=0))
    # for i in df.columns.values.tolist():
    #     print(i,df[i].unique())

    #处理缺失值
    #由于complaint_level，former_complaint_num，former_complaint_fee 缺失值太多，先去掉
    #service2_caller_time 不知道是什么，先删除
    #age中包含数字和字符两种类型，且需要分组，先删除
    drop_feature = ['age','service2_caller_time','complaint_level','former_complaint_num','former_complaint_fee', 'user_id']
    df = df.drop(drop_feature, axis=1)

    #gender，age中为0的行删除
    df = df.drop(df[(df.gender==0)].index.tolist())
    #df = df.drop(df[(df.age==0)].index.tolist())
    
    #对age进行分组

    #加一个套餐内时长
    df['service_caller_time'] = df['local_caller_time']-df['service1_caller_time']
    #加一个当月可以使用总流量
    df['all_traffic_month'] = df['local_trafffic_month']+df['last_month_traffic']
    
    # df = pd.concat([df,service_caller_time])
    # df = pd.concat([df,all_traffic_month])
    print((df == 0).astype(int).sum(axis=0))

process_data()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:21:51 2023

@author: agagora
"""
import pandas as pd

df=pd.read_csv('/home/agagora/Downloads/DATASETS/airlinesatisfaction/train.csv')
de=pd.read_csv('/home/agagora/Downloads/DATASETS/airlinesatisfaction/test.csv')

print(df.shape)
print(df.columns)
print(df.index)
print(df.head(3))
df.to_csv('train_new.csv')
print("=======================")
dft=df['satisfaction']
print(set(dft))
print(dft.value_counts())
pp=list(dft).count('satisfied')
ppt=dft.count()
print('train_satisfied','\n','satisfied',pp,'Percentage:','{:.2f}'.format(pp/ppt*100))
print("==train=group by=satisfaction===")
dfg=df.groupby('satisfaction').count()
print(dfg)

print("=========TEST CSV==============")

de=de[de.columns[1:]]
print(de.shape)
print(de.columns)
print(de.index)
print(de.head()[de.columns[:4]])
print("=====================")
de.to_csv('test_new.csv')


print("==========================")

det=de['satisfaction']
print(det.value_counts())
pp=list(det).count('satisfied')
ppt=det.count()
print('test_satisfied','\n','satisfied',pp,'Percentage:','{:.2f}'.format(pp/ppt*100))
print("==test=group by=satisfaction===")

print((det=='satisfied').count())
deg=de.groupby('satisfaction').count()
print(deg)

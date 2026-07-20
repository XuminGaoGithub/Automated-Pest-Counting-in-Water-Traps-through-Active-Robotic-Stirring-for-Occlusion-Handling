#! /usr/bin/env python
# -*- coding:utf-8 -*-

# -*- coding: utf-8 -*-
'''
本脚本用来将所有的xml标注文件，随机划分成事先设定比例的训练集、验证集、训练验证集、测试集；结果是输出储存这些文件名（只是文件名而没有'.xml'这个后缀）的txt文件；从而得到到voc格式的数据集
本脚本在文件将数据集按7：2：1的比例分成训练集train、验证集val、测试集test
'''
import os
import random
random.seed(0)
xmlfilepath = r'./Annotations'
saveBasePath=r"./ImageSets/Main/"
 
#----------------------------------------------------------------------#
#   想要增加测试集修改trainval_percent
#   train_percent不需要修改
#----------------------------------------------------------------------#
train_percent = 0 #训练集在数据集中的比例
val_percent = 0   #验证集在数据集中的比例
trainval_percent = 0 #训练集和验证集加在一起在数据集中的比例
test_percent = 1   #测试集在数据集中的比例
 
temp_xml = os.listdir(xmlfilepath) #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)
 
num=len(total_xml) #num表示所有标注文件的总数量
total_list=range(num) #total_list是从0到1的总元素数量为num的列表。与total_xml列表一一对应
train_num = int(num*train_percent)#train_num表示所有标注文件中用于训练集的标注文件的数量
val_num = int(num*val_percent) #val_num表示所有标注文件中用于验证集的标注文件的数量
trainval_num = int(num*trainval_percent) #trainval_num表示所有标注文件中用于训练集和验证集的标注文件的数量
test_num = int(num*test_percent) #test_num表示所有标注文件中用于测试集的标注文件的数量
 
trainval_list = random.sample(total_list,trainval_num)#trainval_list时存储用于训练集和验证集的标注文件的序号的列表 #trainval = random.sample(list,tv)函数从列表list不分大小顺序的抽取tv个数，并抽取结果以列表的形式传给trainval
train_list = random.sample(trainval_list,train_num) #train_list是储存作为训练集中用于训练的标注文件在total_list列表中的序号
 
print("train and val size",trainval_num)
print("train size: {} , val size : {} , test size : {}".format(train_num,val_num,test_num))
 
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')
 
for i  in total_list:
    name=total_xml[i][:-4]+'\n' #因为voc格式数据集的储存训练和测试集信息的txt文件是只有文件名的，所以诸如“aa.xml”的文件只将“aa”写入到储存训练和测试集信息的txt文件中
    if i in trainval_list:
        ftrainval.write(name)
        if i in train_list:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
 
ftrainval.close()
ftrain.close()
fval.close()
ftest .close()


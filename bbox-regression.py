#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   bbox-regression.py    
@Version :   1.0 
@Author  :   2014Vee
@Contact :   1976535998@qq.com
@License :   (C)Copyright 2014Vee From UESTC
@Modify Time :   2020/4/14 10:37
@Desciption  :   None
'''

import cv2
import numpy as np
import xml.dom.minidom
import tensorflow as tf
import os
import time
from tensorflow.python.framework import graph_util

slim = tf.contrib.slim

#读取txt文件
train_txt = open('/data/lp/project/ssd.pytorch/txtsave/train.txt')
val_txt = open('/data/lp/project/ssd.pytorch/txtsave/val.txt')
train_content = train_txt.readlines()   #保存的train.txt中的内容
val_content = val_txt.readlines()  #保存的val.txt中的内容
# for linetr in train_content:
#     print ("train_content",linetr.rstrip('\n'))
# for lineva in val_content:
#     print ("val_content",lineva.rstrip('\n'))

#根据txt文件读取图像数据,并归一化图像，并保存缩放比例
train_imgs=[]#缩放后的图像尺寸
train_imgs_ratio=[] #width 缩放比，height缩放比
val_imgs=[]
val_imgs_ratio=[]


h=48
w=192  #归一化的尺寸
c=3   #通道


for linetr in train_content:
    img_path='/data/lp/project/ssd.pytorch/oripic/'+linetr.rstrip('\n')+'.jpg'
    img = cv2.imread(img_path)  #读取原图
#     print("image_name", str(linetr.rstrip('\n')))
#     print("imgshape", img.shape)
    imgresize= cv2.resize(img,(w,h)) #图像归一化
    ratio = np.array([imgresize.shape[0]/img.shape[0], imgresize.shape[1]/img.shape[1]],np.float32) #height缩放比 ,width 缩放比，
    train_imgs_ratio.append(ratio)
    train_imgs.append(imgresize)
train_img_arr = np.asarray(train_imgs,np.float32)  #保存训练图像数据的列表  h w c
print(len(train_img_arr),len(train_imgs_ratio))

for  lineva in val_content:
    img_path='/data/lp/project/ssd.pytorch/oripic/'+lineva.rstrip('\n')+'.jpg'
    img = cv2.imread(img_path) # h w c
    imgresize= cv2.resize(img,(w,h))  #h w c
    ratio = np.array([imgresize.shape[0]/img.shape[0], imgresize.shape[1]/img.shape[1]],np.float32) #height缩放比, width 缩放比，
    val_imgs_ratio.append(ratio)
    val_imgs.append(imgresize)
   # print(imgresize.shape[0], imgresize.shape[1], imgresize.shape[2])
val_img_arr = np.asarray(val_imgs,np.float32)  #保存验证图像的数据的列表 h w c

# print(len(val_img_arr),len(val_imgs_ratio))

# 根据txt文件读取xml,并获取xml中的坐标（xmin,ymin,xmax,ymax）(x表示width,y表示height),并获取经过缩放后的坐标
train_xml = []  # 保存标记的边框坐标
train_xml_resize = []  # 保存标记的边框坐标经过缩放后的坐标，缩放比与图像归一化的缩放比
val_xml = []
val_xml_resize = []
for linetr in train_content:
    xml_path = '/data/lp/project/ssd.pytorch/xml_zc_fz/' + linetr.rstrip(
        '\n') + '.xml'
    print(xml_path)
    xml_DomTree = xml.dom.minidom.parse(xml_path)
    xml_annotation = xml_DomTree.documentElement
    xml_object = xml_annotation.getElementsByTagName('object')
    xml_bndbox = xml_object[0].getElementsByTagName('bndbox')
    xmin_list = xml_bndbox[0].getElementsByTagName('xmin')
    xmin = int(xmin_list[0].childNodes[0].data)
    ymin_list = xml_bndbox[0].getElementsByTagName('ymin')
    ymin = int(ymin_list[0].childNodes[0].data)
    xmax_list = xml_bndbox[0].getElementsByTagName('xmax')
    xmax = int(xmax_list[0].childNodes[0].data)
    ymax_list = xml_bndbox[0].getElementsByTagName('ymax')
    ymax = int(ymax_list[0].childNodes[0].data)
    coordinate = np.array([ymin, xmin, ymax, xmax], np.int)  # h w h w
    train_xml.append(coordinate)  # 保存训练图像的xml的坐标
#     print("bbox:", coordinate)
# print(len(train_xml))

for lineva in val_content:
    xml_path = '/data/lp/project/ssd.pytorch/xml_zc_fz/' + lineva.rstrip(
        '\n') + '.xml'
    print(xml_path)
    xml_DomTree = xml.dom.minidom.parse(xml_path)
    xml_annotation = xml_DomTree.documentElement
    xml_object = xml_annotation.getElementsByTagName('object')
    xml_bndbox = xml_object[0].getElementsByTagName('bndbox')
    xmin_list = xml_bndbox[0].getElementsByTagName('xmin')
    xmin = int(xmin_list[0].childNodes[0].data)
    ymin_list = xml_bndbox[0].getElementsByTagName('ymin')
    ymin = int(ymin_list[0].childNodes[0].data)
    xmax_list = xml_bndbox[0].getElementsByTagName('xmax')
    xmax = int(xmax_list[0].childNodes[0].data)
    ymax_list = xml_bndbox[0].getElementsByTagName('ymax')
    ymax = int(ymax_list[0].childNodes[0].data)
    coordinate = np.array([ymin, xmin, ymax, xmax], np.int)
    val_xml.append(coordinate)  # 保存验证图像的xml的坐标
# print(len(val_xml))

for i in range(0, len(train_imgs_ratio)):
    ymin_ratio = train_xml[i][0] * train_imgs_ratio[i][0]
    xmin_ratio = train_xml[i][1] * train_imgs_ratio[i][1]
    ymax_ratio = train_xml[i][2] * train_imgs_ratio[i][0]
    xmax_ratio = train_xml[i][3] * train_imgs_ratio[i][1]
    coordinate_ratio = np.array([ymin_ratio, xmin_ratio, ymax_ratio, xmax_ratio], np.float32)
    train_xml_resize.append(coordinate_ratio)  # 保存训练图像的标记的xml的缩放后的坐标

for i in range(0, len(val_imgs_ratio)):
    ymin_ratio = val_xml[i][0] * val_imgs_ratio[i][0]
    xmin_ratio = val_xml[i][1] * val_imgs_ratio[i][1]
    ymax_ratio = val_xml[i][2] * val_imgs_ratio[i][0]
    xmax_ratio = val_xml[i][3] * val_imgs_ratio[i][1]
    coordinate_ratio = np.array([ymin_ratio, xmin_ratio, ymax_ratio, xmax_ratio], np.float32)
    val_xml_resize.append(coordinate_ratio)  # 保存训练验证图像的标记的xml的缩放后的坐标


# 按批次取数据，获取batchsize数据
# inputs 图像数据  归一化后的数据
# targets xml坐标数据  归一化后的数据
def getbatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]  # 其实就是按照batchsize做切片
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]  # 这个yield每次都是遇到了就返回类似于关键字return
        # 但是下次执行的时候就是从yield后面的代码进行继续，此时这个函数不是普通函数而是一个生成器了


#损失函数smoothL1范数
def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.

    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)#这个地方打开会有平方项
    return r

#构建网络结构

input_data = tf.placeholder(tf.float32,shape=[None,h,w,c],name='x')  #输入的图像数据（归一化后的图像数据）
input_bound = tf.placeholder(tf.float32,shape=[None,None],name='y') #输入的标记的边框坐标数据（缩放后的xml坐标）
prob=tf.placeholder(tf.float32, name='keep_prob')


#第一个卷积层（192——>96) （48--》24）
#conv1 = slim.repeat(input_data, 2, slim.conv2d, 32, [3, 3], scope='conv1')
conv1 = slim.conv2d(input_data,  32, [3, 3], scope='conv1')##32是指卷积核的个数，[3, 3]是指卷积核尺寸，默认步长是[1,1]
pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')#[2,2]是池化步长

#第二个卷积层（96-48） （24-》12）
#conv2 =  slim.repeat(pool1, 2, slim.conv2d, 64, [3, 3], scope='conv2')
conv2 = slim.conv2d(pool1, 64, [3, 3], scope='conv2')
pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')

#第三个卷积层（48-24） （12-》6）
#conv3 = slim.repeat(pool2, 2, slim.conv2d, 128, [3, 3], scope='conv3')
conv3 = slim.conv2d(pool2, 128, [3, 3], scope='conv3')
pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')

#第四个卷积层（24） （6）
conv4 = slim.conv2d(pool3, 256 ,[3, 3], scope='conv4')
dropout = tf.layers.dropout(conv4, rate=prob, training=True)
#dropout = tf.nn.dropout(conv4,keep_prob)
#pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')

#第五个卷积层（24-12） （6-》3）
#conv5 = slim.repeat(dropout, 2, slim.conv2d, 128, [3, 3], scope='conv5')
conv5 = slim.conv2d(dropout , 128, [3, 3], scope='conv5')
pool5 = slim.max_pool2d(conv5, [2, 2], scope='pool5')

#第六个卷积层（12-6） （3-》1）
#conv6 = slim.repeat(pool5, 2, slim.conv2d, 64, [3, 3], scope='conv6')
conv6 = slim.conv2d(pool5, 64, [3, 3], scope='conv6')
pool6 = slim.max_pool2d(conv6, [2, 2], scope='pool6')

reshape = tf.reshape(pool6, [-1, 6 * 1 * 64])
# print(reshape.get_shape())

fc = slim.fully_connected(reshape, 4, scope='fc')
# print(fc)
# print(input_data)

'''
#第七个卷积层（6-3） （1-》1）
conv7 = slim.conv2d(pool6,  32, [3, 3], scope='conv7')
pool7 = slim.max_pool2d(conv7, [2, 2], scope='pool7')

conv8 = slim.conv2d(pool7, 4, [3, 3], padding=None, activation_fn=None,scope='conv8')
'''


n_epoch =500
batch_size= 32
print (batch_size)


weights = tf.expand_dims(1. * 1., axis=-1)
loss = abs_smooth(fc - input_bound)#fc层和输入标签的差，用平滑L2范数做损失函数
# print(loss)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)#优化用的adam，学习率0.001

#correct_prediction = tf.equal(fc, input_bound)
#correct_prediction = tf.equal(tf.cast(fc,tf.int32), tf.cast(input_bound, tf.int32))

temp_acc = tf.abs(tf.cast(fc,tf.int32) - tf.cast(input_bound, tf.int32)) #fc出来之后的和标签做个差值
compare_np = np.ones((batch_size,4), np.int32) #建立一个和batch_size一样大小，4通道的compare_np
compare_np[:] = 3
print(compare_np)
compare_tf = tf.convert_to_tensor(compare_np) #
# print(compare_tf)
correct_prediction = tf.less(temp_acc,compare_tf)  ##temp_acc对应的元素如果比compare_tf对应的小，那么对应位置返回true
# print(correct_prediction)
loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')##求张量沿着某个方向的和，求完后可以降维度
tf.summary.scalar('loss',loss) #可视化观看常量
# print(loss)
accuracy= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))###tf.cast函数转换类型###
#tf.summary.scalar('accuracy',accuracy) #可视化观看常量
# print(accuracy)


# print(prob)

# pb_file_path = '/data/liuan/jupyter/root/project/keras-retinanet-master/bbox_fz_zc_006000/bbox_pb_model/ocr_bboxregress_batch16_epoch10000.pb'
pb_file_path = '/data/lp/project/ssd.pytorch/ocr_bbox_batch16_epoch'

# 设置可见GPU
gpu_no = '1'  # or '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
# 定义TensorFlow配置
config = tf.ConfigProto()
# 配置GPU内存分配方式
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
# config.gpu_options.per_process_gpu_memory_fraction = 0.8


sess = tf.InteractiveSession(config=config)

# ////////////////////////////////
# ckpt = tf.train.get_checkpoint_state('/home/data/wangchongjin/ad_image/model_save/')
# saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')   # 载入图结构，保存在.meta文件中
# saver.restore(sess,ckpt.model_checkpoint_path)
# //////////////////////////////////
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(
    "/data/lp/project/ssd.pytorch/ocr_bbox_batch16_epoch/record_graph", sess.graph_def)

# saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型


for epoch in range(n_epoch):
    start_time = time.time()

    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in getbatches(train_img_arr, train_xml_resize, batch_size, shuffle=False):
        _, err, acc = sess.run([train_op, loss, accuracy],
                               feed_dict={input_data: x_train_a, input_bound: y_train_a, prob: 0.5})
        train_loss += err
        train_acc += acc
        n_batch += 1

    #     print(epoch)
    #     print("   train loss: %f" % (train_loss/ n_batch))
    #     print("   train acc: %f" % (train_acc/ n_batch))

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in getbatches(val_img_arr, val_xml_resize, batch_size, shuffle=False):
        err, acc = sess.run([loss, accuracy], feed_dict={input_data: x_val_a, input_bound: y_val_a, prob: 0})
        # print(err)
        val_loss += err
        val_acc += acc
        n_batch += 1

        rs = sess.run([merged], feed_dict={input_data: x_val_a, input_bound: y_val_a, prob: 0})
        if n_batch is batch_size:
            writer.add_summary(rs[0], epoch)

    #     print("   validation loss: %f" % (val_loss/ n_batch))
    #     print("   validation acc: %f" % (val_acc/ n_batch))

    #    saver.save(sess, "/home/data/wangchongjin/ad_image/model_save_new/ad.ckpt")
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['fc/Relu'])

    with tf.gfile.FastGFile(pb_file_path + '_' + str(epoch) + '.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())

writer.close()
sess.close()
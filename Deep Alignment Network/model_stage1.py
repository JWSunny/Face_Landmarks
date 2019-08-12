# -*- coding: utf-8 -*-#
#-------------------------------------
# Name:         model_stage1
# Description:  
# Author:       sunjiawei
# Date:         2019/8/12
#-------------------------------------

import tensorflow as tf

''' 主要是利用 tensorflow实现，Deep Alignment Network 论文中阶段1的模型实现 '''
image_size = 112
n_landmarks = 68

####  定义损失函数
def landmarks_loss(GroudTruth, Prediction):
    Gt = tf.reshape(GroudTruth, [-1, n_landmarks, 2])
    Pt = tf.reshape(Prediction, [-1, n_landmarks, 2])
    ####  采用的均方根误差
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(Gt, Pt), 2)), 1)
    ### 论文中采用的其实是3中errors【其实就是归一化后的误差值】；
    # 一种是 均方根误差/外眼角之间的距离值
    # 一种是 均方根误差/瞳孔之间的距离值
    # 一种是 均方根误差/bounding box的对角线的值
    #####  这边尝试使用 左、右眼涉及的所有点的均值，计算两个点之间的距离
    norm = tf.norm(tf.reduce_mean(Gt[:, 36:42, :], 1) - tf.reduce_mean(Gt[:, 42:48, :], 1), axis=1)
    return loss/norm

#### 主要输入原始图(灰度图)，及 meanshape
def DAN_stage1(meanshape):
    meanshape = tf.constant(meanshape, dtype=tf.float32)
    input_image = tf.placeholder(tf.float32, [None, image_size, image_size, 1], name='input')   ### 输入的是灰度图
    gt_labels = tf.placeholder(tf.float32, [None, n_landmarks*2], name='output')   #### 输入的真实的68个关键点坐标
    #####  训练的时候为 True， 使用 batch_normalize 和 dropout一些操作
    S1_isTrain = tf.placeholder(tf.bool)

    with tf.variable_scope('stage1'):
        ####  首先是 4组 卷积操作层 + 池化层
        conv1a = tf.layers.batch_normalization(tf.layers.conv2d(input_image, 64,3,1, padding='same', activation=tf.nn.relu,\
        kernel_initializer=tf.glorot_normal_initializer()),training=S1_isTrain)
        conv1b = tf.layers.batch_normalization(tf.layers.conv2d(conv1a, 64, 3, 1,padding='same', activation=tf.nn.relu,\
        kernel_initializer=tf.glorot_uniform_initializer()), training=S1_isTrain)
        s1_pool1 = tf.layers.max_pooling2d(conv1b, 2, 2, padding='same')

        conv2a = tf.layers.batch_normalization(tf.layers.conv2d(s1_pool1, 128, 3, 1,padding='same', activation=tf.nn.relu,\
        kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        conv2b = tf.layers.batch_normalization(tf.layers.conv2d(conv2a, 128, 3, 1,padding='same', activation=tf.nn.relu,\
        kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        s1_pool2 = tf.layers.max_pooling2d(conv2b, 2, 2, padding='same')

        conv3a = tf.layers.batch_normalization(tf.layers.conv2d(s1_pool2, 256, 3, 1,padding='same', activation=tf.nn.relu,\
        kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        conv3b = tf.layers.batch_normalization(tf.layers.conv2d(conv3a, 256, 3, 1,padding='same', activation=tf.nn.relu,\
        kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        s1_pool3 = tf.layers.max_pooling2d(conv3b, 2, 2, padding='same')

        conv4a = tf.layers.batch_normalization(tf.layers.conv2d(s1_pool3, 512, 3, 1, padding='same', activation=tf.nn.relu,\
        kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        conv4b = tf.layers.batch_normalization(tf.layers.conv2d(conv4a, 512, 3, 1,padding='same', activation=tf.nn.relu,\
        kernel_initializer=tf.glorot_uniform_initializer()), training=S1_isTrain)
        s1_pool4 = tf.layers.max_pooling2d(conv4b, 2, 2, padding='same')

        #### 下面是 2个全连接层
        s1_pool4_flat = tf.contrib.layers.flatten(s1_pool4)
        s1_dropout = tf.layers.dropout(s1_pool4_flat, 0.7, training=S1_isTrain)   #### 防止过拟合，也可以"丢弃"部分节点；
        s1_fc1 = tf.layers.batch_normalization(tf.layers.dense(s1_dropout, 256, activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer()),
                    trainable=S1_isTrain, name='s1_fc1')
        s1_fc2 = tf.layers.dense(s1_fc1, n_landmarks * 2)

        S1_Ret = s1_fc2 + meanshape   ##### 其实使用残差的思想, 得到阶段1，预测的关键点的坐标
        S1_cost = tf.reduce_mean(landmarks_loss(gt_labels, S1_Ret))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'stage1')):
            S1_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S1_cost, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "stage1"))

    #####  返回预测的坐标值，  阶段1的损失值，
    return S1_Ret, S1_cost, S1_Optimizer


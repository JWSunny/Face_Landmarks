# -*- coding: utf-8 -*-#
#-------------------------------------
# Name:         model_stage2
# Description:  
# Author:       sunjiawei
# Date:         2019/8/13
#-------------------------------------

import tensorflow as tf

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

def LandmarkTransformLayer(Landmark, Param, Inverse=False):
    A = tf.reshape(Param[:, 0:4], [-1, 2, 2])
    T = tf.reshape(Param[:, 4:6], [-1, 1, 2])

    Landmark = tf.reshape(Landmark, [-1, n_landmarks, 2])
    if Inverse:
        A = tf.matrix_inverse(A)
        T = tf.matmul(-T, A)

    return tf.reshape(tf.matmul(Landmark, A) + T, [-1, n_landmarks * 2])

''' 主要是关于DAN model的阶段2 的tensorflow实现'''
def DAN_stage2(s2_input_features, S2_InputLandmark, S2_AffineParam, GroundTruth):
    
    '''
    :param s2_input_features:   阶段2输入的总特征【相似变换后的图片、关键点热图 和 上一阶段fc1的特征】
    :param S2_InputLandmark:   阶段2变换后的关键点坐标【公式2利用该坐标，生成关键点热图】
    :param S2_AffineParam:    相应的矩阵变换操作
    :param GroundTruth:      真实的68关键点坐标值
    :return:   预测坐标， loss， 优化的节点
    '''

    S2_isTrain = tf.placeholder(tf.bool)

    with tf.variable_scope('stage2'):
        concatInput = tf.layers.batch_normalization(s2_input_features, training=S2_isTrain)
        conv1a = tf.layers.batch_normalization(tf.layers.conv2d(concatInput, 64, 3, 1,padding='same', activation=tf.nn.relu,
        kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        conv1b = tf.layers.batch_normalization(tf.layers.conv2d(conv1a, 64, 3, 1,padding='same', activation=tf.nn.relu,
        kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        s2_pool1 = tf.layers.max_pooling2d(conv1b, 2, 2, padding='same')

        conv2a = tf.layers.batch_normalization(tf.layers.conv2d(s2_pool1, 128, 3, 1,padding='same', activation=tf.nn.relu,
        kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        conv2b = tf.layers.batch_normalization(tf.layers.conv2d(conv2a, 128, 3, 1, padding='same', activation=tf.nn.relu,
        kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        s2_pool2 = tf.layers.max_pooling2d(conv2b, 2, 2, padding='same')

        conv3a = tf.layers.batch_normalization(tf.layers.conv2d(s2_pool2, 256, 3, 1,padding='same', activation=tf.nn.relu,
        kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        conv3b = tf.layers.batch_normalization(tf.layers.conv2d(conv3a, 256, 3, 1, padding='same', activation=tf.nn.relu,
        kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        s2_pool3 = tf.layers.max_pooling2d(conv3b, 2, 2, padding='same')

        conv4a = tf.layers.batch_normalization(tf.layers.conv2d(s2_pool3, 512, 3, 1,padding='same', activation=tf.nn.relu,
        kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        conv4b = tf.layers.batch_normalization(tf.layers.conv2d(conv4a, 512, 3, 1, padding='same', activation=tf.nn.relu,
        kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
        s2_pool4 = tf.layers.max_pooling2d(conv4b, 2, 2, padding='same')

        s2_pool4_flat = tf.contrib.layers.flatten(s2_pool4)
        s2_DropOut = tf.layers.dropout(s2_pool4_flat, 0.5, training=S2_isTrain)

        s2_fc1 = tf.layers.batch_normalization(tf.layers.dense(s2_DropOut, 256,activation=tf.nn.relu,
        kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)

        s2_fc2 = tf.layers.dense(s2_fc1, n_landmarks * 2)

        s2_Ret = LandmarkTransformLayer(s2_fc2 + S2_InputLandmark, S2_AffineParam, Inverse=True)
        s2_Cost = tf.reduce_mean(landmarks_loss(GroundTruth, s2_Ret))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'Stage2')):
            s2_Optimizer = tf.train.AdamOptimizer(0.001).minimize(s2_Cost,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "stage2"))

    return s2_Ret, s2_Cost, s2_Optimizer
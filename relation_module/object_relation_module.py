#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import torch
import torch.nn as nn

# import torch.nn.init as init
from Modules import Linear
# from faster_rcnn.utils.cython_bbox import object_realtion #, p_softmax
from faster_rcnn.network import SpacialConv_new, np_to_variable


class ObjectRelationModule(nn.Module):

    def __init__(self, d_f, d_k, d_v, d_g):
        super(ObjectRelationModule, self).__init__()
        self.d_f = d_f
        self.d_k = d_k
        self.d_v = d_v
        self.d_g = d_g
        self.Nr = d_f / d_v
        self.proj_k = nn.Linear(self.d_f, self.d_k*self.Nr, bias=False)
        self.proj_q = nn.Linear(self.d_f, self.d_k*self.Nr, bias=False)
        self.proj_v = nn.Linear(self.d_f, self.d_f, bias=False)
        self.prod_g = nn.Conv2d(self.d_g, self.Nr, 1)
        self.spacial_conv = SpacialConv_new(pooling_size=32, d_g=self.d_g)


    def forward(self, rois, pooled_features, im_info, geometry_relation):
        key = self.proj_k(pooled_features)
        query = self.proj_q(pooled_features)
        value = self.proj_v(pooled_features)
        relation = self.appearance_realtion(query, key)
        geometry_weight = self.prod_g(geometry_relation)
        return self.compute_value(value, relation, geometry_weight,pooled_features)


    def appearance_realtion(self, query, key):
        query = query.data.cpu().numpy()
        key = key.data.cpu().numpy()
        key = key.transpose(1,0)
        count = key.shape[1]
        mat_relation = np.zeros([self.Nr,count,count],dtype=float)
        for i in range(self.Nr):
            mat_relation[i] = np.dot(query[:,i*self.d_k:(i+1)*self.d_k],key[i*self.d_k:(i+1)*self.d_k,:])
        mat_relation = np_to_variable(mat_relation,is_cuda=True)
        return mat_relation


    def compute_value(self, value, relation, geometry_weight,pooled_features):
        relation = relation.data.cpu().numpy()
        geometry_weight = np.squeeze(geometry_weight.data.cpu().numpy())
        pooled_features = pooled_features.data.cpu().numpy()

        value = value.data.cpu().numpy()
        count = relation.shape[1]
        weight =  np.zeros([self.Nr, count, count])
        add_feature = np.zeros([self.Nr, count, self.d_v])
        weight[0][:, :] = self.p_softmax(relation[0][:, :], geometry_weight[:, 0])
        add_feature[0][:, :] = np.dot(weight[0][:, :], value[:, 0 * self.Nr:(0 + 1) * self.Nr])
        feature = add_feature[0][:, :]
        for i in range(1, self.Nr):
            weight[i][:, :] = self.p_softmax(relation[i][:, :], geometry_weight[:, i])
            add_feature[i][:, :] = np.dot(weight[i][:, :], value[:, i*self.Nr:(i+1)*self.Nr])
            feature = np.hstack([feature, add_feature[i][:, :]])
        feature += pooled_features
        feature = np_to_variable(feature, is_cuda=True)
        return feature



    def p_softmax(self, relation, geometry_weight):
        num = relation.shape[1]
        min = relation.min()
        relation -= min
        relation = np.exp(relation)
        weight_temp = relation * geometry_weight.reshape(num,num)
        weight_sum = weight_temp.sum(axis=1)
        weight_sum = weight_sum[:, np.newaxis].repeat(num, axis=1)
        weight = weight_temp/weight_sum
        return weight












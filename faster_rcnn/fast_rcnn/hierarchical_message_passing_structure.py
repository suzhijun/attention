import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..network import FC
from torch.autograd import Variable
from torch.nn import Parameter
from ..utils.timer import Timer
# import ipdb
# from mps_base import Hierarchical_Message_Passing_Structure_base
from config import cfg

TIME_IT = False

class Hierarchical_Message_Passing_Structure(nn.Module):
	def __init__(self, nhidden, n_object_cats, n_predicate_cats):
		super(Hierarchical_Message_Passing_Structure, self).__init__()

		self.fc_object = FC(nhidden, n_object_cats, relu=True)
		self.fc_predicate = FC(nhidden, n_predicate_cats, relu=True)

		self.pred2sub = FC(n_predicate_cats, n_object_cats, relu=True)
		self.pred2obj = FC(n_predicate_cats, n_object_cats, relu=True)
		self.sub2obj = FC(n_object_cats, n_object_cats, relu=True)
		self.obj2sub = FC(n_object_cats, n_object_cats, relu=True)
		self.sub2pred = FC(n_object_cats, n_predicate_cats, relu=True)
		self.obj2pred = FC(n_object_cats, n_predicate_cats, relu=True)
		self.n_object_cats = n_object_cats
		self.n_predicate_cats = n_predicate_cats

	def forward(self, obj_labels, phrase_labels, mat_object, mat_phrase, object_features, phrase_features):

		# mps_object [object_batchsize, 2, n_phrase] : the 2 channel means inward(object) and outward(subject) list
		# mps_phrase [phrase_batchsize, 2]
		t = Timer()
		# ipdb.set_trace()

		# prepare transform data
		t.tic()
		pred_size = phrase_features.size(0)
		instance_size = object_features.size(0)

		transform_instance = self.fc_object(object_features)
		transform_pred = self.fc_predicate(phrase_features)

		pred_to_sub = self.pred2sub(phrase_labels)
		pred_to_obj = self.pred2obj(phrase_labels)
		sub_to_obj = self.sub2obj(obj_labels)
		obj_to_sub = self.obj2sub(obj_labels)
		sub_to_pred = self.sub2pred(obj_labels)
		obj_to_pred = self.obj2pred(obj_labels)

		sub_to_pred_matrix = mat_object[:, 0, :]
		sub_to_pred_matrix = torch.LongTensor(sub_to_pred_matrix).cuda()
		sub_weight = Variable(sub_to_pred_matrix.float().sum(dim=1) + 1e-7).unsqueeze(-1)
		obj_to_pred_matrix = mat_object[:, 1, :]
		obj_to_pred_matrix = torch.LongTensor(obj_to_pred_matrix).cuda()
		obj_weight = Variable(obj_to_pred_matrix.float().sum(dim=1) + 1e-7).unsqueeze(-1)

		sub_o_ind, obj_s_ind = torch.LongTensor(mat_phrase.T).cuda()
		sub_p_ind, pred_s_ind = sub_to_pred_matrix.nonzero().transpose(0, 1)
		obj_p_ind, pred_o_ind = obj_to_pred_matrix.nonzero().transpose(0, 1)

		if TIME_IT:
			torch.cuda.synchronize()
			print '\t\t[prepare data]:\t%.3fs' % (t.toc(average=False))

		# sub_to_obj
		t.tic()
		sub_to_obj_f = Variable(torch.zeros([instance_size, instance_size, self.n_object_cats]).cuda())
		sub_to_obj_f[obj_s_ind, sub_o_ind] = sub_to_obj[sub_o_ind]
		sub_to_obj_f = sub_to_obj_f.sum(dim=1)

		if TIME_IT:
			torch.cuda.synchronize()
			print '\t\t[sub_to_obj]:\t%.3fs' % (t.toc(average=False))

		# obj_to_sub
		t.tic()
		obj_to_sub_f = Variable(torch.zeros([instance_size, instance_size, self.n_object_cats]).cuda())
		obj_to_sub_f[sub_o_ind, obj_s_ind] = obj_to_sub[obj_s_ind]
		obj_to_sub_f = obj_to_sub_f.sum(dim=1)

		if TIME_IT:
			torch.cuda.synchronize()
			print '\t\t[obj_to_sub]:\t%.3fs' % (t.toc(average=False))

		# pred_to_sub
		t.tic()
		pred_to_sub_f = Variable(torch.zeros([instance_size, pred_size, self.n_object_cats]).cuda())
		pred_to_sub_f[sub_p_ind, pred_s_ind] = pred_to_sub[pred_s_ind]
		pred_to_sub_f = pred_to_sub_f.sum(dim=1)

		if TIME_IT:
			torch.cuda.synchronize()
			print '\t\t[pred_to_sub]:\t%.3fs' % (t.toc(average=False))

		# sub_to_pred
		t.tic()

		sub_to_pred_f = Variable(torch.zeros([pred_size, instance_size, self.n_predicate_cats]).cuda())
		sub_to_pred_f[pred_s_ind, sub_p_ind] = sub_to_pred[sub_p_ind]
		sub_to_pred_f = sub_to_pred_f.sum(dim=1)

		if TIME_IT:
			torch.cuda.synchronize()
			print '\t\t[sub_to_pred]:\t%.3fs' % (t.toc(average=False))

		# pred_to_obj
		t.tic()
		pred_to_obj_f = Variable(torch.zeros([instance_size, pred_size, self.n_object_cats]).cuda())
		pred_to_obj_f[obj_p_ind, pred_o_ind] = pred_to_obj[pred_o_ind]
		pred_to_obj_f = pred_to_obj_f.sum(dim=1)

		if TIME_IT:
			torch.cuda.synchronize()
			print '\t\t[pred_to_obj]:\t%.3fs' % (t.toc(average=False))

		# obj_to_pred
		t.tic()
		obj_to_pred_f = Variable(torch.zeros([pred_size, instance_size, self.n_predicate_cats]).cuda())
		obj_to_pred_f[pred_o_ind, obj_p_ind] = obj_to_pred[obj_p_ind]
		obj_to_pred_f = obj_to_pred_f.sum(dim=1)

		if TIME_IT:
			torch.cuda.synchronize()
			print '\t\t[obj_to_pred]:\t%.3fs' % (t.toc(average=False))

		out_obj_score = transform_instance + (pred_to_sub_f + obj_to_sub_f)/sub_weight + (pred_to_obj_f+sub_to_obj_f)/obj_weight
		out_phrase_score = transform_pred + sub_to_pred_f + obj_to_pred_f
		out_obj_labels = F.softmax(out_obj_score)
		out_phrase_labels = F.softmax(out_phrase_score)
		# out_obj_labels.register_hook(hook_tmp)
		# out_phrase_labels.register_hook(hook_tmp)
		return out_obj_score, out_phrase_score, out_obj_labels, out_phrase_labels

def hook_tmp(g):
	print(g)

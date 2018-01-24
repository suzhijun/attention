import cv2
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.timer import Timer
from utils.blob import im_list_to_blob
from fast_rcnn.nms_wrapper import nms
from rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes

import network
from network import Conv2d, FC
# from roi_pooling.modules.roi_pool_py import RoIPool
from roi_pooling.modules.roi_pool import RoIPool
import torchvision.models as models
from resnet import resnet50 as resnet50

def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
	dets = np.hstack((pred_boxes,
					  scores[:, np.newaxis])).astype(np.float32)
	keep = nms(dets, nms_thresh)
	if inds is None:
		return pred_boxes[keep], scores[keep]
	return pred_boxes[keep], scores[keep], inds[keep]


class RPN(nn.Module):
	_feat_stride = [16, ]

	anchor_scales_kmeans = [19.944, 9.118, 35.648, 42.102, 23.476, 15.882, 6.169, 9.702, 6.072, 32.254, 3.294, 10.148,
							22.443, 13.831, 16.250, 27.969, 14.181, 27.818, 34.146, 29.812, 14.219, 22.309, 20.360, 24.025, 40.593, ]
	anchor_ratios_kmeans = [2.631, 2.304, 0.935, 0.654, 0.173, 0.720, 0.553, 0.374, 1.565, 0.463, 0.985, 0.914, 0.734,
							2.671, 0.209, 1.318, 1.285, 2.717, 0.369, 0.718, 0.319, 0.218, 1.319, 0.442, 1.437, ]
	anchor_scales_normal = [4, 8, 16, 32, 64]
	anchor_ratios_normal = [0.25, 0.5, 1, 2, 4]

	def __init__(self, use_kmeans_anchors=False):
		super(RPN, self).__init__()

		if use_kmeans_anchors:
			print 'using k-means anchors'
			self.anchor_scales = self.anchor_scales_kmeans
			self.anchor_ratios = self.anchor_ratios_kmeans

		else:
			print 'using normal anchors'
			self.anchor_scales, self.anchor_ratios = \
				np.meshgrid(self.anchor_scales_normal, self.anchor_ratios_normal, indexing='ij')
			self.anchor_scales = self.anchor_scales.reshape(-1)
			self.anchor_ratios = self.anchor_ratios.reshape(-1)


		self.anchor_num = len(self.anchor_scales)
		# self.anchor_num_relationship = len(self.anchor_scales_relationship)

		self.features = models.vgg16(pretrained=True).features
		self.features.__delattr__('30') # to delete the max pooling
		# resnet = resnet50(pretrained=True)
		# self.features = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
		# 							  resnet.layer1, resnet.layer2, resnet.layer3)

		# by default, fix the first four layers
		# network.set_trainable_param(list(self.features.parameters())[:8], requires_grad=False)

		self.conv1 = Conv2d(512, 512, 3, same_padding=True)
		self.score_conv = Conv2d(512, self.anchor_num*2, 1, relu=False, same_padding=False)
		self.bbox_conv = Conv2d(512, self.anchor_num*4, 1, relu=False, same_padding=False)
		# self.conv1 = Conv2d(1024, 512, 3, same_padding=True)
		# self.score_conv = Conv2d(512, self.anchor_num*2, 1, relu=False, same_padding=False)
		# self.bbox_conv = Conv2d(512, self.anchor_num*4, 1, relu=False, same_padding=False)

		# loss
		self.cross_entropy = None
		self.loss_box = None

		# self.loss_box_relationship = None

		# initialize the parameters
		self.initialize_parameters()

	def initialize_parameters(self, normal_method='normal'):

		if normal_method == 'normal':
			normal_fun = network.weights_normal_init
		elif normal_method == 'MSRA':
			normal_fun = network.weights_MSRA_init
		else:
			raise (Exception('Cannot recognize the normal method:'.format(normal_method)))

		normal_fun(self.conv1, 0.025)
		normal_fun(self.score_conv, 0.025)
		normal_fun(self.bbox_conv, 0.01)


	@property
	def loss(self):
		return  self.cross_entropy + self.loss_box*0.5


	def forward(self, im_data, im_info, gt_objects=None, dontcare_areas=None):
		im_data = Variable(im_data.cuda())
		features = self.features(im_data)

		rpn_conv1 = self.conv1(features)
		rpn_cls_score = self.score_conv(rpn_conv1)
		rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
		rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
		rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, self.anchor_num*2)
		rpn_bbox_pred = self.bbox_conv(rpn_conv1)

		# proposal layer
		cfg_key = 'TRAIN' if self.training else 'TEST'

		object_rois, scores = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
													   cfg_key, self._feat_stride, self.anchor_scales,
													   self.anchor_ratios, is_relationship=False)
		# if self.training:
		rpn_data = self.anchor_target_layer(rpn_cls_score, gt_objects, dontcare_areas,
												im_info, self.anchor_scales, self.anchor_ratios, self._feat_stride)

		self.cross_entropy, self.loss_box = \
			self.build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)

		return features, object_rois, scores


	def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
		# classification loss
		rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
		rpn_label = rpn_data[0]

		# print rpn_label.size(), rpn_cls_score.size()

		rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze()).cuda()
		rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
		rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

		fg_cnt = torch.sum(rpn_label.data.ne(0))

		_, predict = torch.max(rpn_cls_score.data, 1)
		error = torch.sum(torch.abs(predict-rpn_label.data))
		#  try:
		if predict.size()[0] < 256:
			print predict.size()
			print rpn_label.size()
			print fg_cnt

		rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)
		# print rpn_cross_entropy

		# box loss
		rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
		rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
		rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)

		# print 'Smooth L1 loss: ', F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False)
		# print 'fg_cnt', fg_cnt
		rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False)/(fg_cnt+1e-4)
		# print 'rpn_loss_box', rpn_loss_box
		# print rpn_loss_box

		return rpn_cross_entropy, rpn_loss_box

	@staticmethod
	def reshape_layer(x, d):
		input_shape = x.size()
		# x = x.permute(0, 3, 1, 2)
		# b c w h
		x = x.view(
			input_shape[0],
			int(d),
			int(float(input_shape[1]*input_shape[2])/float(d)),
			input_shape[3]
		)
		# x = x.permute(0, 2, 3, 1)
		return x

	@staticmethod
	def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key,
					   _feat_stride, anchor_scales, anchor_ratios, is_relationship):
		rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
		rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
		x, scores = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
									  cfg_key, _feat_stride, anchor_scales, anchor_ratios,
									  is_relationship=is_relationship)
		x = network.np_to_variable(x, is_cuda=True)
		return x.view(-1, 5), scores

	@staticmethod
	def anchor_target_layer(rpn_cls_score, gt_boxes, dontcare_areas, im_info, anchor_scales,
							anchor_ratios, _feat_stride, is_relationship=False):
		"""
		rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
		gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
		#gt_ishard: (G, 1), 1 or 0 indicates difficult or not
		dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
		im_info: a list of [image_height, image_width, scale_ratios]
		_feat_stride: the downsampling ratio of feature map to the original input image
		anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
		----------
		Returns
		----------
		rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
		rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
						that are the regression objectives
		rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
		rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
		beacuse the numbers of bgs and fgs mays significiantly different
		"""
		rpn_cls_score = rpn_cls_score.data.cpu().numpy()
		rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
			anchor_target_layer_py(rpn_cls_score, gt_boxes, dontcare_areas, im_info, anchor_scales,
								   anchor_ratios, _feat_stride, is_relationship=is_relationship)

		rpn_labels = network.np_to_variable(rpn_labels, is_cuda=True, dtype=torch.LongTensor)
		rpn_bbox_targets = network.np_to_variable(rpn_bbox_targets, is_cuda=True)
		rpn_bbox_inside_weights = network.np_to_variable(rpn_bbox_inside_weights, is_cuda=True)
		rpn_bbox_outside_weights = network.np_to_variable(rpn_bbox_outside_weights, is_cuda=True)

		return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

	def load_from_npz(self, params):
		# params = np.load(npz_file)
		self.features.load_from_npz(params)

		pairs = {'conv1.conv': 'rpn_conv/3x3', 'score_conv.conv': 'rpn_cls_score', 'bbox_conv.conv': 'rpn_bbox_pred'}
		own_dict = self.state_dict()
		for k, v in pairs.items():
			key = '{}.weight'.format(k)
			param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(3, 2, 0, 1)
			own_dict[key].copy_(param)

			key = '{}.bias'.format(k)
			param = torch.from_numpy(params['{}/biases:0'.format(v)])
			own_dict[key].copy_(param)


class FasterRCNN(nn.Module):
	SCALES = (600,)
	MAX_SIZE = 1000
	PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

	def __init__(self, use_kmeans_anchors=False, classes=None, debug=False):
		super(FasterRCNN, self).__init__()

		# self.n_classes = 151
		if classes is not None:
			self.classes = np.asarray(classes)
			self.n_classes = len(classes)

		self.rpn = RPN(use_kmeans_anchors)
		self.roi_pool = RoIPool(7, 7, 1.0/16)
		# self.fc6 = FC(1024 * 7 * 7, 4096)
		self.fc6 = FC(512*7*7, 4096)
		self.fc7 = FC(4096, 4096)
		self.score_fc = FC(4096, self.n_classes, relu=False)
		self.bbox_fc = FC(4096, self.n_classes * 4, relu=False)

		# loss
		self.cross_entropy = None
		self.loss_box = None

		# for log
		self.debug = debug

	@property
	def loss(self):
		# print self.cross_entropy
		# print self.loss_box
		# print self.rpn.cross_entropy
		# print self.rpn.loss_box
		return self.cross_entropy + self.loss_box*0.5

	def forward(self, im_data, im_info, gt_boxes):
		features, rois, scores = self.rpn(im_data, im_info, gt_boxes)

		if self.training:
			roi_data = self.proposal_target_layer(rois, gt_boxes, self.n_classes, dontcare_areas=None)
			rois = roi_data[0]
		# roi pool
		pooled_features = self.roi_pool(features, rois)
		x = pooled_features.view(pooled_features.size()[0], -1)
		x = self.fc6(x)
		x = F.dropout(x, training=self.training)
		x = self.fc7(x)
		x = F.dropout(x, training=self.training)

		cls_score = self.score_fc(x)
		cls_prob = F.softmax(cls_score)
		bbox_pred = self.bbox_fc(x)

		if self.training:
			self.cross_entropy, self.loss_box = self.build_loss(cls_score, bbox_pred, roi_data)

		return cls_prob, bbox_pred, rois

	def build_loss(self, cls_score, bbox_pred, roi_data):
		# classification loss
		label = roi_data[1].squeeze()
		fg_cnt = torch.sum(label.data.ne(0))
		bg_cnt = label.data.numel() - fg_cnt

		# for log
		maxv, predict = cls_score.data.max(1)
		self.tp = torch.sum(predict[:fg_cnt].eq(label.data[:fg_cnt])) if fg_cnt > 0 else 0
		self.tf = torch.sum(predict[fg_cnt:].eq(label.data[fg_cnt:]))
		self.fg_cnt = fg_cnt
		self.bg_cnt = bg_cnt

		ce_weights = torch.ones(cls_score.size()[1])
		ce_weights[0] = float(fg_cnt) / bg_cnt
		ce_weights = ce_weights.cuda()
		cross_entropy = F.cross_entropy(cls_score, label, weight=ce_weights)

		# bounding box regression L1 loss
		bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[2:]
		bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
		bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)

		loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-4)

		return cross_entropy, loss_box

	@staticmethod
	def proposal_target_layer(rpn_rois, gt_boxes, num_classes, dontcare_areas):
		"""
		----------
		rpn_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
		gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
		# gt_ishard: (G, 1) {0 | 1} 1 indicates hard
		dontcare_areas: (D, 4) [ x1, y1, x2, y2]
		num_classes
		----------
		Returns
		----------
		rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
		labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
		bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
		bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
		bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
		"""
		rpn_rois = rpn_rois.data.cpu().numpy()
		rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
			proposal_target_layer_py(rpn_rois, gt_boxes, dontcare_areas, num_classes)
		# print labels.shape, bbox_targets.shape, bbox_inside_weights.shape
		rois = network.np_to_variable(rois, is_cuda=True)
		labels = network.np_to_variable(labels, is_cuda=True, dtype=torch.LongTensor)
		bbox_targets = network.np_to_variable(bbox_targets, is_cuda=True)
		bbox_inside_weights = network.np_to_variable(bbox_inside_weights, is_cuda=True)
		bbox_outside_weights = network.np_to_variable(bbox_outside_weights, is_cuda=True)

		return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

	def interpret_faster_rcnn(self, cls_prob, bbox_pred, rois, im_info, im_shape, nms=True, clip=True, min_score=0.0):
		# find class
		scores, inds = cls_prob.data.max(1)
		scores, inds = scores.cpu().numpy(), inds.cpu().numpy()

		keep = np.where((inds > 0) & (scores >= min_score))
		scores, inds = scores[keep], inds[keep]

		# Apply bounding-box regression deltas
		keep = keep[0]
		box_deltas = bbox_pred.data.cpu().numpy()[keep]
		box_deltas = np.asarray([
			box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
		], dtype=np.float)
		boxes = rois.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
		pred_boxes = bbox_transform_inv(boxes, box_deltas)
		if clip:
			pred_boxes = clip_boxes(pred_boxes, im_shape)

		# nms
		if nms and pred_boxes.shape[0] > 0:
			pred_boxes, scores, inds = nms_detections(pred_boxes, scores, 0.3, inds=inds)

		return pred_boxes, scores, inds, self.classes[inds]

	def detect(self, image, thr=0.3):
		im_data, im_scales = self.get_image_blob(image)
		im_info = np.array(
			[[im_data.shape[1], im_data.shape[2], im_scales[0]]],
			dtype=np.float32)

		cls_prob, bbox_pred, rois = self(im_data, im_info)
		pred_boxes, scores, classes = \
			self.interpret_faster_rcnn(cls_prob, bbox_pred, rois, im_info, image.shape, min_score=thr)
		return pred_boxes, scores, classes

	def get_image_blob_noscale(self, im):
		processed_ims = [im]
		im_scale_factors = [1.0]

		blob = im_list_to_blob(processed_ims)

		return blob, np.array(im_scale_factors)

	def get_image_blob(self, im):
		"""Converts an image into a network input.
		Arguments:
			im (ndarray): a color image in BGR order
		Returns:
			blob (ndarray): a data blob holding an image pyramid
			im_scale_factors (list): list of image scales (relative to im) used
				in the image pyramid
		"""
		im_orig = im.astype(np.float32, copy=True)

		im_shape = im_orig.shape
		im_size_min = np.min(im_shape[0:2])
		im_size_max = np.max(im_shape[0:2])

		processed_ims = []
		im_scale_factors = []

		for target_size in self.SCALES:
			im_scale = float(target_size) / float(im_size_min)
			# Prevent the biggest axis from being more than MAX_SIZE
			if np.round(im_scale * im_size_max) > self.MAX_SIZE:
				im_scale = float(self.MAX_SIZE) / float(im_size_max)
			im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
							interpolation=cv2.INTER_LINEAR)
			im_scale_factors.append(im_scale)
			processed_ims.append(im)

		# Create a blob to hold the input images
		blob = im_list_to_blob(processed_ims)

		return blob, np.array(im_scale_factors)

	# def load_from_npz(self, params):
	#     self.rpn.load_from_npz(params)
	#
	#     pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 'score_fc.fc': 'cls_score', 'bbox_fc.fc': 'bbox_pred'}
	#     own_dict = self.state_dict()
	#     for k, v in pairs.items():
	#         key = '{}.weight'.format(k)
	#         param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(1, 0)
	#         own_dict[key].copy_(param)
	#
	#         key = '{}.bias'.format(k)
	#         param = torch.from_numpy(params['{}/biases:0'.format(v)])
	#         own_dict[key].copy_(param)


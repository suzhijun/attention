import cv2
import numpy as np
import numpy.random as npr
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils.timer import Timer
from utils.HDN_utils import check_relationship_recall
from fast_rcnn.nms_wrapper import nms
from rpn_msr.proposal_target_layer_pair import proposal_target_layer as proposal_target_layer_py
from fast_rcnn.bbox_transform import bbox_transform_inv_hdn, clip_boxes
from fast_rcnn.hierarchical_message_passing_structure import Hierarchical_Message_Passing_Structure
from RPN import RPN
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps, bbox_intersections
from utils.make_cover import compare_rel_rois

import network
from network import Conv2d, FC
# from roi_pooling.modules.roi_pool_py import RoIPool
from roi_pooling.modules.roi_pool import RoIPool
from MSDN_base import HDN_base


DEBUG = False
TIME_IT = cfg.TIME_IT


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
	dets = np.hstack((pred_boxes,
					  scores[:, np.newaxis])).astype(np.float32)
	keep = nms(dets, nms_thresh)
	if inds is None:
		return pred_boxes[keep], scores[keep], keep
	return pred_boxes[keep], scores[keep], inds[keep], keep

class Hierarchical_Descriptive_Model(HDN_base):
	def __init__(self, nhidden, n_object_cats, n_predicate_cats, MPS_iter, object_loss_weight,
				 predicate_loss_weight,
				 dropout=False,
				 use_kmeans_anchors=True,
				 gate_width=128,
				 use_region_reg=False,
				 use_kernel=False):

		super(Hierarchical_Descriptive_Model, self).__init__(nhidden, n_object_cats, n_predicate_cats,  MPS_iter, object_loss_weight,
				 predicate_loss_weight, dropout, use_region_reg)

		self.rpn = RPN(use_kmeans_anchors)
		self.roi_pool_object = RoIPool(7, 7, 1.0/16)
		self.roi_pool_phrase = RoIPool(7, 7, 1.0/16)
		self.fc6_obj = FC(512 * 7 * 7, nhidden, relu=True)
		self.fc7_obj = FC(nhidden, nhidden, relu=False)
		self.fc6_phrase = FC(512 * 7 * 7, nhidden, relu=True)
		self.fc7_phrase = FC(nhidden, nhidden, relu=False)
		if MPS_iter == 0:
			self.mps = None
		else:
			self.mps = Hierarchical_Message_Passing_Structure(nhidden, dropout,
							gate_width=gate_width, use_kernel_function=use_kernel) # the hierarchical message passing structure
			network.weights_normal_init(self.mps, 0.01)

		self.score_obj = FC(nhidden, self.n_classes_obj, relu=False)
		self.bbox_obj = FC(nhidden, self.n_classes_obj * 4, relu=False)
		self.score_pred = FC(nhidden, self.n_classes_pred, relu=False)
		self.bbox_pred = FC(nhidden, self.n_classes_pred * 4, relu=False)


		network.weights_normal_init(self.score_obj, 0.01)
		network.weights_normal_init(self.bbox_obj, 0.005)
		network.weights_normal_init(self.score_pred, 0.01)
		network.weights_normal_init(self.bbox_pred, 0.005)


	def forward(self, im_data, im_info, gt_objects=None, gt_relationships=None, gt_regions=None, graph_generation=False):

		self.timer.tic()
		features, object_rois, region_rois, scores_object, scores_relationship = \
			self.rpn(im_data, im_info, gt_objects.numpy(), gt_regions.numpy())

		if not self.training and gt_objects is not None:
			zeros = torch.zeros(gt_objects.shape[0], 1)
			object_rois_gt = torch.cat((zeros, gt_objects[:, :4]), dim=1)
			object_rois_gt = Variable(object_rois_gt.cuda())
			object_rois[:object_rois_gt.size(0)] = object_rois_gt

		if not self.training and gt_regions is not None:
			zeros = torch.zeros(gt_regions.shape[0], 1)
			region_rois = torch((zeros, gt_regions[:, :4]), dim=1)
			region_rois = Variable(region_rois.cuda())

		if TIME_IT:
			torch.cuda.synchronize()
			print '\t[RPN]: %.3fs'%self.timer.toc(average=False)

		# if self.training:
		# 	# ToDo: set train object number or use all rpn proposals
		# 	subject_inds, object_inds, phrase_rois = compare_rel_rois(
		# 		object_rois, region_rois, scores_object, scores_relationship,
		# 		topN_obj=object_rois.size()[0], topN_rel=region_rois.size()[0],
		# 		obj_rel_thresh=cfg.TRAIN.MPN_OBJ_REL_THRESH,
		# 		max_objects=cfg.TRAIN.MPN_MAX_OBJECTS, topN_covers=cfg.TRAIN.MPN_COVER_NUM,
		# 		cover_thresh=cfg.TRAIN.MPN_MAKE_COVER_THRESH)
		# else:
		# 	object_rois_num = min(cfg.TEST.MPN_BBOX_NUM, object_rois.size()[0])
		# 	region_rois_num = min(cfg.TEST.MPN_REGION_NUM, region_rois.size()[0])
		# 	object_rois = object_rois[:object_rois_num, :]
		# 	region_rois = region_rois[:region_rois_num, :]
		# 	subject_inds, object_inds, phrase_rois = compare_rel_rois(
		# 		object_rois, region_rois, scores_object[:object_rois_num], scores_relationship[:region_rois_num],
		# 		topN_obj=cfg.TEST.MPN_BBOX_NUM, topN_rel=cfg.TEST.MPN_REGION_NUM,
		# 		obj_rel_thresh=cfg.TEST.MPN_OBJ_REL_THRESH,
		# 		max_objects=cfg.TEST.MPN_MAX_OBJECTS, topN_covers=cfg.TEST.MPN_COVER_NUM,
		# 		cover_thresh=cfg.TEST.MPN_MAKE_COVER_THRESH)

		self.timer.tic()
		roi_data_object, roi_data_predicate, mat_object, mat_phrase = \
			self.proposal_target_layer(object_rois, region_rois, scores_object, scores_relationship,
									   gt_objects, gt_relationships, gt_regions, self.n_classes_obj,
									   self.n_classes_pred, self.training, graph_generation=graph_generation)

		if TIME_IT:
			torch.cuda.synchronize()
			print '\t[Proposal]: %.3fs'%self.timer.toc(average=False)


		self.timer.tic()
		object_rois = roi_data_object[0]
		phrase_rois = roi_data_predicate[0]

		# roi pool
		pooled_object_features = self.roi_pool_object(features, object_rois)
		pooled_object_features = pooled_object_features.view(pooled_object_features.size()[0], -1)
		pooled_object_features = self.fc6_obj(pooled_object_features)

		if self.dropout:
			pooled_object_features = F.dropout(pooled_object_features, training = self.training)

		pooled_object_features = self.fc7_obj(pooled_object_features)
		if self.dropout:
			pooled_object_features = F.dropout(pooled_object_features, training = self.training)

		pooled_phrase_features = self.roi_pool_phrase(features, phrase_rois)
		pooled_phrase_features = pooled_phrase_features.view(pooled_phrase_features.size()[0], -1)
		pooled_phrase_features = self.fc6_phrase(pooled_phrase_features)

		if self.dropout:
			pooled_phrase_features = F.dropout(pooled_phrase_features, training = self.training)

		pooled_phrase_features = self.fc7_phrase(pooled_phrase_features)

		if self.dropout:
			pooled_phrase_features = F.dropout(pooled_phrase_features, training = self.training)

		# bounding box regression before message passing
		bbox_object = self.bbox_obj(F.relu(pooled_object_features))
		bbox_phrase = self.bbox_pred(F.relu(pooled_phrase_features))

		if TIME_IT:
			torch.cuda.synchronize()
			print '\t[Pre-MPS]: %.3fs'%self.timer.toc(average=False)


		# hierarchical message passing structure
		self.timer.tic()
		if self.MPS_iter < 0:
			if self.training:
				self.MPS_iter = npr.choice(self.MPS_iter_range)
			else:
				self.MPS_iter = cfg.TEST.MPS_ITER_NUM

		for i in xrange(self.MPS_iter):
			pooled_object_features, pooled_phrase_features = \
				self.mps(pooled_object_features, pooled_phrase_features, mat_object)

		if TIME_IT:
			torch.cuda.synchronize()
			print '\t[Massage Passing]:  %.3fs'%self.timer.toc(average=False)


		self.timer.tic()
		pooled_object_features = F.relu(pooled_object_features)
		pooled_phrase_features = F.relu(pooled_phrase_features)

		cls_score_object = self.score_obj(pooled_object_features)
		cls_prob_object = F.softmax(cls_score_object)

		cls_score_predicate = self.score_pred(pooled_phrase_features)
		cls_prob_predicate = F.softmax(cls_score_predicate)

		if TIME_IT:
			torch.cuda.synchronize()
			print '\t[Post-MPS]: %.3fs'%self.timer.toc(average=False)


		self.timer.tic()
		if self.training:
			self.cross_entropy_object, self.loss_obj_box, self.tp, self.tf, self.fg_cnt, self.bg_cnt = \
				self.build_loss(cls_score_object, bbox_object, roi_data_object, obj=True)
			self.cross_entropy_predicate, self.loss_pred_box, self.tp_pred, self.tf_pred, \
			self.fg_cnt_pred, self.bg_cnt_pred = \
				self.build_loss(cls_score_predicate, bbox_phrase, roi_data_predicate)

		if TIME_IT:
			torch.cuda.synchronize()
			print '\t[Loss]:  %.3fs'%self.timer.toc(average=False)

		return (cls_prob_object, bbox_object, object_rois, scores_object), \
				(cls_prob_predicate, bbox_phrase, phrase_rois, mat_phrase)

	@staticmethod
	def proposal_target_layer(object_rois, region_rois, scores_object, scores_relationship,
							  gt_objects, gt_relationships, gt_box_relationship, n_classes_obj, n_classes_pred,
							  is_training=False, graph_generation=False):

		"""
		----------
		object_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2], Variable, cuda
		phrase_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2], Variable, cuda
		gt_objects:   (G_obj, 5) [x1 ,y1 ,x2, y2, obj_class] int, tensor
		gt_relationships: (G_obj, G_obj) [pred_class] int (-1 for no relationship), tensor
		gt_regions:   (G_region, 4+40) [x1, y1, x2, y2, word_index] (-1 for padding), tensor
		# gt_ishard: (G_region, 4+40) {0 | 1} 1 indicates hard
		# dontcare_areas: (D, 4) [ x1, y1, x2, y2]
		n_classes_obj
		n_classes_pred
		is_training to indicate whether in training scheme
		----------
		Returns
		----------
		rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
		labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
		bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
		bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
		bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
		"""

		object_labels, object_rois, bbox_targets_object, bbox_inside_weights_object, bbox_outside_weights_object, \
		phrase_labels, phrase_rois, bbox_targets_phrase, bbox_inside_weights_phrase, bbox_outside_weights_phrase, \
		mat_object, mat_phrase = \
			proposal_target_layer_py(object_rois, region_rois, scores_object, scores_relationship,
									 gt_objects, gt_relationships, gt_box_relationship, n_classes_obj, n_classes_pred,
									 is_training, graph_generation=graph_generation)

		# print labels.shape, bbox_targets.shape, bbox_inside_weights.shape
		if is_training:
			object_labels = network.np_to_variable(object_labels, is_cuda=True, dtype=torch.LongTensor)
			bbox_targets_object = network.np_to_variable(bbox_targets_object, is_cuda=True)
			bbox_inside_weights_object = network.np_to_variable(bbox_inside_weights_object, is_cuda=True)
			bbox_outside_weights_object = network.np_to_variable(bbox_outside_weights_object, is_cuda=True)
			phrase_labels = network.np_to_variable(phrase_labels, is_cuda=True, dtype=torch.LongTensor)
			bbox_targets_phrase = network.np_to_variable(bbox_targets_phrase, is_cuda=True)
			bbox_inside_weights_phrase = network.np_to_variable(bbox_inside_weights_phrase, is_cuda=True)
			bbox_outside_weights_phrase = network.np_to_variable(bbox_outside_weights_phrase, is_cuda=True)

		object_rois = network.np_to_variable(object_rois, is_cuda=True)
		phrase_rois = network.np_to_variable(phrase_rois, is_cuda=True)

		return (object_rois, object_labels, bbox_targets_object, bbox_inside_weights_object,
				bbox_outside_weights_object), \
			   (phrase_rois, phrase_labels, bbox_targets_phrase, bbox_inside_weights_phrase,
				bbox_outside_weights_phrase), \
			   mat_object, mat_phrase


	def proposal_target_layer_test(self, object_rois, relationship_rois,
								   scores_object, scores_relationship, graph_generation=False):

		# object_rois = object_rois.data.cpu().numpy()
		# relationship_rois = relationship_rois.data.cpu().numpy()

		object_rois, pair_rois, relationship_rois, mat_object, mat_phrase, mat_region = \
			self.setup_pair(object_rois, relationship_rois, scores_object, scores_relationship, graph_generation=graph_generation)
		object_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, pair_labels, \
		bbox_targets_region, bbox_inside_weights_region, bbox_outside_weights_region = [None] * 8

		# object_rois = network.np_to_variable(object_rois, is_cuda=True)
		pair_rois = network.np_to_variable(pair_rois, is_cuda=True)
		# relationship_rois = network.np_to_variable(relationship_rois, is_cuda=True)

		return (object_rois, object_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights), \
			   (pair_rois, pair_labels), \
			   (relationship_rois, bbox_targets_region, bbox_inside_weights_region, bbox_outside_weights_region), \
			   mat_object, mat_phrase, mat_region


	def setup_pair(self, object_rois, relationship_rois, scores_object, scores_relationship,
					graph_generation=False):
		# overlaps: (rois x gt_boxes)
		roi_num = min(cfg.TEST.MPN_BBOX_NUM, object_rois.size(0))
		rois = object_rois[:roi_num, :]
		relationship_num = min(cfg.TEST.MPN_REGION_NUM, relationship_rois.size(0))
		relationship_rois = relationship_rois[:relationship_num, :]

		subject_inds, object_inds, pair_rois = compare_rel_rois(
			rois, relationship_rois, scores_object[:roi_num], scores_relationship[:relationship_num],
			topN_obj=cfg.TEST.MPN_BBOX_NUM, topN_rel=cfg.TEST.MPN_REGION_NUM, obj_rel_thresh=cfg.TEST.MPN_OBJ_REL_THRESH,
			max_objects=cfg.TEST.MPN_MAX_OBJECTS, topN_covers=cfg.TEST.MPN_COVER_NUM, cover_thresh=cfg.TEST.MPN_MAKE_COVER_THRESH)

		subject_inds, object_inds, pair_rois = subject_inds.cpu().numpy(), object_inds.cpu().numpy(), pair_rois.data.cpu().numpy()

		subject_inds, object_inds = np.append(subject_inds, object_inds), np.append(object_inds, subject_inds)
		pair_rois = np.vstack((pair_rois, pair_rois))

		mat_phrase = np.zeros((subject_inds.size, 2), dtype=np.int64)
		mat_phrase[:, 0] = subject_inds
		mat_phrase[:, 1] = object_inds

		mat_object = np.zeros((roi_num, 2, pair_rois.shape[0]), dtype=np.int64)
		for i in range(pair_rois.shape[0]):
			mat_object[subject_inds[i], 0, i] = 1
			mat_object[object_inds[i], 1, i] = 1


		overlaps_phrase = bbox_intersections(
			np.ascontiguousarray(relationship_rois.data.cpu().numpy()[:, 1:5], dtype=np.float),
			np.ascontiguousarray(pair_rois[:, 1:5], dtype=np.float))

		max_overlaps_phrase = overlaps_phrase.max(axis=1)

		if graph_generation:
			keep_inds = np.where(max_overlaps_phrase >= cfg.TEST.MPN_PHRASE_REGION_OVERLAP_THRESH)[0]
		else:
			keep_inds = range(relationship_rois.size(0))

		relationship_rois = relationship_rois[keep_inds, :]

		mat_region = (overlaps_phrase[keep_inds, :] > cfg.TEST.MPN_PHRASE_REGION_OVERLAP_THRESH).astype(np.int64)
		mat_phrase = np.concatenate((mat_phrase, mat_region.transpose()), 1)

		return rois, pair_rois, relationship_rois, mat_object, mat_phrase, mat_region


	def interpret_HDN(self, cls_prob, bbox_pred, rois, cls_prob_predicate,
						mat_phrase, rpn_scores_object, im_info, nms=True, clip=True, min_score=0.0,
						top_N=100, use_gt_boxes=False, use_rpn_scores=False):
		scores, inds = cls_prob[:, 1:].data.max(1)
		inds += 1
		scores, inds = scores.cpu().numpy(), inds.cpu().numpy()
		predicate_scores, predicate_inds = cls_prob_predicate[:, 1:].data.max(1)
		predicate_inds += 1
		predicate_scores, predicate_inds = predicate_scores.cpu().numpy(), predicate_inds.cpu().numpy()

		keep = np.where((inds > 0) & (scores >= min_score))
		scores, inds = scores[keep], inds[keep]

		# Apply bounding-box regression deltas
		keep = keep[0]
		box_deltas = bbox_pred.data.cpu().numpy()[keep]
		box_deltas = np.asarray([
			box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
		], dtype=np.float)
		boxes = rois.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
		if use_gt_boxes:
			nms = False
			clip = False
			pred_boxes = boxes
		else:
			pred_boxes = bbox_transform_inv_hdn(boxes, box_deltas)

		if clip:
			pred_boxes = clip_boxes(pred_boxes, im_info[0][:2] / im_info[0][2])

		# nms
		if nms and pred_boxes.shape[0] > 0:
			pred_boxes, scores, inds, keep_keep = nms_detections(pred_boxes, scores, 0.60, inds=inds)
			keep = keep[keep_keep]


		sub_list = np.array([], dtype=int)
		obj_list = np.array([], dtype=int)
		pred_list = np.array([], dtype=int)

		# print 'keep', keep
		# print 'mat_phrase', mat_phrase


		for i in range(mat_phrase.shape[0]):
			sub_id = np.where(keep == mat_phrase[i, 0])[0]
			obj_id = np.where(keep == mat_phrase[i, 1])[0]
			if len(sub_id) > 0 and len(obj_id) > 0:
				sub_list = np.append(sub_list, sub_id[0])
				obj_list = np.append(obj_list, obj_id[0])
				pred_list = np.append(pred_list, i)

		if use_rpn_scores:
			total_scores = predicate_scores.squeeze()[pred_list] \
							* scores[sub_list].squeeze() * scores[obj_list].squeeze() \
						   * rpn_scores_object[sub_list].squeeze() * rpn_scores_object[obj_list].squeeze()
		else:
			total_scores = predicate_scores.squeeze()[pred_list] \
						   * scores[sub_list].squeeze() * scores[obj_list].squeeze()

		top_N_list = total_scores.argsort()[::-1][:top_N]
		predicate_inds = predicate_inds.squeeze()[pred_list[top_N_list]]

		subject_inds = inds[sub_list[top_N_list]]
		object_inds = inds[obj_list[top_N_list]]
		subject_boxes = pred_boxes[sub_list[top_N_list]]
		object_boxes = pred_boxes[obj_list[top_N_list]]

		return pred_boxes, scores, inds, subject_inds, object_inds, subject_boxes, object_boxes, predicate_inds


	def interpret_RMRPN(self, cls_prob_object, bbox_pred_object, rois_object,
						cls_prob_predicate, bbox_pred_predicate, rois_predicate,
						mat_phrase, rpn_scores_object, im_info, nms=True, clip=True, min_score=0.0,
						top_N=100, use_gt_boxes=False, use_rpn_scores=False):
		scores, inds = cls_prob_object[:, 1:].data.max(1)
		inds += 1
		scores, inds = scores.cpu().numpy(), inds.cpu().numpy()
		predicate_scores, predicate_inds = cls_prob_predicate[:, 1:].data.max(1)
		predicate_inds += 1
		predicate_scores, predicate_inds = predicate_scores.cpu().numpy(), predicate_inds.cpu().numpy()

		keep = np.where((inds > 0) & (scores >= min_score))
		scores, inds = scores[keep], inds[keep]

		# Apply bounding-box regression deltas
		keep = keep[0]
		box_deltas = bbox_pred_object.data.cpu().numpy()[keep]
		box_deltas = np.asarray([
			box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
		], dtype=np.float)
		boxes = rois_object.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
		if use_gt_boxes:
			nms = False
			clip = False
			pred_boxes = boxes
		else:
			pred_boxes = bbox_transform_inv_hdn(boxes, box_deltas)

		if clip:
			pred_boxes = clip_boxes(pred_boxes, im_info[0][:2] / im_info[0][2])

		# nms
		if nms and pred_boxes.shape[0] > 0:
			pred_boxes, scores, inds, keep_keep = nms_detections(pred_boxes, scores, 0.60, inds=inds)
			keep = keep[keep_keep]

		sub_list = np.array([], dtype=int)
		obj_list = np.array([], dtype=int)
		pred_list = np.array([], dtype=int)

		# print 'keep', keep
		# print 'mat_phrase', mat_phrase
		# keep predicate(phrase) whose sub & obj kept
		for i in range(mat_phrase.shape[0]):
			sub_id = np.where(keep == mat_phrase[i, 0])[0]
			obj_id = np.where(keep == mat_phrase[i, 1])[0]
			if len(sub_id) > 0 and len(obj_id) > 0:
				sub_list = np.append(sub_list, sub_id[0])
				obj_list = np.append(obj_list, obj_id[0])
				pred_list = np.append(pred_list, i)

		# keep_inds = np.full(inds.size, -1)
		# keep_inds[keep] = np.arange(keep.size)
		# sub_list_n = keep_inds[mat_phrase[i, 0]]
		# obj_list_n = keep_inds[mat_phrase[i, 1]]
		# keep_pred_inds = np.where((sub_list_n > -1) & (obj_list_n > -1))[0]
		# sub_list_n = sub_list[keep_pred_inds]
		# obj_list_n = obj_list[keep_pred_inds]

		predicate_scores = predicate_scores[pred_list]
		predicate_inds = predicate_inds[pred_list]
		box_deltas_predicate = bbox_pred_predicate.data.cpu().numpy()[pred_list]

		box_deltas_predicate = np.asarray([
			box_deltas_predicate[i, (predicate_inds[i] * 4) : (predicate_inds[i] * 4 + 4)] for i in range(len(predicate_inds))
		], dtype=np.float)
		boxes_predicate = rois_predicate.data.cpu().numpy()[pred_list, 1:5] / im_info[0][2]
		pred_boxes_predicate = bbox_transform_inv_hdn(boxes_predicate, box_deltas_predicate)

		if nms and pred_boxes_predicate.shape[0] > 0:
			pred_boxes_predicate, predicate_scores, predicate_inds, keep_pred_list = \
				nms_detections(pred_boxes_predicate, predicate_scores, 0.60, inds=predicate_inds)
			sub_list = sub_list[keep_pred_list]
			obj_list = obj_list[keep_pred_list]

		if use_rpn_scores:
			total_scores = predicate_scores.squeeze() \
							* scores[sub_list].squeeze() * scores[obj_list].squeeze() \
						   * rpn_scores_object[sub_list].squeeze() * rpn_scores_object[obj_list].squeeze()
		else:
			total_scores = predicate_scores.squeeze() \
						   * scores[sub_list].squeeze() * scores[obj_list].squeeze()
		# keep top N phrase
		top_N_list = total_scores.argsort()[::-1][:top_N]
		predicate_inds = predicate_inds[top_N_list]
		pred_boxes_predicate = pred_boxes_predicate[top_N_list]

		subject_inds = inds[sub_list[top_N_list]]
		object_inds = inds[obj_list[top_N_list]]
		subject_boxes = pred_boxes[sub_list[top_N_list]]
		object_boxes = pred_boxes[obj_list[top_N_list]]

		return pred_boxes, scores, inds, subject_inds, object_inds, subject_boxes, object_boxes, \
			   predicate_inds, pred_boxes_predicate

	#
	# def describe(self, im_path, top_N=10):
	# 		image = cv2.imread(im_path)
	# 		# print 'image.shape', image.shape
	# 		im_data, im_scales = self.get_image_blob_noscale(image)
	# 		# print 'im_data.shape', im_data.shape
	# 		# print 'im_scales', im_scales
	#
	# 		im_info = np.array(
	# 			[[im_data.shape[1], im_data.shape[2], im_scales[0]]],
	# 			dtype=np.float32)
	#
	# 		object_result, predicate_result, region_result = self(im_data, im_info)
	#
	# 		object_boxes, object_scores, object_inds, sub_assignment, obj_assignment, predicate_inds, region_assignment\
	# 				 = self.interpret_result(object_result[0], object_result[1], object_result[2], \
	# 					predicate_result[0], predicate_result[1], \
	# 					im_info, image.shape)
	#
	# 		region_caption, bbox_pred, region_rois, logprobs = region_result[:]
	# 		boxes = region_rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]
	# 		box_deltas = bbox_pred.data.cpu().numpy()
	# 		pred_boxes = bbox_transform_inv_hdn(boxes, box_deltas)
	# 		pred_boxes = clip_boxes(pred_boxes, image.shape)
	#
	# 		# print 'im_scales[0]', im_scales[0]
	# 		return (region_caption.numpy(), logprobs.numpy(), pred_boxes, \
	# 				object_boxes, object_inds, object_scores, \
	# 			sub_assignment, obj_assignment, predicate_inds, region_assignment)


	def evaluate(self, im_data, im_info, gt_objects, gt_relationships, gt_regions,
				 thr=0.5, nms=False, top_Ns = [100], use_gt_boxes=False, use_gt_regions=False, only_predicate=False,
				 use_rpn_scores=False, use_predicate_boxes=True):

		if use_gt_boxes:
			gt_boxes_object = gt_objects[:, :4] * im_info[2]
		else:
			gt_boxes_object = None


		if use_gt_regions:
			gt_boxes_regions = gt_regions[:, :4] * im_info[2]
		else:
			gt_boxes_regions = None

		object_result, predicate_result = \
			self(im_data, im_info, gt_boxes_object, gt_relationships=None, gt_regions=gt_boxes_regions, graph_generation=True)

		cls_prob_object, bbox_object, object_rois, rpn_scores_object = object_result
		cls_prob_predicate, bbox_predicate, predicate_rois, mat_phrase = predicate_result

		# interpret the model output
		obj_boxes, obj_scores, obj_inds, subject_inds, object_inds, \
			subject_boxes, object_boxes, predicate_inds, predicate_boxes = \
				self.interpret_RMRPN(cls_prob_object, bbox_object, object_rois,
									 cls_prob_predicate, bbox_predicate, predicate_rois,
									 mat_phrase, rpn_scores_object, im_info,
									 nms=nms, top_N=max(top_Ns), use_gt_boxes=use_gt_boxes, use_rpn_scores=use_rpn_scores)

		gt_objects[:, :4] /= im_info[0][2]
		rel_cnt, rel_correct_cnt = check_relationship_recall(gt_objects.numpy(), gt_relationships.numpy(), gt_regions.numpy(),
										subject_inds, object_inds, predicate_inds,
										subject_boxes, object_boxes, predicate_boxes, top_Ns, thres=thr,
										only_predicate=only_predicate, use_predicate_boxes=use_predicate_boxes)

		return rel_cnt, rel_correct_cnt



	def build_loss_objectiveness(self, region_objectiveness, targets):
		loss_objectiveness = F.cross_entropy(region_objectiveness, targets)
		maxv, predict = region_objectiveness.data.max(1)
		labels = targets.squeeze()
		fg_cnt = torch.sum(labels.data.ne(0))
		bg_cnt = labels.data.numel() - fg_cnt
		if fg_cnt > 0:
			self.tp_reg = torch.sum(predict[:fg_cnt].eq(labels.data[:fg_cnt]))
		else:
			self.tp_reg = 0.
		if bg_cnt > 0:
			self.tf_reg = torch.sum(predict[fg_cnt:].eq(labels.data[fg_cnt:]))
		else:
			self.tf_reg = 0.
		self.fg_cnt_reg = fg_cnt
		self.bg_cnt_reg = bg_cnt
		return loss_objectiveness

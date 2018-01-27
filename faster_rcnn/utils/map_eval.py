import numpy as np
import torch

from faster_rcnn.faster_rcnn import nms_detections
from faster_rcnn.utils.cython_bbox import bbox_overlaps
from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes


def im_detect(net, im_data, im_info):
	features, pooled_features, cls_score, cls_prob, bbox_pred, rois, score  = net(im_data, im_info, gt_boxes=None)

	scores = cls_prob.data.cpu().numpy()
	boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]
	# Apply bounding-box regression deltas
	box_deltas = bbox_pred.data.cpu().numpy()
	pred_boxes = bbox_transform_inv(boxes, box_deltas)
	pred_boxes = clip_boxes(pred_boxes, im_info[0][:2] / im_info[0][2])

	return scores, pred_boxes


def cls_ap(rec, prec):
	# correct AP calculation
	# first append sentinel values at the end
	mrec = np.concatenate(([0.], rec, [1.]))
	mpre = np.concatenate(([0.], prec, [0.]))

	# compute the precision envelope
	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

	# to calculate area under PR curve, look for points
	# where X axis (recall) changes value
	i = np.where(mrec[1:] != mrec[:-1])[0]

	# and sum (\Delta recall) * prec
	ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
	return ap


def cls_eval(cls_scores, cls_tp, cls_gt_num):
	'''
	:param cls_scores: unsorted class scores of all images
	:param cls_tp: unsorted class tp of all images, [1, 1, 0, 1, ...]
	:param cls_gt_num: int
	:return: cls_ap
	'''
	sorted_cls_inds = np.argsort(-cls_scores)
	cls_tp = cls_tp[sorted_cls_inds]
	cls_fp = 1 - cls_tp

	tp = np.cumsum(cls_tp)
	fp = np.cumsum(cls_fp)
	recall = tp / float(cls_gt_num)
	precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
	ap = cls_ap(recall, precision)

	return ap


def image_cls_eval(scores, boxes, gt_boxes, object_class,
				   score_thresh=0.05, overlap_thresh=0.5, nms_thresh=0.6):
	'''
	scores/boxes of some class of one image
	keep these that satisfy score_thresh and overlaps_thresh
	and get tp [1, 1, 0, 1, ...]
	:param scores:
	:param boxes:
	:param gt_boxes:
	:param object_class: int
	:param score_thresh:
	:param overlap_thresh:
	:param nms_thresh:
	:return:
	'''
	inds = np.where(scores[:] > score_thresh)[0]
	cls_scores = scores[inds]
	cls_boxes = boxes[inds]
	cls_boxes, cls_scores = nms_detections(cls_boxes, cls_scores, nms_thresh)

	# get gt_boxes of this class
	cls_gt_boxes = gt_boxes[gt_boxes[:, 4] == object_class]

	if cls_gt_boxes.shape[0] == 0 or cls_scores.size == 0:
		cls_tp = np.zeros(cls_scores.size)
		return cls_scores, cls_tp, cls_gt_boxes.shape[0]

	cls_overlaps = bbox_overlaps(
		np.ascontiguousarray(cls_boxes, dtype=np.float),
		np.ascontiguousarray(cls_gt_boxes[:, :4], dtype=np.float)
	)

	cls_max_overlap = cls_overlaps.max(axis=1)
	cls_assignment = cls_overlaps.argmax(axis=1)
	cls_assignment[cls_max_overlap < overlap_thresh] = -1
	# keep first rois that assigned to a gt box
	_, cls_assignment_keep_inds = np.unique(np.append(np.array([-1]), cls_assignment), return_index=True)
	cls_assignment_keep_inds = cls_assignment_keep_inds[1:] - 1

	cls_tp = np.zeros(cls_scores.size)
	cls_tp[cls_assignment_keep_inds] = 1

	return cls_scores, cls_tp, cls_gt_boxes.shape[0]


def image_eval(target_net, im_data, im_info, gt_boxes, object_classes,  max_per_image=300,
			   score_thresh=0.05, overlap_thresh=0.5, nms_thresh=0.6):

	scores, boxes = im_detect(target_net, im_data, im_info)
	classes_scores = []  # length = 150
	classes_tf = []
	classes_gt_num = []
	image_scores = np.array([])
	for j in range(1, len(object_classes)):
		# May be there is a problem, for one roi could be assigned to many gt boxes
		# How to set the score thresh is a big problem
		cls_scores, cls_tp, cls_gt_num = \
			image_cls_eval(scores[:, j], boxes[:, j * 4:(j + 1) * 4], gt_boxes, j,
						   score_thresh=score_thresh, overlap_thresh=overlap_thresh, nms_thresh=nms_thresh)
		classes_scores += [cls_scores]
		classes_tf += [cls_tp]
		classes_gt_num += [cls_gt_num]
		image_scores = np.append(image_scores, cls_scores)

	# Limit to max_per_image detections *over all classes*
	if image_scores.size > max_per_image:
		image_thresh = np.sort(image_scores)[-max_per_image]
		for k in range(len(object_classes)-1):
			keep = np.where(classes_scores[k] >= image_thresh)
			classes_scores[k] = classes_scores[k][keep]
			classes_tf[k] = classes_tf[k][keep]

	return classes_scores, classes_tf, classes_gt_num

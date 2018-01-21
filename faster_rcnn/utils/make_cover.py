from __future__ import print_function
import os
import torch
import numpy as np
import time

from torch.autograd import Variable

from timer import Timer
from faster_rcnn.utils.timer import Timer

from faster_rcnn.utils.cython_bbox import region_objects, bbox_overlaps


TIME_IT = False


def region_pairs(region):
	object_num = len(region[region >= 0])
	return object_num * (object_num - 1)


def get_area(region):
	region_area = (region[:, 2] - region[:, 0] + 1) * \
					(region[:, 3] - region[:, 1] + 1)
	return region_area


def test(rel_obj_index):
	ob_i = np.array([])
	ob_j = np.array([])
	rel_id = np.array([])
	for i in range(len(rel_obj_index)):
		indexes = rel_obj_index[i][rel_obj_index[i] >= 0]
		id_i, id_j = np.meshgrid(indexes, indexes, indexing='ij')
		# upper triangle
		id_iu = np.triu_indices(len(indexes), k=1)
		id_i = id_i[id_iu]
		id_j = id_j[id_iu]
		# resize and append
		ob_i = np.append(ob_i, id_i.reshape(-1))
		ob_j = np.append(ob_j, id_j.reshape(-1))
		rel_id = np.append(rel_id, [i] * len(id_j.reshape(-1)))
	return ob_i, ob_j, rel_id


def compare_rel_rois(object_rois, relationship_rois, scores_object, scores_relationship,
					 topN_obj=128, topN_rel=128, obj_rel_thresh=0.8, max_objects=9, topN_covers=4096,
					 cover_thresh=0.5):
	timer = Timer()
	timer.tic()
	rel_obj_index = region_objects(
		np.ascontiguousarray(object_rois[:topN_obj, 1:], dtype=np.float),
		np.ascontiguousarray(relationship_rois[:topN_rel, 1:], dtype=np.float),
		threshold=obj_rel_thresh,
		max_objects=max_objects)

	# rel_obj_mat = bbox_intersections(relationship_rois[:topN_rel, 1:], object_rois[:topN_obj, 1:])
	# rel_obj_mat = rel_obj_mat.data.cpu().numpy()
	if TIME_IT:
		print('\t[rel_obj_mat]: %.3fs'%timer.toc(average=False))

	# get subject-object pair !!! don't distinguish subject or object
	# repeat to a 36000 column
	timer.tic()
	# 36 = 9 * 8 / 2, and 9 is maximum objects in one rel_cover
	total_pairs = max_objects*(max_objects-1)/2
	ob_i, ob_j = np.zeros((rel_obj_index.shape[0], total_pairs)), np.zeros((rel_obj_index.shape[0], total_pairs))
	rel_id = np.arange(rel_obj_index.shape[0])
	rel_id = np.repeat(rel_id, total_pairs)
	for i in range(rel_obj_index.shape[0]):
		# indexes = np.where(rel_obj_mat[i] >= 0.8)[0][:9]
		indexes = rel_obj_index[i]
		id_i, id_j = np.meshgrid(indexes, indexes, indexing='ij')
		# upper triangle
		id_iu = np.triu_indices(len(indexes), k=1)
		ob_i[i] = id_i[id_iu]
		ob_j[i] = id_j[id_iu]

	ob_i = ob_i.reshape(-1)
	ob_j = ob_j.reshape(-1)
	keep_obj_id = np.where((ob_i > -1) & (ob_j > -1))
	ob_i = ob_i[keep_obj_id].astype(int)
	ob_j = ob_j[keep_obj_id].astype(int)
	rel_id = rel_id[keep_obj_id].astype(int)
	# sub_t, obj_t, rel_t = test(rel_obj_index)
	if TIME_IT:
		print('\t[make pair]: %.3fs'%timer.toc(average=False))

	timer.tic()
	instance_covers = np.hstack((np.minimum(object_rois[ob_i][:, 1:3], object_rois[ob_j][:, 1:3]),
							   np.maximum(object_rois[ob_i][:, 3:], object_rois[ob_j][:, 3:])))

	# get intersections between covers and relationship_rois
	intersections = np.hstack((np.maximum(instance_covers[:, :2], relationship_rois[rel_id][:, 1:3]),
							   np.minimum(instance_covers[:, 2:], relationship_rois[rel_id][:, 3:])))

	instance_cover_inds = np.arange(instance_covers.shape[0])

	# get area of covers, relationship_rois, intersections
	instance_cover_area = get_area(instance_covers)
	# relationship_area = get_area(rel_relationship[:, 1:])
	intersections_area = get_area(intersections)
	overlaps = intersections_area/instance_cover_area

	fg_cover_inds = np.where(overlaps >= cover_thresh)[0]

	instance_cover_inds = instance_cover_inds[fg_cover_inds]
	overlaps = overlaps[fg_cover_inds]
	ob_i = ob_i[fg_cover_inds]
	ob_j = ob_j[fg_cover_inds]
	index = np.argsort(overlaps)
	instance_cover_inds = instance_cover_inds[index]
	ob_i = ob_i[index]
	ob_j = ob_j[index]
	rel_id = rel_id[index]

	stack_ids = np.vstack([ob_i, ob_j, rel_id, instance_cover_inds]).T
	# tmp = stack_ids[:, 0] + stack_ids[:, 1]*1j
	id_stack_ids = np.unique(stack_ids[:,:2], axis=0, return_index=True)[1]
	stack_ids = stack_ids[np.sort(id_stack_ids)]

	subject_id = stack_ids[:, 0]
	object_id = stack_ids[:, 1]
	relationship_id = stack_ids[:, 2]
	instance_cover_inds = stack_ids[:, 3]

	score_sbj = scores_object[subject_id]
	score_obj = scores_object[object_id]
	score_rel = scores_relationship[relationship_id]
	score_cover = (score_sbj*score_obj*score_rel).reshape(-1)
	score_ind = (-score_cover).argsort()[:topN_covers]
	subject_id = subject_id[score_ind]
	object_id = object_id[score_ind]
	instance_cover_inds = instance_cover_inds[score_ind]
	if TIME_IT:
		print('\t[select overlap]: %.3fs'%timer.toc(average=False))
	return subject_id, object_id, instance_covers[instance_cover_inds]


# def get_pair(object_rois):
# 	obj_overlaps = bbox_overlaps(np.ascontiguousarray(object_rois[:, 1:], dtype=np.float),
# 	                             np.ascontiguousarray(object_rois[:, 1:], dtype=np.float))
# 	is_overlap = obj_overlaps > 0
# 	id_u = np.triu_indices(object_rois.shape[0], k=1)
# 	index = np.where(is_overlap[id_u] == True)
# 	subject_id = index[0]
# 	object_id = index[1]
# 	return subject_id, object_id

# def compare_rel_rois(object_rois, relationship_rois, scores_object, scores_relationship,
#                      topN_obj=128, topN_rel=128, obj_rel_thresh=0.8, max_objects=9, topN_covers=4096,
#                      cover_thresh=0.5):
#     '''
#     :param object_rois: numpy array
#     :param relationship_rois: numpy array
#     :param thresh: float
#     :return: subject_id, object_id, rel_proposals: torch cuda
#     '''
#     # get nine objects in each relationship rois
#     timer = Timer()
#     timer.tic()
#     rel_obj_index = region_objects(
#         np.ascontiguousarray(object_rois.data.cpu().numpy()[:topN_obj, 1:], dtype=np.float),
#         np.ascontiguousarray(relationship_rois.data.cpu().numpy()[:topN_rel, 1:], dtype=np.float),
#         threshold=obj_rel_thresh,
#         max_objects=max_objects)
#
#     # rel_obj_mat = bbox_intersections(relationship_rois[:topN_rel, 1:], object_rois[:topN_obj, 1:])
#     # rel_obj_mat = rel_obj_mat.data.cpu().numpy()
#     if TIME_IT:
#         torch.cuda.synchronize()
#         print('\t[rel_obj_mat]: %.3fs' % timer.toc(average=False))
#
#     # print('relationship_rois number', len(rel_obj_index))
#     # print('object number', len(object_rois))
#     # print('objects number in all relationship_rois', len(rel_obj_index[rel_obj_index >= 0]))
#     # pairs_num = np.sum(np.apply_along_axis(region_pairs, 1, rel_obj_index))
#     # print('pair_num', pairs_num)
#
#     # get subject-object pair !!! don't distinguish subject or object
#     # repeat to a 36000 column
#     timer.tic()
#     # 36 = 9 * 8 / 2, and 9 is maximum objects in one rel_cover
#     total_pairs = max_objects*(max_objects-1)/2
#     ob_i, ob_j = np.zeros((rel_obj_index.shape[0], total_pairs)), np.zeros((rel_obj_index.shape[0], total_pairs))
#     rel_id = np.arange(rel_obj_index.shape[0])
#     rel_id = np.repeat(rel_id, total_pairs)
#     for i in range(rel_obj_index.shape[0]):
#         # indexes = np.where(rel_obj_mat[i] >= 0.8)[0][:9]
#         indexes = rel_obj_index[i]
#         id_i, id_j = np.meshgrid(indexes, indexes, indexing='ij')
#         # upper triangle
#         id_iu = np.triu_indices(len(indexes), k=1)
#         ob_i[i] = id_i[id_iu]
#         ob_j[i] = id_j[id_iu]
#
#     ob_i = ob_i.reshape(-1)
#     ob_j = ob_j.reshape(-1)
#     keep_obj_id = np.where((ob_i > -1) & (ob_j > -1))
#     ob_i = ob_i[keep_obj_id]
#     ob_j = ob_j[keep_obj_id]
#     rel_id = rel_id[keep_obj_id]
#     # sub_t, obj_t, rel_t = test(rel_obj_index)
#     if TIME_IT:
#         torch.cuda.synchronize()
#         print('\t[make pair]: %.3fs' % timer.toc(average=False))
#
#     timer.tic()
#     stack_ids = np.vstack([ob_i, ob_j, rel_id]).T
#     tmp = stack_ids[:, 0] + stack_ids[:, 1] * 1j
#     id_stack_ids = np.unique(tmp, return_index=True)[1]
#     stack_ids = stack_ids[id_stack_ids]
#
#     subject_id = torch.from_numpy(stack_ids[:, 0].astype('int')).cuda()
#     object_id = torch.from_numpy(stack_ids[:, 1].astype('int')).cuda()
#     relationship_id = torch.from_numpy(stack_ids[:, 2].astype('int')).cuda()
#     if TIME_IT:
#         torch.cuda.synchronize()
#         print('\t[unique & to tensor]: %.3fs' % timer.toc(average=False))
#
#     timer.tic()
#     # reflect to rois
#     rel_subject = torch.index_select(object_rois, 0, subject_id).data
#     score_sbj = scores_object[subject_id.cpu().numpy()]
#     rel_object = torch.index_select(object_rois, 0, object_id).data
#     score_obj = scores_object[object_id.cpu().numpy()]
#     rel_relationship = torch.index_select(relationship_rois, 0, relationship_id).data
#     score_rel = scores_relationship[relationship_id.cpu().numpy()]
#
#     score_cover = score_sbj*score_obj*score_rel
#     score_cover = torch.FloatTensor(score_cover.reshape(-1)).cuda()
#     if TIME_IT:
#         torch.cuda.synchronize()
#         print('\t[select & score]: %.3fs' % timer.toc(average=False))
#
#     timer.tic()
#     # get covers
#     rel_proposals = torch.cat((torch.zeros(rel_object.size()[0], 1).cuda(),
#                             torch.min(rel_subject[:, 1:3], rel_object[:, 1:3]),
#                             torch.max(rel_subject[:, 3:], rel_object[:, 3:])), dim=1)
#     # get intersections between covers and relationship_rois
#     intersections = torch.cat((torch.max(rel_proposals[:, 1:3], rel_relationship[:, 1:3]),
#                             torch.min(rel_proposals[:, 3:], rel_relationship[:, 3:])), dim=1)
#     # get area of covers, relationship_rois, intersections
#     proposals_area = get_area(rel_proposals[:, 1:])
#     relationship_area = get_area(rel_relationship[:, 1:])
#     intersections_area = get_area(intersections)
#
#     # calculate IoU between predict sbj-obj pair and predict relationship proposals
#     overlap = intersections_area / (proposals_area + relationship_area - intersections_area)
#     if TIME_IT:
#         torch.cuda.synchronize()
#         print('\t[overlap ratio]: %.3fs' % timer.toc(average=False))
#
#     # score_cover = score_cover.mul(overlap.data)
#     timer.tic()
#     keep_ind = overlap > cover_thresh
#     # np_keep = keep_ind.data.cpu().numpy()
#     # print('after compared with relationship_rois left pairs num', np.count_nonzero(np_keep))
#     # print('torch.sum', torch.sum(keep_ind))  # mistake here
#     # keep_ind is ByteTensor, ByteTensor.sum() > 255, so overflow
#
#     subject_id = torch.masked_select(subject_id.long(), keep_ind)
#     object_id = torch.masked_select(object_id.long(), keep_ind)
#
#     # expand keep_ind to (left_pairs_num, 5)
#     rel_keep_ind = keep_ind.view(-1, 1)
#     rel_keep_ind = rel_keep_ind.expand(rel_keep_ind.size()[0], 5)
#     # mask select left pairs
#     rel_proposals = torch.masked_select(rel_proposals, rel_keep_ind).view(-1, 5)
#
#     score_cover = torch.masked_select(score_cover, keep_ind)
#     score_ind = score_cover.unsqueeze(0).sort(dim=1, descending=True)[1].squeeze(0)[:topN_covers]
#     subject_id = subject_id[score_ind]
#     object_id = object_id[score_ind]
#     rel_proposals = rel_proposals[score_ind]
#     if TIME_IT:
#         torch.cuda.synchronize()
#         print('\t[select overlap]: %.3fs' % timer.toc(average=False))
#
#     return subject_id, object_id, Variable(rel_proposals)


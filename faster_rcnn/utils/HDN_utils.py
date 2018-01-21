import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import logging
from .cython_bbox import bbox_overlaps, bbox_intersections
# from bbox import bbox_overlaps as bbox_overlaps_py


def get_model_name(arguments):


    if arguments.nesterov:
        arguments.model_name += '_nesterov'

    if arguments.MPS_iter < 0:
        print 'Using random MPS iterations to training'
        arguments.model_name += '_rand_iters'
    else:
        arguments.model_name += '_{}_iters'.format(arguments.MPS_iter)


    # if arguments.use_kernel_function:
    #     arguments.model_name += '_with_kernel'
    if arguments.load_RPN or arguments.resume_model:
        arguments.model_name += '_alltrain'
    else:
        arguments.model_name += '_end2end'
    if arguments.dropout:
        arguments.model_name += '_dropout'
    arguments.model_name += '_{}'.format(arguments.dataset_option)
    # if arguments.disable_language_model:
    #     arguments.model_name += '_no_caption'
    # else:
    #     if arguments.rnn_type == 'LSTM_im':
    #         arguments.model_name += '_H_LSTM'
    #     elif arguments.rnn_type == 'LSTM_normal':
    #         arguments.model_name += '_I_LSTM'
    #     elif arguments.rnn_type == 'LSTM_baseline':
    #         arguments.model_name += '_B_LSTM'
    #     else:
    #         raise Exception('Error in RNN type')
    #     if arguments.caption_use_bias:
    #         arguments.model_name += '_with_bias'
    #     else:
    #         arguments.model_name += '_no_bias'
    #     if arguments.caption_use_dropout > 0:
    #         arguments.model_name += '_with_dropout_{}'.format(arguments.caption_use_dropout).replace('.', '_')
    #     else:
    #         arguments.model_name += '_no_dropout'
    #     arguments.model_name += '_nembed_{}'.format(arguments.nembedding)
    #     arguments.model_name += '_nhidden_{}'.format(arguments.nhidden_caption)
    #
    #     if arguments.region_bbox_reg:
    #         arguments.model_name += '_with_region_regression'

    if arguments.resume_model:
        arguments.model_name += '_resume'

    # if arguments.finetune_language_model:
    #     arguments.model_name += '_finetune'
    if arguments.optimizer == 0:
        arguments.model_name += '_SGD'
        arguments.solver = 'SGD'
    elif arguments.optimizer == 1:
        arguments.model_name += '_Adam'
        arguments.solver = 'Adam'
    elif arguments.optimizer == 2:    
        arguments.model_name += '_Adagrad'
        arguments.solver = 'Adagrad'
    else:
        raise Exception('Unrecognized optimization algorithm specified!')

    return arguments


def group_features(net_):
    vgg_features_fix = list(net_.rpn.features.parameters())[:8]
    vgg_features_var = list(net_.rpn.features.parameters())[8:]
    vgg_feature_len = len(list(net_.rpn.features.parameters()))
    rpn_feature_len = len(list(net_.rpn.parameters())) - vgg_feature_len
    rpn_features = list(net_.rpn.parameters())[vgg_feature_len:]
    # language_features = list(net_.caption_prediction.parameters())
    # language_feature_len = len(language_features)
    hdn_features = list(net_.parameters())[(rpn_feature_len + vgg_feature_len):]
    print 'vgg feature length:', vgg_feature_len
    print 'rpn feature length:', rpn_feature_len
    print 'HDN feature length:', len(hdn_features)
    # print 'language_feature_len:', language_feature_len
    return vgg_features_fix, vgg_features_var, rpn_features, hdn_features



def check_recall(rois, gt_objects, top_N, thresh=0.5):
    overlaps = bbox_overlaps(
        np.ascontiguousarray(rois.cpu().data.numpy()[:top_N, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_objects[:, :4], dtype=np.float))

    overlap_gt = np.amax(overlaps, axis=0)
    correct_cnt = np.sum(overlap_gt >= thresh)
    total_cnt = overlap_gt.size 
    return correct_cnt, total_cnt


def check_relationship_recall(gt_objects, gt_relationships,
                              subject_inds, object_inds, predicate_inds,
                              subject_boxes, object_boxes, top_Ns, thresh=0.5, only_predicate=False):
	# rearrange the ground truth
	gt_rel_sub_idx, gt_rel_obj_idx = np.where(gt_relationships > 0)  # ground truth number
	gt_sub = gt_objects[gt_rel_sub_idx, :5]
	gt_obj = gt_objects[gt_rel_obj_idx, :5]
	gt_rel = gt_relationships[gt_rel_sub_idx, gt_rel_obj_idx]

	rel_cnt = len(gt_rel)
	rel_correct_cnt = np.zeros(len(top_Ns))
	max_topN = max(top_Ns)

	# compute the overlap
	sub_overlaps = bbox_overlaps(
		np.ascontiguousarray(subject_boxes[:max_topN], dtype=np.float),
		np.ascontiguousarray(gt_sub[:, :4], dtype=np.float))
	obj_overlaps = bbox_overlaps(
		np.ascontiguousarray(object_boxes[:max_topN], dtype=np.float),
		np.ascontiguousarray(gt_obj[:, :4], dtype=np.float))

	for idx, top_N in enumerate(top_Ns):

		for gt_id in xrange(rel_cnt):
			fg_candidate = np.where(np.logical_and(
				sub_overlaps[:top_N, gt_id] >= thresh,
				obj_overlaps[:top_N, gt_id] >= thresh))[0]

			for candidate_id in fg_candidate:
				if only_predicate:
					if predicate_inds[candidate_id] == gt_rel[gt_id]:
						rel_correct_cnt[idx] += 1
						break
				else:
					if subject_inds[candidate_id] == gt_sub[gt_id, 4] and \
									predicate_inds[candidate_id] == gt_rel[gt_id] and \
									object_inds[candidate_id] == gt_obj[gt_id, 4]:
						rel_correct_cnt[idx] += 1
						break
	return rel_cnt, rel_correct_cnt


# def check_relationship_recall(gt_objects, gt_relationships, gt_regions,
#         subject_inds, object_inds, predicate_inds,
#         subject_boxes, object_boxes, predicate_boxes, top_Ns, thres=0.5, only_predicate=False, use_predicate_boxes=False):
#     # rearrange the ground truth
#     gt_rel_sub_idx, gt_rel_obj_idx = np.where(gt_relationships > 0) # ground truth number
#     gt_sub = gt_objects[gt_rel_sub_idx, :5]
#     gt_obj = gt_objects[gt_rel_obj_idx, :5]
#     gt_rel = gt_relationships[gt_rel_sub_idx, gt_rel_obj_idx]
#
#     rel_cnt = len(gt_rel)
#     rel_correct_cnt = np.zeros(len(top_Ns))
#     max_topN = max(top_Ns)
#
#     # compute the overlap
#     sub_overlaps = bbox_overlaps(
#         np.ascontiguousarray(subject_boxes[:max_topN], dtype=np.float),
#         np.ascontiguousarray(gt_sub[:, :4], dtype=np.float))
#     obj_overlaps = bbox_overlaps(
#         np.ascontiguousarray(object_boxes[:max_topN], dtype=np.float),
#         np.ascontiguousarray(gt_obj[:, :4], dtype=np.float))
#
#     if use_predicate_boxes:
#         predicate_overlaps = bbox_overlaps(
#             np.ascontiguousarray(predicate_boxes[:max_topN], dtype=np.float),
#             np.ascontiguousarray(gt_regions[:, :4], dtype=np.float))
#
#
#     for idx, top_N in enumerate(top_Ns):
#
#         for gt_id in xrange(rel_cnt):
#             fg_candidate = np.where(np.logical_and(
#                 sub_overlaps[:top_N, gt_id] >= thres,
#                 obj_overlaps[:top_N, gt_id] >= thres))[0]
#
#             for candidate_id in fg_candidate:
#                 if only_predicate:
#                     if use_predicate_boxes:
#                         if predicate_inds[candidate_id] == gt_rel[gt_id] and predicate_overlaps[candidate_id, gt_id] > thres:
#                             rel_correct_cnt[idx] += 1
#                             break
#                     else:
#                         if predicate_inds[candidate_id] == gt_rel[gt_id]:
#                             rel_correct_cnt[idx] += 1
#                             break
#                 else:
#                     if subject_inds[candidate_id] == gt_sub[gt_id, 4] and \
#                             predicate_inds[candidate_id] == gt_rel[gt_id] and \
#                             object_inds[candidate_id] == gt_obj[gt_id, 4]:
#                         if use_predicate_boxes:
#                             if predicate_overlaps[candidate_id, gt_id] > thres:
#                                 rel_correct_cnt[idx] += 1
#                                 break
#                         else:
#                             rel_correct_cnt[idx] += 1
#                             break
#     return rel_cnt, rel_correct_cnt


def check_obj_rel_recall(gt_objects, gt_relationships, gt_boxes_relationship, rel_covers, obj_rois,
                              subject_inds, object_inds, cover_thresh=0.5, object_thresh=0.5, log=False):
    '''
    :param gt_objects: n*5, np.array
    :param gt_relationships: object_num * object_num, np.array
    :param gt_boxes_relationship: np.array
    :param rel_covers: (batch(0), x1, y1, x2, y2), np.array
    :param obj_rois: (batch(0), x1, y1, x2, y2), np.array
    :param subject_inds: subject, id of obj_rois, np.array
    :param object_inds: object, id of obj_rois np.array
    :param cover_thresh:
    :return:
    '''
    logger = logging.getLogger('check_rel')
    if log:
        logging.basicConfig(level=logging.INFO)

    # compute the overlap between object_rois and gt_objects
    obj_overlaps = bbox_overlaps(np.ascontiguousarray(obj_rois[:, 1:], dtype=np.float),
                                 np.ascontiguousarray(gt_objects[:,:4], dtype=np.float))
    # remove objects that overlaps < 0.5 with any gt_objects
    obj_gt_assignment = obj_overlaps.argmax(axis=1)
    max_obj_overlaps = obj_overlaps.max(axis=1)

    logger.info('gt_objects: %d', gt_objects.shape[0])
    logger.info('left objects: %d', len(np.where(max_obj_overlaps >= object_thresh)[0]))
    logger.info('left all: %d', len(np.where(obj_overlaps >= object_thresh)[0]))

    # overlaps between relationship cover and gt_relationship
    rel_overlaps = bbox_overlaps(np.ascontiguousarray(rel_covers, dtype=np.float),
                                 np.ascontiguousarray(gt_boxes_relationship[:,:4], dtype=np.float))
    # get sub_obj and obj_sub combination
    subject_inds, object_inds = np.append(subject_inds, object_inds), np.append(object_inds, subject_inds)
    rel_overlaps = np.vstack((rel_overlaps, rel_overlaps))

    obj_gt_assignment[np.where(max_obj_overlaps < object_thresh)] = -1
    cover_obj_ind = obj_gt_assignment[object_inds]
    cover_sub_ind = obj_gt_assignment[subject_inds]
    keep_obj_ind = np.logical_and(cover_obj_ind >= 0, cover_sub_ind >= 0)
    # keep_obj_ind = np.append(keep_obj_ind, keep_obj_ind)

    # set rel_overlaps == 0 that smaller than region_thresh
    rel_overlaps[np.logical_not(keep_obj_ind)] = 0
    rel_overlaps[rel_overlaps < cover_thresh] = 0

    # keep rel_cover that overlap >= cover_thresh with any gt_relationship_boxes
    # and keep all mapping gt_relationship
    selected_rel_cover_inds, rel_gt_assignment = np.where(rel_overlaps >= cover_thresh)
    # keep relationship cover
    subject_selected = subject_inds[selected_rel_cover_inds]
    object_selected = object_inds[selected_rel_cover_inds]

    # get subject id and object id of each gt_relationship
    gt_rel_sub_idx, gt_rel_obj_idx = np.where(gt_relationships > 0)  # ground truth number

    # get region keep inds
    fg_cover_bool = np.logical_and(
        (obj_gt_assignment[subject_selected] == gt_rel_sub_idx[rel_gt_assignment]),
        (obj_gt_assignment[object_selected] == gt_rel_obj_idx[rel_gt_assignment]))
    bg_cover_part = np.logical_not(fg_cover_bool)
    rel_overlaps[selected_rel_cover_inds[bg_cover_part], rel_gt_assignment[bg_cover_part]] = 0

    fg_cover_inds = np.where(rel_overlaps.max(axis=1) > 0)[0]
    fg_cover_gt_assignment = rel_overlaps.argmax(axis=1)[fg_cover_inds]

    recall = len(np.unique(fg_cover_gt_assignment))

    # fg_cover num
    fg_cover_num = len(fg_cover_gt_assignment)
    gt_relationships_num = len(gt_rel_sub_idx)

    # part fg objects(mapping to fg_cover)
    fg_pair_subject_inds = subject_inds[fg_cover_inds]
    fg_pair_object_inds = object_inds[fg_cover_inds]
    fg_object_inds_part = np.unique(np.append(fg_pair_subject_inds, fg_pair_object_inds))

    logger.info('recall: %d' % recall)
    logger.info('ground_truth relationship: %d' % len(gt_rel_sub_idx))
    logger.info('fg_relationship proposal number: %d' % fg_cover_num)

    return recall, fg_cover_num, fg_object_inds_part.shape[0], \
           fg_cover_num + gt_relationships_num, fg_object_inds_part.shape[0] + gt_objects.shape[0]


def obj_in_predicate(object_rois, relationship_rois, top_k):
    num_obj = object_rois.size(0)
    num_rel = relationship_rois.size(0)
    box_index = torch.zeros(num_rel, top_k)
    repeat_relationship = relationship_rois.repeat(1, num_obj).view(-1, 4)
    repeat_obj = object_rois.repeat(num_rel, 1)
    index = (repeat_obj[:, 0] <= repeat_relationship[:, 0]).data & \
            (repeat_obj[:, 1] <= repeat_relationship[:, 1]).data & \
            (repeat_obj[:, 2] >= repeat_relationship[:, 2]).data & \
            (repeat_obj[:, 3] >= repeat_relationship[:, 3]).data
    index = index.view(num_rel, num_obj).nonzero()
    print(index)
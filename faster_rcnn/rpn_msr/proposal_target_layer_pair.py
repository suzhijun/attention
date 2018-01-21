# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import yaml
import numpy as np
import numpy.random as npr
import torch
from torch.autograd import Variable
# import ipdb

from ..utils.cython_bbox import bbox_overlaps, bbox_intersections
from ..utils.make_cover import compare_rel_rois
# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import bbox_transform

# <<<< obsolete

DEBUG = True


def proposal_target_layer(object_rois, region_rois, scores_object, scores_relationship,
                          gt_objects, gt_relationships, gt_regions, n_classes_obj,
                          n_classes_pred, is_training, graph_generation=False):
    #     object_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2] proposed by RPN,
    #     region_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2] proposed by RPN,
    #     gt_objects:   (G_obj, 5) [x1 ,y1 ,x2, y2, obj_class] float,
    #     gt_relationships: (G_obj, G_obj) [pred_class] int (-1 for no relationship),
    #     gt_regions:   (G_region, 4+40) [x1, y1, x2, y2, word_index] (imdb.eos for padding),
    #     # gt_ishard: (G_region, 4+40) {0 | 1} 1 indicates hard
    #     # dontcare_areas: (D, 4) [ x1, y1, x2, y2]
    #     n_classes_obj
    #     n_classes_pred
    #     is_training to indicate whether in training scheme

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source

    # TODO(rbg): it's annoying that sometimes I have extra info before
    # and other times after box coordinates -- normalize to one format

    # Include ground-truth boxes in the set of candidate rois

    # assert is_training == True, 'Evaluation Code haven\'t been implemented'

    # Sample rois with classification labels and bounding box regression
    # targets
    if is_training:

        all_rois = object_rois
        zeros = np.zeros((gt_objects.shape[0], 1), dtype=gt_objects.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_objects[:, :4])))
        )
        all_scores_object = np.append(scores_object, np.ones(gt_objects.shape[0], dtype=scores_object.dtype))
        all_rois_region = region_rois
        zeros = np.zeros((gt_regions.shape[0], 1), dtype=gt_regions.dtype)
        all_rois_region = np.vstack(
            (all_rois_region, np.hstack((zeros, gt_regions[:, :4])))
        )
        all_scores_relationship = np.append(scores_relationship,
                                            np.ones(gt_regions.shape[0], dtype=scores_relationship.dtype))

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
            'Only single item batches are supported'

        object_labels, object_rois, bbox_targets_object, bbox_inside_weights_object, \
        phrase_labels, phrase_rois, bbox_targets_phrase, bbox_inside_weights_phrase, mat_object = \
            _sample_rois(all_rois, all_rois_region, all_scores_object, all_scores_relationship,
                         gt_objects, gt_relationships, gt_regions, 1,
                         n_classes_obj, n_classes_pred)
        mat_phrase = None  # it's useless in the training phase
        # assert phrase_labels.shape[1] == cfg.TRAIN.LANGUAGE_MAX_LENGTH
        object_labels = object_labels.reshape(-1, 1)
        phrase_labels = phrase_labels.reshape(-1, 1)
        bbox_targets_object = bbox_targets_object.reshape(-1, n_classes_obj * 4)
        bbox_targets_phrase = bbox_targets_phrase.reshape(-1, n_classes_pred * 4)
        bbox_inside_weights_object = bbox_inside_weights_object.reshape(-1, n_classes_obj * 4)
        bbox_inside_weights_phrase = bbox_inside_weights_phrase.reshape(-1, n_classes_pred * 4)
        bbox_outside_weights_object = np.array(bbox_inside_weights_object > 0).astype(np.float32)
        bbox_outside_weights_phrase = np.array(bbox_inside_weights_phrase > 0).astype(np.float32)
    else:
        subject_inds, object_inds, phrase_rois = compare_rel_rois(
            object_rois, region_rois, scores_object, scores_relationship,
            topN_obj=cfg.TEST.MPN_BBOX_NUM, topN_rel=cfg.TEST.MPN_REGION_NUM,
            obj_rel_thresh=cfg.TEST.MPN_OBJ_REL_THRESH,
            max_objects=cfg.TEST.MPN_MAX_OBJECTS, topN_covers=cfg.TEST.MPN_COVER_NUM,
            cover_thresh=cfg.TEST.MPN_MAKE_COVER_THRESH)

        object_rois, phrase_rois, mat_object, mat_phrase = \
            _setup_connection(object_rois, phrase_rois,
                              subject_inds, object_inds, graph_generation=graph_generation)
        object_labels, bbox_targets_object, bbox_inside_weights_object, bbox_outside_weights_object, \
        phrase_labels, bbox_targets_phrase, bbox_inside_weights_phrase, bbox_outside_weights_phrase = [None] * 8
    # print 'phrase_roi', phrase_roi
    # print 'object_rois'
    # print object_rois
    # print 'phrase_rois'
    # print phrase_rois

    if DEBUG:
        # print 'phrase_roi'
        # print phrase_roi
        # print 'object num fg: {}'.format((object_labels > 0).sum())
        # print 'object num bg: {}'.format((object_labels == 0).sum())
        # print 'relationship num fg: {}'.format((phrase_labels > 0).sum())
        # print 'relationship num bg: {}'.format((phrase_labels == 0).sum())
        count = 1
        fg_num = np.where(object_labels > 0)[0].size
        bg_num = np.where(object_labels == 0)[0].size
        print('object num fg avg: {}'.format(fg_num / count))
        print('object num bg avg: {}'.format(bg_num / count))
        print('ratio: {:.3f}'.format(float(fg_num) / float(bg_num)))
        count_rel = 1
        fg_num_rel = (phrase_labels > 0).sum()
        bg_num_rel = (phrase_labels == 0).sum()
        print('relationship num fg avg: {}'.format(fg_num_rel / count_rel))
        print('relationship num bg avg: {}'.format(bg_num_rel / count_rel))
        print('ratio: {:.3f}'.format(float(fg_num_rel) / float(bg_num_rel)))
        # print mat_object.shape
        # print mat_phrase.shape
        # print 'phrase_roi'
        # print phrase_roi

    # mps_object [object_batchsize, 2, n_phrase] : the 2 channel means inward(object) and outward(subject) list
    # mps_phrase [phrase_batchsize, 2 + n_phrase]
    # mps_phrase [phrase_batchsize, n_phrase]
    assert object_rois.shape[1] == 5
    assert phrase_rois.shape[1] == 5
    return object_labels, object_rois, bbox_targets_object, bbox_inside_weights_object, bbox_outside_weights_object, \
           phrase_labels, phrase_rois, bbox_targets_phrase, bbox_inside_weights_phrase, bbox_outside_weights_phrase, \
           mat_object, mat_phrase


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th) where N =256 here

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    # if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    #     # Optionally normalize targets by a precomputed mean and stdev
    #     targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
    #                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _get_fg_phrase_inds(obj_gt_assignment, phrase_overlaps, subject_inds, object_inds, gt_phraseationships):
    '''
    get fg phrase inds and fg phrase gt assignment
    :param obj_gt_assignment: 
    :param phrase_overlaps: 
    :param subject_inds: 
    :param object_inds: 
    :param gt_phraseationships: 
    :return: 
    '''

    # to keep more positive sample
    # we think one phrase rois could assigned to many gt_phraseationship_boxes that overlap >= FG_THRESH_phrase
    # so we keep these phrase rois and their corresponding gt_phraseationship
    selected_phrase_rois_inds, phrase_gt_assignment = np.where(phrase_overlaps >= cfg.TRAIN.MPN_FG_THRESH_PHRASE)

    # get corresponding obj id
    subject_selected = subject_inds[selected_phrase_rois_inds]
    object_selected = object_inds[selected_phrase_rois_inds]

    # get subject id and object id of each gt_relationship
    gt_phrase_sub_idx, gt_phrase_obj_idx = np.where(gt_phraseationships > 0)

    # keep these phrase whose subject, object both assigned to right ground truth object
    selected_fg_phrase_inds = np.logical_and(
        (obj_gt_assignment[subject_selected] == gt_phrase_sub_idx[phrase_gt_assignment]),
        (obj_gt_assignment[object_selected] == gt_phrase_obj_idx[phrase_gt_assignment]))
    # set phrase_overlaps of these FALSE phrase to zeros
    selected_bg_phrase_inds = np.logical_not(selected_fg_phrase_inds)
    phrase_overlaps[selected_phrase_rois_inds[selected_bg_phrase_inds],
                    phrase_gt_assignment[selected_bg_phrase_inds]] = 0

    # choose fg_phrase that maximum phrase_overlap >= cfg.TRAIN.MPN_FG_THRESH_PHRASE
    max_phrase_overlaps = phrase_overlaps.max(axis=1)
    fg_phrase_inds = np.where(max_phrase_overlaps >= cfg.TRAIN.MPN_FG_THRESH_PHRASE)[0]
    # print(fg_phrase_inds.size)
    fg_phrase_gt_assignment = phrase_overlaps.argmax(axis=1)[fg_phrase_inds]
    return fg_phrase_inds, fg_phrase_gt_assignment


def _sample_rois(object_rois, region_rois, scores_object, scores_relationship,
                     gt_objects, gt_relationships, gt_regions, num_images, n_classes_obj, n_classes_pred):
    """Sample Object RoIs and Relationship phrase RoIs, comprising foreground and background
    examples.
    """
    # ipdb.set_trace()
    # -------- object overlap and other operations ---------

    # get fg object rois number
    rois_per_image = int(cfg.TRAIN.MPN_BATCH_SIZE / num_images)
    fg_rois_per_image = int(np.round(cfg.TRAIN.MPN_FG_FRACTION * rois_per_image))
    # get fg phrase rois number
    phrase_rois_per_image = int(cfg.TRAIN.MPN_BATCH_SIZE_PHRASE / num_images)
    fg_phrase_rois_per_image = int(np.round(cfg.TRAIN.MPN_FG_FRACTION_PHRASE * phrase_rois_per_image))

    # --------- get positive object ----------

    # In training stage use gt_obj to choose proper predict obj_roi,
    # here obj_roi got by rpn and concat with gt_box
    obj_overlaps = bbox_overlaps(
        np.ascontiguousarray(object_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_objects[:, :4], dtype=np.float))
    obj_gt_assignment = obj_overlaps.argmax(axis=1)
    max_obj_overlaps = obj_overlaps.max(axis=1)

    # Predict label of rpn+gt objs = pre_sampled objs
    obj_labels = gt_objects[obj_gt_assignment, 4]
    # get fg object inds and bg object inds
    fg_object_inds = np.where(max_obj_overlaps >= cfg.TRAIN.MPN_FG_THRESH)[0]
    bg_object_inds = np.where((max_obj_overlaps < cfg.TRAIN.MPN_BG_THRESH_HI) &
                              (max_obj_overlaps >= cfg.TRAIN.MPN_BG_THRESH_LO))[0]

    # ----------- get positive region rois ----------

    region_overlaps = bbox_overlaps(
        np.ascontiguousarray(region_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_regions[:, :4], dtype=np.float))
    keep_fg_region_inds = np.where(region_overlaps.max(axis=1) >= cfg.TRAIN.MPN_FG_THRESH_REGION)[0]

    # ---------- get all possible triplet ------------
    # by fg object rois and fg region rois
    # get fg phrase(cover) rois and corresponding subject/object list(subject_ids, object_ids)
    # which contain index of object rois.
    # if topN_covers==None, return all combinations
    # note: subject_ids/object_ids need to be paired to get all sub-obj and obj-sub combination
    # value of subject/object_ids range from 0 to number of fg_object
    subject_ids, object_ids, phrase_rois = \
        compare_rel_rois(object_rois[fg_object_inds], region_rois[keep_fg_region_inds],
                         scores_object[fg_object_inds], scores_relationship[keep_fg_region_inds],
                         topN_obj=fg_object_inds.size, topN_rel=keep_fg_region_inds.size,
                         obj_rel_thresh=cfg.TRAIN.MPN_OBJ_REL_THRESH, max_objects=fg_object_inds.size,
                         topN_covers=None, cover_thresh=cfg.TRAIN.MPN_MAKE_COVER_THRESH)
    # mapping subject/object_ids to original object rois index
    subject_inds = fg_object_inds[subject_ids]
    object_inds = fg_object_inds[object_ids]

    # -------------- phrase overlap ----------------

    phrase_overlaps = bbox_overlaps(
        np.ascontiguousarray(phrase_rois[:, 1:], dtype=np.float),
        np.ascontiguousarray(gt_regions[:, :4], dtype=np.float)
    )
    # get all sub-obj and obj-sub combination
    subject_inds, object_inds = np.append(subject_inds, object_inds), np.append(object_inds, subject_inds)
    phrase_rois = np.vstack((phrase_rois, phrase_rois))
    phrase_overlaps = np.vstack((phrase_overlaps, phrase_overlaps))

    if DEBUG:
        pairs = np.vstack((subject_inds, object_inds)).T
        gt_rels = np.where(gt_relationships > 0)
        gt_pairs = np.vstack((gt_rels[0] + object_rois.shape[0] - gt_objects.shape[0],
                              gt_rels[1] + object_rois.shape[0] - gt_objects.shape[0])).T
        recall_gt_rel = 0
        for i, gt_pair in enumerate(gt_pairs):
            mapping_gt_bool = (gt_pair == pairs).all(axis=1)
            if mapping_gt_bool.any():
                recall_gt_rel += 1
                mapping_cover_inds = np.where(mapping_gt_bool==True)[0]
                print('gt_rel_box', i, gt_regions[i])
                print('cover', mapping_cover_inds, phrase_rois[mapping_gt_bool])
                mapping_overlaps = \
                    bbox_overlaps(
                        np.ascontiguousarray(phrase_rois[mapping_cover_inds, 1:], dtype=np.float),
                        np.ascontiguousarray(gt_regions[:, :4], dtype=np.float))
                print('mpping_overlaps', mapping_overlaps)
        print('gt_relationship', gt_rels[0].size)
        print('recall_gt_rel', recall_gt_rel)

    # ---------- get fg_phrase and bg_phrase ----------

    # get fg phrase inds
    phrase_gt_assignment = phrase_overlaps.argmax(axis=1)
    fg_phrase_inds, fg_phrase_gt_assignment = _get_fg_phrase_inds(
        obj_gt_assignment, phrase_overlaps, subject_inds, object_inds, gt_relationships)
    phrase_gt_assignment[fg_phrase_inds] = fg_phrase_gt_assignment

    # get fix number fg phrase
    fg_phrase_per_this_image = min(fg_phrase_rois_per_image, fg_phrase_inds.size)
    if fg_phrase_inds.size > 0:
        fg_phrase_inds = npr.choice(fg_phrase_inds, size=fg_phrase_per_this_image, replace=False)
    fg_phrase_gt_assignment = phrase_gt_assignment[fg_phrase_inds]
    fg_phrase_labels = gt_regions[fg_phrase_gt_assignment, 4]
    # get corresponding subect inds / object inds
    fg_sub_inds, fg_obj_inds = subject_inds[fg_phrase_inds], object_inds[fg_phrase_inds]

    # ---------- get positive fg object and bg object inds -------------

    # first of all, get fg object from positive phrase(corresponding triplet)
    fg_object_inds_part = np.unique(np.append(fg_sub_inds, fg_obj_inds))

    # if fg object from positive phrase more than maximum fg object number,
    #     just random choice (or according to score?)
    # else insufficient from other objects that overlap > thresh
    fg_object_per_this_image = min(fg_rois_per_image, fg_object_inds.size)
    if fg_object_inds_part.size >= fg_object_per_this_image:
        fg_object_inds = npr.choice(fg_object_inds_part,
                                    size=fg_object_per_this_image,
                                    replace=False)
    else:
        fg_object_other_part = fg_object_per_this_image - fg_object_inds_part.size
        fg_object_inds = np.append(fg_object_inds_part,
                                   npr.choice(np.setdiff1d(fg_object_inds, fg_object_inds_part),
                                              size=fg_object_other_part, replace=False))
    # get bg object inds
    bg_object_per_this_image = min(rois_per_image - fg_object_inds.size, bg_object_inds.size)
    if bg_object_inds.size > 0:
        bg_object_inds = npr.choice(bg_object_inds,
                                    size=bg_object_per_this_image,
                                    replace=False)

    # ----------- get kept object inds and corresponding data -----------

    keep_object_inds = np.append(fg_object_inds, bg_object_inds)
    object_labels = obj_labels[keep_object_inds]
    object_labels[fg_object_inds.size:] = 0

    bbox_target_data_object = _compute_targets(
        object_rois[keep_object_inds][:, 1:5], gt_objects[obj_gt_assignment[keep_object_inds], :4], object_labels)
    bbox_targets_object, bbox_inside_weights_object = \
        _get_bbox_regression_labels(bbox_target_data_object, n_classes_obj)

    # ----------- get bg phrase inds ------------

    # get possible bg phrase rois by chosen fg&bg object rois
    # ToDo: is it right use Test Setting?
    bg_subject_ids, bg_object_ids, bg_phrase_rois = compare_rel_rois(
        object_rois[keep_object_inds], region_rois,
        scores_object[keep_object_inds], scores_relationship,
        topN_obj=keep_object_inds.size, topN_rel=cfg.TEST.MPN_REGION_NUM,
        obj_rel_thresh=cfg.TEST.MPN_OBJ_REL_THRESH,
        max_objects=cfg.TEST.MPN_MAX_OBJECTS, topN_covers=cfg.TEST.MPN_COVER_NUM,
        cover_thresh=cfg.TEST.MPN_MAKE_COVER_THRESH
    )
    # to get bg phrase gt_assignment
    bg_phrase_overlaps = bbox_overlaps(
        np.ascontiguousarray(bg_phrase_rois[:, 1:], dtype=np.float),
        np.ascontiguousarray(gt_regions[:, :4], dtype=np.float)
    )

    # get all sub-obj and obj-sub combination
    bg_subject_ids, bg_object_ids = np.append(bg_subject_ids, bg_object_ids), np.append(bg_object_ids, bg_subject_ids)
    bg_phrase_rois = np.vstack((bg_phrase_rois, bg_phrase_rois))
    bg_phrase_overlaps = np.vstack((bg_phrase_overlaps, bg_phrase_overlaps))

    # get bg phrase inds and corresponding sub_list, obj_list
    # bg_sub_list/bg_obj_list, value is index of keep_object_inds
    bg_phrase_rois_per_this_image = min(phrase_rois_per_image - fg_phrase_inds.size, bg_phrase_rois.shape[0])
    bg_phrase_inds = npr.choice(bg_phrase_rois.shape[0], size=bg_phrase_rois_per_this_image, replace=False)
    bg_sub_list, bg_obj_list = bg_subject_ids[bg_phrase_inds], bg_object_ids[bg_phrase_inds]
    bg_phrase_gt_assignment = bg_phrase_overlaps.argmax(axis=1)[bg_phrase_inds]

    # ---------- get phrase keep inds and other corresponding data -----------

    # set phrase label
    phrase_labels = np.append(fg_phrase_labels, np.zeros(bg_phrase_inds.size))
    phrase_rois = np.vstack((phrase_rois[fg_phrase_inds], bg_phrase_rois[bg_phrase_inds]))
    keep_phrase_gt_assignment = np.append(fg_phrase_gt_assignment, bg_phrase_gt_assignment)

    bbox_target_data_phrase = _compute_targets(
        phrase_rois[:, 1:5], gt_regions[keep_phrase_gt_assignment, :4], phrase_labels)
    bbox_targets_phrase, bbox_inside_weights_phrase = \
        _get_bbox_regression_labels(bbox_target_data_phrase, n_classes_pred)

    # ---------- get mat-pair ----------

    mat_object = _get_mat_object(
        keep_object_inds, fg_sub_inds, fg_obj_inds, bg_sub_list, bg_obj_list, object_rois.shape[0])

    return object_labels, object_rois[keep_object_inds], bbox_targets_object, bbox_inside_weights_object,\
           phrase_labels, phrase_rois, bbox_targets_phrase, bbox_inside_weights_phrase, mat_object


def _setup_connection(object_rois, phrase_rois, subject_inds, object_inds, graph_generation=False):

    object_num = min(cfg.TEST.MPN_BBOX_NUM, object_rois.shape[0])
    object_rois = object_rois[:object_num, :]
    # get sub_obj and obj_sub combination
    subject_inds, object_inds = np.append(subject_inds, object_inds), np.append(object_inds, subject_inds)
    phrase_rois = np.vstack((phrase_rois, phrase_rois))

    # prepare connection matrix
    mat_object = np.zeros((object_rois.shape[0], 2, phrase_rois.shape[0]), dtype=np.int64)
    for i in range(phrase_rois.shape[0]):
        mat_object[subject_inds[i], 0, i] = 1
        mat_object[object_inds[i], 1, i] = 1

    mat_phrase = np.zeros((subject_inds.size, 2), dtype=np.int64)
    mat_phrase[:, 0] = subject_inds
    mat_phrase[:, 1] = object_inds

    return object_rois, phrase_rois, mat_object, mat_phrase


def _get_mat_object(keep_object_inds, fg_sub_inds, fg_obj_inds, bg_sub_list, bg_obj_list, obj_rois_num):
    '''
    :param keep_object_inds: index of kept objects(fg + bg), value is index of original object rois
    :param fg_sub_inds: subject inds of fg phrase, value is index of original object rois
    :param fg_obj_inds: object inds of fg phrase, value is index of original object rois
    :param bg_sub_list: subject inds of bg phrase, value is index of keep_object_inds(fg object rois, bg object rois)
    :param bg_obj_list: object inds of bg phrase, value is index of keep_object_inds(fg object rois, bg object rois)
    :param obj_rois_num: original object rois number
    :return: 
    '''

    keep_object_num = keep_object_inds.size
    keep_phrase_num = fg_sub_inds.size + bg_sub_list.size

    # like a dictionary, value of object_mapping is the new index of object rois in keep_object_inds
    object_mapping = np.full(obj_rois_num, keep_object_num)
    object_mapping[keep_object_inds] = np.arange(keep_object_num)

    # mapping index of original object rois in fg_sub/obj_inds to index of keep_object_inds
    fg_sub_list, fg_obj_list = object_mapping[fg_sub_inds], object_mapping[fg_obj_inds]

    # to get the sub_list and obj_list which keep relationships between object and relationship phrase
    sub_list = np.append(fg_sub_list, bg_sub_list)
    obj_list = np.append(fg_obj_list, bg_obj_list)

    # objects * (sub or obj) * phrase(relationship proposal)
    mat_object = np.zeros((keep_object_num, 2, keep_phrase_num), dtype=np.int64)

    # different with MSDN, our phrase Maybe not corresponding to any sub or obj
    # so we have to prepare_message in message passing process just like objects
    # the transpose of mat_object could be used as the select_mat in function prepare_message
    for i in range(keep_phrase_num):
        if sub_list[i] < keep_object_num:
            mat_object[sub_list[i], 0, i] = 1
        if obj_list[i] < keep_object_num:
            mat_object[obj_list[i], 1, i] = 1

    return mat_object
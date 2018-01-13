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
import pdb

from ..utils.cython_bbox import bbox_overlaps, bbox_intersections
from ..utils.make_cover import compare_rel_rois
# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import bbox_transform

# <<<< obsolete

DEBUG = False


def proposal_target_layer(object_rois, region_rois, scores_object, scores_relationship,
                          gt_objects, gt_relationships, gt_regions, n_classes_obj,
                          n_classes_pred, is_training, graph_generation=False):
    #     object_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2] proposed by RPN, pytorch cuda variable
    #     region_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2] proposed by RPN, pytorch cuda variable
    #     gt_objects:   (G_obj, 5) [x1 ,y1 ,x2, y2, obj_class] float, tensor
    #     gt_relationships: (G_obj, G_obj) [pred_class] int (-1 for no relationship), tensor
    #     gt_regions:   (G_region, 4+40) [x1, y1, x2, y2, word_index] (imdb.eos for padding), tensor
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
        zeros = torch.zeros(gt_objects.size()[0], 1).cuda()
        # add gt_obj to predict_rois
        all_rois = torch.cat(
            (all_rois, Variable(torch.cat((zeros, gt_objects[:, :4].cuda()), dim=1))), dim=0)
        all_scores_object = np.append(scores_object, np.ones(gt_objects.shape[0], dtype=scores_object.dtype))

        all_rois_region = region_rois
        zeros = torch.zeros(gt_regions.size()[0], 1).cuda()
        all_rois_region = torch.cat(
            (all_rois_region, Variable(torch.cat((zeros, gt_regions[:, :4].cuda()), dim=1))), dim=0)
        all_scores_relationship = np.append(scores_relationship,
                                            np.ones(gt_regions.shape[0], dtype=scores_relationship.dtype))

        subject_ids, object_ids, all_rois_phrase = compare_rel_rois(
            all_rois, all_rois_region, all_scores_object, all_scores_relationship,
            topN_obj=all_rois.size()[0], topN_rel=all_rois_region.size()[0],
            obj_rel_thresh=cfg.TRAIN.MPN_OBJ_REL_THRESH,
            max_objects=cfg.TRAIN.MPN_MAX_OBJECTS, topN_covers=cfg.TRAIN.MPN_COVER_NUM,
            cover_thresh=cfg.TRAIN.MPN_MAKE_COVER_THRESH)
        # Sanity check: single batch only
        # assert np.all(all_rois[:, 0] == 0), \
        #     'Only single item batches are supported'

        # last step, jia add gt_object and gt_regions to object proposals and relationship proposals,
        # for some reason, the compare_rel_rois function didn't recall all gt_relationship
        # so we add gt_relationship to all_rois_phrase directly here
        all_rois_phrase = all_rois_phrase.data.cpu().numpy()
        zeros = np.zeros((gt_regions.numpy().shape[0], 1), dtype=gt_regions.numpy().dtype)
        all_rois_phrase = np.vstack((all_rois_phrase, np.hstack((zeros, gt_regions.numpy()[:, :4]))))
        gt_rel_sub_idx, gt_rel_obj_idx = np.where(gt_relationships.numpy() > 0)
        gt_rel_sub_idx, gt_rel_obj_idx = gt_rel_sub_idx + object_rois.size()[0], gt_rel_obj_idx + object_rois.size()[0]
        subject_inds = np.append(subject_ids.cpu().numpy(), gt_rel_sub_idx)
        object_inds = np.append(object_ids.cpu().numpy(), gt_rel_obj_idx)

        object_labels, object_rois, bbox_targets_object, bbox_inside_weights_object, \
        phrase_labels, phrase_rois, bbox_targets_phrase, bbox_inside_weights_phrase, mat_object = \
            _sample_rois(all_rois.data.cpu().numpy(), all_rois_phrase,
                         subject_inds, object_inds,
                         gt_objects.numpy(), gt_relationships.numpy(), gt_regions.numpy(), 1,
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
        object_rois_num = min(cfg.TEST.MPN_BBOX_NUM, object_rois.size()[0])
        region_rois_num = min(cfg.TEST.MPN_REGION_NUM, region_rois.size()[0])
        truncated_object_rois = object_rois[:object_rois_num, :]
        truncated_region_rois = region_rois[:region_rois_num, :]
        subject_inds, object_inds, phrase_rois = compare_rel_rois(
            truncated_object_rois, truncated_region_rois, scores_object, scores_relationship,
            topN_obj=cfg.TEST.MPN_BBOX_NUM, topN_rel=cfg.TEST.MPN_REGION_NUM,
            obj_rel_thresh=cfg.TEST.MPN_OBJ_REL_THRESH,
            max_objects=cfg.TEST.MPN_MAX_OBJECTS, topN_covers=cfg.TEST.MPN_COVER_NUM,
            cover_thresh=cfg.TEST.MPN_MAKE_COVER_THRESH)

        object_rois, phrase_rois, mat_object, mat_phrase = \
            _setup_connection(truncated_object_rois.data.cpu().numpy(), phrase_rois.data.cpu().numpy(),
                              subject_inds.cpu().numpy(), object_inds.cpu().numpy(), graph_generation=graph_generation)
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
        fg_num = (object_labels > 0).sum()
        bg_num = (object_labels == 0).sum()
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


def _sample_rois(object_rois, phrase_rois, subject_inds, object_inds,
                 gt_objects, gt_relationships, gt_regions, num_images, n_classes_obj, n_classes_pred):
    """Sample Object RoIs and Relationship phrase RoIs, comprising foreground and background
    examples.
    """
    assert phrase_rois.shape[0] == subject_inds.shape[0]

    phrase_rois_per_image = int(cfg.TRAIN.MPN_BATCH_SIZE_PHRASE / num_images)
    fg_phrase_rois_per_image = int(np.round(cfg.TRAIN.MPN_FG_FRACTION_PHRASE * phrase_rois_per_image))

    # -------- object overlap and other operations ---------

    # In training stage use gt_obj to choose proper predict obj_roi,
    # here obj_roi got by rpn and concat with gt_box
    obj_overlaps = bbox_overlaps(
        np.ascontiguousarray(object_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_objects[:, :4], dtype=np.float))
    obj_gt_assignment = obj_overlaps.argmax(axis=1)
    max_obj_overlaps = obj_overlaps.max(axis=1)

    # Predict label of rpn+gt objs = pre_sampled objs
    obj_labels = gt_objects[obj_gt_assignment, 4]
    fg_object_inds = np.where(max_obj_overlaps >= cfg.TRAIN.MPN_FG_THRESH)[0]
    bg_object_inds = np.where((max_obj_overlaps < cfg.TRAIN.MPN_BG_THRESH_HI) &
                              (max_obj_overlaps >= cfg.TRAIN.MPN_BG_THRESH_LO))[0]

    # -------- phrase overlap and other operations ----------

    # overlaps between relationship cover and gt_relationship
    rel_overlaps = bbox_overlaps(
        np.ascontiguousarray(phrase_rois[:, 1:], dtype=np.float),
        np.ascontiguousarray(gt_regions[:, :4], dtype=np.float)
    )

    # get sub_obj and obj_sub combination
    subject_inds, object_inds = np.append(subject_inds, object_inds), np.append(object_inds, subject_inds)
    phrase_rois = np.vstack((phrase_rois, phrase_rois))
    rel_overlaps = np.vstack((rel_overlaps, rel_overlaps))

    # TODO: choose these overlap smaller than BG_THRESH_HI_PHRASE or all phrase rois except fg_phrases
    bg_phrase_inds = np.where((cfg.TRAIN.MPN_BG_THRESH_HI_PHRASE > rel_overlaps.max(axis=1)) &
                              (rel_overlaps.max(axis=1) >= cfg.TRAIN.MPN_BG_THRESH_LO_PHRASE))[0]
    phrase_gt_assignment = rel_overlaps.argmax(axis=1)

    # ---------- get fg_phrase and bg_phrase ----------

    fg_phrase_inds, fg_phrase_gt_assignment = _get_fg_phrase_inds(
        obj_overlaps, obj_gt_assignment, rel_overlaps, subject_inds, object_inds, gt_relationships)
    phrase_gt_assignment[fg_phrase_inds] = fg_phrase_gt_assignment

    # get fg_phrase size
    fg_phrase_per_this_image = min(fg_phrase_rois_per_image, fg_phrase_inds.size)
    if fg_phrase_inds.size > 0:
        fg_phrase_inds = npr.choice(fg_phrase_inds, size=fg_phrase_per_this_image, replace=False)
    # get bg_phrase size
    bg_phrase_rois_per_this_image = phrase_rois_per_image - fg_phrase_inds.size
    if bg_phrase_inds.size > 0:
        bg_phrase_inds = npr.choice(bg_phrase_inds,
                                    size=min(bg_phrase_rois_per_this_image, bg_phrase_inds.size),
                                    replace=False)

    # set phrase label
    keep_phrase_inds = np.append(fg_phrase_inds, bg_phrase_inds)
    phrase_labels = gt_regions[phrase_gt_assignment[keep_phrase_inds], 4]
    phrase_labels[fg_phrase_inds.size:] = 0
    # ---------- get fg_object and bg_object ----------

    fg_object_inds, bg_object_inds = _get_fg_bg_object_inds(
        subject_inds, object_inds, fg_phrase_inds, bg_phrase_inds, fg_object_inds, bg_object_inds)

    fg_object_labels = obj_labels[fg_object_inds]
    object_labels = np.append(fg_object_labels, np.zeros(bg_object_inds.size))
    keep_object_inds = np.append(fg_object_inds, bg_object_inds)

    # ---------- get mat-pair ----------

    mat_object = _get_mat_object(
        keep_object_inds, keep_phrase_inds, subject_inds, object_inds, object_rois.shape[0])

    # ---------- get correspond rois and return -----------

    object_rois = object_rois[keep_object_inds]
    bbox_target_data_object = _compute_targets(
        object_rois[:, 1:5], gt_objects[obj_gt_assignment[keep_object_inds], :4], object_labels)
    bbox_targets_object, bbox_inside_weights_object = \
        _get_bbox_regression_labels(bbox_target_data_object, n_classes_obj)

    phrase_rois = phrase_rois[keep_phrase_inds]
    bbox_target_data_phrase = _compute_targets(
        phrase_rois[:, 1:5], gt_regions[phrase_gt_assignment[keep_phrase_inds], :4], phrase_labels)
    bbox_targets_phrase, bbox_inside_weights_phrase = \
        _get_bbox_regression_labels(bbox_target_data_phrase, n_classes_pred)

    return object_labels, object_rois, bbox_targets_object, bbox_inside_weights_object,\
           phrase_labels, phrase_rois, bbox_targets_phrase, bbox_inside_weights_phrase, mat_object


def _setup_connection(object_rois, phrase_rois, subject_inds, object_inds, graph_generation=False):

    # TEST.BBOX_NUM = object_rois.shape[0] = 512
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


def _get_fg_phrase_inds(obj_overlaps, obj_gt_assignment, rel_overlaps,
                    subject_inds, object_inds, gt_relationships):
    # get fg_phrase_inds

    # remove objects that overlaps < 0.5 with any gt_objects
    max_obj_overlaps = obj_overlaps.max(axis=1)
    obj_gt_assignment[np.where(max_obj_overlaps < cfg.TRAIN.MPN_FG_THRESH)] = -1
    pair_obj_ind = obj_gt_assignment[object_inds]
    pair_sub_ind = obj_gt_assignment[subject_inds]
    keep_pair_ind = np.logical_and(pair_obj_ind >= 0, pair_sub_ind >= 0)

    # ---------- get fg_phrase and bg_phrase ----------

    # set rel_overlaps == 0 that smaller than phrase_thres
    rel_overlaps[np.logical_not(keep_pair_ind)] = 0
    rel_overlaps[rel_overlaps < cfg.TRAIN.MPN_FG_THRESH_PHRASE] = 0
    # keep phrase rois that overlap >= FG_THRESH_phrase with any gt_relationship_boxes
    # and keep all mapping gt_relationship
    selected_phrase_rois_inds, rel_gt_assignment = np.where(rel_overlaps >= cfg.TRAIN.MPN_FG_THRESH_PHRASE)

    # keep sub_obj
    subject_selected = subject_inds[selected_phrase_rois_inds]
    object_selected = object_inds[selected_phrase_rois_inds]

    # get subject id and object id of each gt_relationship
    gt_rel_sub_idx, gt_rel_obj_idx = np.where(gt_relationships > 0)
    pair_sub_gt_assignment = gt_rel_sub_idx[rel_gt_assignment]
    pair_obj_gt_assignment = gt_rel_obj_idx[rel_gt_assignment]

    # get phrase keep inds
    obj_overlaps_bool = obj_overlaps >= cfg.TRAIN.MPN_FG_THRESH
    selected_fg_phrase_inds = np.logical_and(obj_overlaps_bool[subject_selected, pair_sub_gt_assignment],
                                             obj_overlaps_bool[object_selected, pair_obj_gt_assignment])
    selected_bg_phrase_inds = np.logical_not(selected_fg_phrase_inds)
    rel_overlaps[selected_phrase_rois_inds[selected_bg_phrase_inds],
                 rel_gt_assignment[selected_bg_phrase_inds]] = 0

    max_rel_overlaps = rel_overlaps.max(axis=1)
    fg_phrase_inds = np.where(max_rel_overlaps >= cfg.TRAIN.MPN_FG_THRESH_PHRASE)[0]
    # print(fg_phrase_inds.size)
    # bg_phrase_inds = np.where(max_rel_overlaps==0)[0]
    fg_phrase_gt_assignment = rel_overlaps.argmax(axis=1)[fg_phrase_inds]
    return fg_phrase_inds, fg_phrase_gt_assignment


def _get_fg_bg_object_inds(subject_inds, object_inds,
                           fg_phrase_inds, bg_phrase_inds, fg_object_inds, bg_object_inds, num_images=1):
    # get fg_object_inds and bg_object_inds
    rois_per_image = int(cfg.TRAIN.MPN_BATCH_SIZE / num_images)
    fg_rois_per_image = int(np.round(cfg.TRAIN.MPN_FG_FRACTION * rois_per_image))

    fg_object_inds_part = np.unique(np.append(subject_inds[fg_phrase_inds], object_inds[fg_phrase_inds]))
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
    bg_object_inds_part = np.unique(np.append(subject_inds[bg_phrase_inds], object_inds[bg_phrase_inds]))
    bg_object_per_this_image = min(rois_per_image - fg_object_inds.size, bg_object_inds.size)
    if bg_object_inds_part.size >= bg_object_per_this_image:
        bg_object_inds = npr.choice(bg_object_inds_part,
                                    size=bg_object_per_this_image,
                                    replace=False)
    else:
        bg_object_other_part = bg_object_per_this_image - bg_object_inds_part.size
        bg_object_inds = np.append(bg_object_inds_part,
                                   npr.choice(np.setdiff1d(bg_object_inds, bg_object_inds_part),
                                              size=bg_object_other_part, replace=False))
    return fg_object_inds, bg_object_inds


def _get_mat_object(keep_object_inds, keep_phrase_inds, subject_inds, object_inds, obj_rois_num):

    # mapping subject_inds from 0-obj_rois_num to 0-keep_object_inds.size
    # to get the sub_list and obj_list which keep relationships between object and relationship phrase
    keep_object_num = keep_object_inds.size
    object_mapping = np.arange(obj_rois_num)
    useless_object = np.setdiff1d(object_mapping, keep_object_inds)
    object_mapping[keep_object_inds] = np.arange(keep_object_num)
    object_mapping[useless_object] = keep_object_num

    sub_list = object_mapping[subject_inds[keep_phrase_inds]]
    obj_list = object_mapping[object_inds[keep_phrase_inds]]

    # objects * (sub or obj) * phrase(relationship proposal)
    mat_object = np.zeros((keep_object_inds.size, 2, keep_phrase_inds.size), dtype=np.int64)

    # different with MSDN, our phrase(or phrase in MSDN) may be not correspond to any sub or obj
    # so we have to prepare_message in message passing process just like objects
    # the select_mat in function prepare_message can use the transpose of mat_object
    for i in range(keep_phrase_inds.size):
        if sub_list[i] < keep_object_num:
            mat_object[sub_list[i], 0, i] = 1
        if obj_list[i] < keep_object_num:
            mat_object[obj_list[i], 1, i] = 1

    return mat_object
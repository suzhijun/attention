import os
import torch
import numpy as np
import time
import ipdb
from faster_rcnn import network
from faster_rcnn.RPN import RPN  # Hierarchical_Descriptive_Model
from faster_rcnn.utils.timer import Timer
from faster_rcnn.utils.HDN_utils import check_recall, check_obj_rel_recall, check_relationship_recall
from faster_rcnn.utils.make_cover import compare_rel_rois


from faster_rcnn.datasets.visual_genome_loader import visual_genome
from faster_rcnn.fast_rcnn.config import cfg
# from faster_rcnn.fast_rcnn.bbox_transform import enlarge_rois_clip
import argparse

import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser('Options for training RPN in pytorch')

## training settings
parser.add_argument('--lr', type=float, default=0.01, help='To disable the Lanuage Model ')
parser.add_argument('--max_epoch', type=int, default=5, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--log_interval', type=int, default=500, help='Interval for Logging')
parser.add_argument('--disable_clip_gradient', action='store_true', help='Whether to clip the gradient')
parser.add_argument('--use_kmeans_anchors', default=True, help='Whether to use kmeans anchors')
parser.add_argument('--step_size', type=int, default=2, help='step to decay the learning rate')

## Environment Settings
parser.add_argument('--pretrained_model', type=str, default='model/pretrained_models/VGG_imagenet.npy',
					help='Path for the to-evaluate model')
parser.add_argument('--dataset_option', type=str, default='small', help='The dataset to use (small | normal | fat)')
parser.add_argument('--output_dir', type=str, default='./output/RPN', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='RPN_relationship', help='model name for snapshot')
parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
args = parser.parse_args()


def main():
	global args
	print "Loading training set and testing set..."
	# train_set = visual_genome(args.dataset_option, 'train')
	test_set = visual_genome('small', 'test')
	# print test_set.num_object_classes
	# print test_set.num_predicate_classes
	print "Done."

	# train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
	net = RPN(args.use_kmeans_anchors)
	network.load_net('./output/RPN/RPN_relationship_best_kmeans.h5', net)
	# network.set_trainable(net.features, requires_grad=False)
	net.cuda()

	# best_recall = np.array([0.0, 0.0])

	# Testing
	recall = test(test_loader, net)

	print('Recall: '
		  'object: {recall[0]: .3f}%'
		  'relationship: {recall[1]: .3f}%'.format(recall=recall*100))


def test(test_loader, target_net):
	box_num = np.array([0, 0])
	correct_cnt, total_cnt = np.array([0, 0]), np.array([0, 0])
	print '========== Testing ======='
	target_net.eval()

	batch_time = network.AverageMeter()
	cover_cnt = 0

	cover_gt_cnt = 0
	fg_cover = 0
	fg_object = 0
	cover_gt = 0
	object_gt = 0
	num = 0
	end = time.time()
	for i, (im_data, im_info, gt_objects, gt_relationships, gt_boxes_relationship) in enumerate(test_loader):
		# num += 1
		# if num > 20:
		# 	break
		correct_cnt_t, total_cnt_t = np.array([0, 0]), np.array([0, 0])
		# Forward pass

		object_rois, relationship_rois, scores_object, scores_relationship = target_net(im_data, im_info.numpy(), gt_objects.numpy(),
													gt_boxes_relationship.numpy())[1:]

		# TODO: add rules
		# img_shape = im_info[0][:2]
		# object_rois = object_rois[:, 1:5]
		# relationship_rois = enlarge_rois_clip(relationship_rois[:, 1:5], 1.2, img_shape)
		# obj_in_predicate(object_rois, relationship_rois, 9)

		# subject_id, object_id, relationship_cover: Variable
		ipdb.set_trace()
		subject_id, object_id, relationship_cover = compare_rel_rois(
			object_rois.data.cpu().numpy(), relationship_rois.data.cpu().numpy(), scores_object, scores_relationship,
			topN_obj=256, topN_rel=96,
			obj_rel_thresh=0.6, max_objects=15, topN_covers=2048, cover_thresh=0.6)

		# print('relationship_cover size', relationship_cover.size())
		# unique_obj = np.unique(np.append(subject_id.cpu().numpy(), object_id.cpu().numpy()))
		# print('max', np.max(unique_obj))
		# print(unique_obj.size)

		cover_gt_num = check_recall(relationship_rois, gt_boxes_relationship.numpy()[0], top_N=relationship_rois.size(0), thresh=0.5)
		cover_cnt += cover_gt_num[0]
		# all_rois = object_rois.data.cpu().numpy()
		# zeros = np.zeros((gt_objects.numpy().shape[1], 1), dtype=gt_objects.numpy().dtype)
		# # add gt_obj to predict_rois
		# all_rois = np.vstack(
		# 	(all_rois, np.hstack((zeros, gt_objects.numpy()[0][:, :4])))
		# )
		# all_rois_phrase = relationship_rois.data.cpu().numpy()
		# zeros = np.zeros((gt_boxes_relationship.numpy().shape[1], 1), dtype=gt_boxes_relationship.numpy()[0].dtype)
		# all_rois_phrase = np.vstack(
		# 	(all_rois_phrase, np.hstack((zeros, gt_boxes_relationship.numpy()[0][:, :4])))
		# )
		# scores_object = np.append(scores_object, np.ones(gt_objects.size()[1], dtype=scores_object.dtype))
		# scores_relationship = np.append(scores_relationship, np.ones(gt_boxes_relationship.size()[1], dtype=scores_relationship.dtype))
		# subject_id, object_id, relationship_cover = compare_rel_rois(
		# 	all_rois, all_rois_phrase, scores_object, scores_relationship,
		# 	topN_obj=all_rois.shape[0], topN_rel=all_rois_phrase.shape[0],
		# 	obj_rel_thresh=0.6, max_objects=18, topN_covers=2048, cover_thresh=0.6)

		# all_rois_phrase_all = relationship_cover
		# zeros = np.zeros((gt_boxes_relationship.numpy().shape[1], 1), dtype=gt_boxes_relationship.numpy()[0].dtype)
		# all_rois_phrase_all = np.vstack(
		# 	(all_rois_phrase_all, np.hstack((zeros, gt_boxes_relationship.numpy()[0][:, :4])))
		# )
		# gt_rel_sub_idx, gt_rel_obj_idx = np.where(gt_relationships.numpy()[0] > 0)
		# gt_rel_sub_idx, gt_rel_obj_idx = gt_rel_sub_idx + object_rois.size()[0], gt_rel_obj_idx + object_rois.size()[0]
		# subject_inds = np.append(subject_id, gt_rel_sub_idx)
		# object_inds = np.append(object_id, gt_rel_obj_idx)
		cover_obj_check = check_obj_rel_recall(gt_objects.numpy()[0], gt_relationships.numpy()[0], gt_boxes_relationship.numpy()[0],
											   relationship_cover, object_rois.data.cpu().numpy()[:256, :],
											   # all_rois_phrase, all_rois,
											   subject_id, object_id,
											   # subject_inds, object_inds,
											   cover_thresh=0.5, object_thresh=0.5, log=False)
		cover_gt_cnt += cover_obj_check[0]
		fg_cover += cover_obj_check[1]
		fg_object += cover_obj_check[2]
		cover_gt += cover_obj_check[3]
		object_gt += cover_obj_check[4]

		# test
		# object_labels, object_rois_pro, bbox_targets_object, bbox_inside_weights_object, bbox_outside_weights_object, \
		# region_labels, region_rois_pro, bbox_targets_region, bbox_inside_weights_region, bbox_outside_weights_region, \
		# mat_object = proposal_target_layer(object_rois.data.cpu().numpy(), relationship_cover.data.cpu().numpy(),
		# 								   subject_id.cpu().numpy(), object_id.cpu().numpy(),
		# 								   gt_objects.numpy()[0], gt_relationships.numpy()[0], gt_boxes_relationship.numpy()[0],
		# 								   n_classes_obj=151, n_classes_pred=51, is_training=False, graph_generation=False)
		# print('object_labels', object_labels[:20])
		# print(object_labels.shape)
		# print('bbox_targets_object', bbox_targets_object[:10, :60])
		# print(bbox_targets_object.shape)
		# print('bbox_inside_weights_object', bbox_inside_weights_object[:10, :60])
		# print(bbox_inside_weights_object.shape)
		# print('region \n')
		# print('region_labels', region_labels[:20])
		# print(region_labels.shape)
		# print('region_rois_pro', region_rois_pro[:10])
		# print(region_rois_pro.shape)
		# print('bbox_targets_region', bbox_targets_region[:10])
		# print(bbox_targets_region.shape)
		# print('bbox_inside_weights_region', bbox_inside_weights_region[:10, :60])
		# print(bbox_inside_weights_region.shape)
		# print('object_rois_pro', object_rois_pro[:10])
		# print('object_rois_pro shape', object_rois_pro.shape)
		# print('region_rois_pro', region_rois_pro[:10])
		# print(region_rois_pro.shape)
		# print('mat \n')
		# print(mat_object[:32, 0, :32])
		# print(mat_object.shape)
		# print(mat_object[:32, 0, :32].T)

		box_num[0] += object_rois.size(0)
		box_num[1] += relationship_rois.size(0)
		correct_cnt_t[0], total_cnt_t[0] = check_recall(object_rois, gt_objects.numpy()[0], 256, thresh=0.5)
		correct_cnt_t[1], total_cnt_t[1] = check_recall(relationship_rois, gt_boxes_relationship.numpy()[0], 96, thresh=0.5)
		correct_cnt += correct_cnt_t
		total_cnt += total_cnt_t
		batch_time.update(time.time()-end)
		end = time.time()
		if (i+1)%100 == 0 and i > 0:
			print('([{0}/{10}]  Time: {1:2.3f}s/img).\n'
				  '[object] Avg: {2:2.2f} Boxes/im, Top-256 recall: {3:2.3f} ({4:d}/{5:d})\n'
				  '[relationship] Avg: {6:2.2f} Boxes/im, Top-96 recall: {7:2.3f} ({8:d}/{9:d})'.format(
				i+1, batch_time.avg,
				box_num[0]/float(i+1), correct_cnt[0]/float(total_cnt[0])*100, correct_cnt[0], total_cnt[0],
				box_num[1]/float(i+1), correct_cnt[1]/float(total_cnt[1])*100, correct_cnt[1], total_cnt[1],
				len(test_loader)))
			print('[relationship_cover number]: {0}\n'
				  '[cover vs gt_relationship_boxes average recall]: {1:.3f}\n'
				  '[cover & sub & obj vs gt_relationship_boxes average recall]: {2:.3f}').format(
				relationship_cover.shape[0], cover_cnt/float(total_cnt[1])*100, cover_gt_cnt/float(total_cnt[1])*100)
			print('average fg_cover: {0:.2f}'
				  '\taverage fg_object: {1:.2f}'
				  '\taverage cover_gt: {2:.2f}'
				  '\taverage object_gt: {3:.2f}').format(
				fg_cover / float(i), fg_object / float(i), cover_gt / float(i), object_gt / float(i))

	recall = correct_cnt/total_cnt.astype(np.float)
	print '====== Done Testing ===='
	return recall


if __name__ == '__main__':
	main()
import os
import torch
import numpy as np
import time
from faster_rcnn import network
from faster_rcnn.RPN import RPN
from faster_rcnn.utils.HDN_utils import check_recall

from faster_rcnn.datasets.visual_genome_loader import visual_genome
import argparse



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
	print "Loading testing set..."
	# train_set = visual_genome(args.dataset_option, 'train')
	test_set = visual_genome('small', 'test')
	print "Done."

	test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
	net = RPN(args.use_kmeans_anchors)
	network.load_net('./output/RPN/RPN_relationship_best_kmeans.h5', net)
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
	# cover_cnt = 0
	#
	# cover_gt_cnt = 0
	# fg_cover = 0
	# fg_object = 0
	# cover_gt = 0
	# object_gt = 0
	# num = 0
	end = time.time()
	for i, (im_data, im_info, gt_objects, gt_relationships, gt_boxes_relationship) in enumerate(test_loader):
		# num += 1
		# if num > 320:
		# 	break
		correct_cnt_t, total_cnt_t = np.array([0, 0]), np.array([0, 0])
		# Forward pass

		object_rois, relationship_rois, scores_object, scores_relationship = target_net(im_data, im_info.numpy(), gt_objects.numpy()[0],
		                                            gt_boxes_relationship.numpy()[0])[1:]

		# TODO: add rules


		# subject_id, object_id, relationship_cover = compare_rel_rois(
		# 	object_rois, relationship_rois, scores_object, scores_relationship, topN_covers=2048, thresh=0.5)
		#
		# cover_gt_num = check_recall(relationship_cover, gt_boxes_relationship[0].numpy(),
		# 							 top_N=relationship_cover.size()[0])
		# cover_cnt += cover_gt_num[0]
		# cover_obj_check = check_obj_rel_recall(gt_objects[0].numpy(), gt_relationships[0].numpy(),
		# 									 gt_boxes_relationship[0].numpy(), relationship_cover,
		# 									 subject_id.cpu().numpy(), object_id.cpu().numpy(),
		# 									 object_rois, cover_thresh=0.4, object_thresh=0.4, log=num)
		# cover_gt_cnt += cover_obj_check[0]
		# fg_cover += cover_obj_check[1]
		# fg_object += cover_obj_check[2]
		# cover_gt += cover_obj_check[3]
		# object_gt += cover_obj_check[4]
		#
		box_num[0] += object_rois.size(0)
		box_num[1] += relationship_rois.size(0)
		correct_cnt_t[0], total_cnt_t[0] = check_recall(object_rois, gt_objects[0].numpy(), 256, thresh=0.5)
		correct_cnt_t[1], total_cnt_t[1] = check_recall(relationship_rois, gt_boxes_relationship[0].numpy(), 256, thresh=0.5)
		correct_cnt += correct_cnt_t
		total_cnt += total_cnt_t
		batch_time.update(time.time()-end)
		end = time.time()
		if (i+1)%100 == 0 and i > 0:
			print('[{0}/{10}]  Time: {1:2.3f}s/img).'
			      '\t[object] Avg: {2:2.2f} Boxes/im, Top-256 recall: {3:2.3f} ({4:d}/{5:d})'
			      '\t[relationship] Avg: {6:2.2f} Boxes/im, Top-256 recall: {7:2.3f} ({8:d}/{9:d})'.format(
				i+1, batch_time.avg,
				box_num[0]/float(i+1), correct_cnt[0]/float(total_cnt[0])*100, correct_cnt[0], total_cnt[0],
				box_num[1]/float(i+1), correct_cnt[1]/float(total_cnt[1])*100, correct_cnt[1], total_cnt[1],
				len(test_loader)))
			# print('relationship_cover number: {0}'
			# 	  '\tcover vs gt_relationship_boxes average recall: {1:.3f}'
			# 	  '\tcover & sub & obj vs gt_relationship_boxes average recall: {2:.3f}').format(
			# 	relationship_cover.size()[0], cover_cnt/float(total_cnt[1])*100, cover_gt_cnt/float(total_cnt[1])*100)
			# print('average fg_cover: {0:.2f}'
			# 	  '\taverage fg_object: {1:.2f}'
			# 	  '\taverage cover_gt: {2:.2f}'
			# 	  '\taverage object_gt: {3:.2f}').format(
			# 	fg_cover / float(i), fg_object / float(i), cover_gt / float(i), object_gt / float(i))

	recall = correct_cnt/total_cnt.astype(np.float)
	print '====== Done Testing ===='
	return recall


if __name__ == '__main__':
	main()

import os
import torch
import numpy as np
import time
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN  # Hierarchical_Descriptive_Model
from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.fast_rcnn.nms_wrapper import nms
from faster_rcnn.utils.timer import Timer
from faster_rcnn.utils.HDN_utils import check_recall

from faster_rcnn.datasets.visual_genome_object import visual_genome
from faster_rcnn.fast_rcnn.config import cfg
import argparse


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser('Options for training RPN in pytorch')
parser.add_argument('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors')
## Environment Settings
parser.add_argument('--pretrained_model', type=str, default='model/pretrained_models/VGG_imagenet.npy',
                    help='Path for the to-evaluate model')
parser.add_argument('--dataset_option', type=str, default='small', help='The dataset to use (small | normal | fat)')
parser.add_argument('--output_dir', type=str, default='./output/detection', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='Faster_RCNN', help='model name for snapshot')
parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
parser.add_argument('--resume_model', type=str, default='./output/detection/RPN_object1_best.h5', help='The model we resume')
args = parser.parse_args()


def main():
	global args
	print "Loading training set and testing set..."
	# train_set = visual_genome(args.dataset_option, 'train')
	test_set = visual_genome(args.dataset_option, 'test')
	object_classes = test_set.object_classes
	num_classes = len(object_classes)
	print "Done."

	# train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
	net = FasterRCNN(use_kmeans_anchors=args.use_normal_anchors, classes=object_classes)
	network.load_net(args.resume_model, net)
	# network.set_trainable(net.features, requires_grad=False)
	net.cuda()


	# Testing
	recall = test(test_loader, net, num_classes)

	print('Recall: '
	      'object: {recall[0]: .3f}%'
	      'relationship: {recall[1]: .3f}%'.format(recall=recall*100))


def test(test_loader, target_net, num_classes):

	box_num, correct_cnt, total_cnt = 0, 0, 0
	tp, tf, fg, bg = 0., 0., 0, 0
	thresh = 0.05
	max_per_image = 300
	print '========== Testing ======='
	target_net.eval()

	batch_time = network.AverageMeter()
	end = time.time()
	for i, (im_data, im_info, gt_boxes) in enumerate(test_loader):
		cls_prob, bbox_pred, rois = target_net(im_data, im_info.numpy(), gt_boxes.numpy()[0])
		target_net.interpret_faster_rcnn(cls_prob, bbox_pred, rois, im_info, im_shape, nms=True, clip=True, min_score=0.0)
		scores = cls_prob.data.cpu().numpy()
		boxes = rois.data.cpu().numpy()[:, 1:5]/im_info[0][2]
		box_deltas = bbox_pred.data.cpu().numpy()
		pred_boxes = bbox_transform_inv(boxes, box_deltas)
		pred_boxes = clip_boxes(pred_boxes, im_data.shape)
		all_boxes = []
		for j in xrange(1, num_classes):
			inds = np.where(scores[:, j] > thresh)[0]
			cls_scores = scores[inds, j]
			cls_boxes = pred_boxes[inds, j*4:(j+1)*4]
			cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
				.astype(np.float32, copy=False)
			keep = nms(cls_dets, cfg.TEST.NMS)
			cls_dets = cls_dets[keep, :]
			all_boxes[j] = cls_dets
		if max_per_image > 0:
			image_scores = np.hstack([all_boxes[j][:, -1]
			                          for j in xrange(1, num_classes)])
			if len(image_scores) > max_per_image:
				image_thresh = np.sort(image_scores)[-max_per_image]
				for j in xrange(1, num_classes):
					keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
					all_boxes[j] = all_boxes[j][keep, :]
		print(all_boxes)
		# Forward pass
		# features, object_rois, scores = target_net.rpn(im_data, im_info.numpy(), gt_boxes.numpy()[0])
		# box_num+= object_rois.size(0)
		# correct_cnt_t, total_cnt_t = check_recall(object_rois, gt_boxes.numpy()[0], 1000)
		# correct_cnt += correct_cnt_t
		# total_cnt += total_cnt_t
		# batch_time.update(time.time()-end)
		# end = time.time()
		# tp += float(target_net.tp)
		# tf += float(target_net.tf)
		# fg += target_net.fg_cnt
		# bg += target_net.bg_cnt
		if (i+1)%100 == 0 and i > 0:
			print('[{0}/{6}]  Time: {1:2.3f}s/img).'
			      '\t[object] Avg: {2:2.2f} Boxes/im, Top-1000 recall: {3:2.3f} ({4:d}/{5:d})'.format(
				i+1, batch_time.avg,box_num/float(i+1), correct_cnt/float(total_cnt)*100,
				correct_cnt, total_cnt, len(test_loader)))
			print('\tTP: %.3f%%, TF: %.3f%%, fg/bg=(%d/%d)'%(tp/fg*100., tf/bg*100., fg, bg))

	recall = correct_cnt/float(total_cnt)
	print '====== Done Testing ===='
	return recall


if __name__ == '__main__':
	main()

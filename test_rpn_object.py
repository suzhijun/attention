import os
import torch
import numpy as np
import time
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN  # Hierarchical_Descriptive_Model
from faster_rcnn.utils.HDN_utils import check_recall

from faster_rcnn.datasets.visual_genome_loader import visual_genome
from faster_rcnn.fast_rcnn.config import cfg
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
parser.add_argument('--base_model', type=str, default='vgg', help='base model: vgg or resnet50 or resnet101')

## Environment Settings
parser.add_argument('--pretrained_model', type=str, default='model/pretrained_models/VGG_imagenet.npy',
                    help='Path for the to-evaluate model')
parser.add_argument('--dataset_option', type=str, default='small', help='The dataset to use (small | normal | fat)')
parser.add_argument('--model_name', type=str, default='RPN_relationship', help='model name for snapshot')
parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
args = parser.parse_args()


def main():
	global args
	print "Loading training set and testing set..."
	# train_set = visual_genome(args.dataset_option, 'train')
	test_set = visual_genome('small', 'test')
	object_classes = test_set.object_classes
	print "Done."

	# train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
	net = FasterRCNN(use_kmeans_anchors=args.use_kmeans_anchors, n_classes=len(object_classes), model=args.base_model)
	network.load_net('./output/detection/Faster_RCNN_small_vgg_12epoch_epoch_11.h5', net)
	# network.load_net('./output/detection/RPN_object1_best.h5', net)
	# network.set_trainable(net.features, requires_grad=False)
	net.cuda()

	# Testing
	recall = test(test_loader, net)

	print('Recall: '
	      'object: {recall: .3f}%'.format(recall=recall*100))


def test(test_loader, target_net):

	box_num, correct_cnt, total_cnt = 0, 0, 0
	print '========== Testing ======='
	target_net.eval()

	batch_time = network.AverageMeter()
	end = time.time()
	for i, (im_data, im_info, gt_boxes, gt_relationship) in enumerate(test_loader):
		# Forward pass
		features, object_rois, scores = target_net.rpn(im_data, im_info.numpy(), gt_boxes.numpy()[0])
		box_num += object_rois.size(0)
		correct_cnt_t, total_cnt_t = check_recall(object_rois, gt_boxes.numpy()[0], 64)
		correct_cnt += correct_cnt_t
		total_cnt += total_cnt_t
		batch_time.update(time.time()-end)
		end = time.time()
		if (i+1)%100 == 0 and i > 0:
			print('[{0}/{6}]  Time: {1:2.3f}s/img).'
			      '\t[object] Avg: {2:2.2f} Boxes/im, Top-64 recall: {3:2.3f} ({4:d}/{5:d})'.format(
				i+1, batch_time.avg, box_num/float(i+1), correct_cnt/float(total_cnt)*100,
				correct_cnt, total_cnt, len(test_loader)))

	recall = correct_cnt/float(total_cnt)
	print '====== Done Testing ===='
	return recall


if __name__ == '__main__':
	main()

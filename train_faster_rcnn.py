import os
import torch
import numpy as np
import time
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN  # Hierarchical_Descriptive_Model
from faster_rcnn.utils.timer import Timer
from faster_rcnn.utils.HDN_utils import check_recall

from faster_rcnn.datasets.visual_genome_loader import visual_genome
from faster_rcnn.utils.map_eval import cls_eval, image_eval
import argparse

import pdb

parser = argparse.ArgumentParser('Options for training RPN in pytorch')
parser.add_argument('--gpu', type=str, default='0', help='GPU id')
## training settings
parser.add_argument('--lr', type=float, default=0.01, help='To disable the Lanuage Model ')
parser.add_argument('--max_epoch', type=int, default=12, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.95, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--log_interval', type=int, default=10, help='Interval for Logging')
parser.add_argument('--clip_gradient', action='store_true', help='Whether to clip the gradient')
parser.add_argument('--use_kmeans_anchors', default=True, help='Whether to use kmeans anchors')
parser.add_argument('--step_size', type=int, default=3, help='step to decay the learning rate')
parser.add_argument('--base_model', type=str, default='resnet101', help='base model: vgg or resnet50 or resnet101')

## Environment Settings
parser.add_argument('--pretrained_model', type=str, default='model/pretrained_models/VGG_imagenet.npy',
					help='Path for the to-evaluate model')
parser.add_argument('--dataset_option', type=str, default='small', help='The dataset to use (small | normal | fat)')
parser.add_argument('--output_dir', type=str, default='./output/detection', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='Faster_RCNN_resnet101', help='model name for snapshot')
parser.add_argument('--resume_model', action='store_true', help='Resume model from the entire model')
parser.add_argument('--detection_model', default='./output/detection/HDN_2_iters_alt_small_SGD_epoch_0.h5', help='The model used for resuming entire training')
# parser.add_argument('--pretrain', default=True, help='Resume training from RPN')
# parser.add_argument('--pretrained_model', type=str, default='./output/pretrained_model/resnet101_vg.pth', help='The Model used for resuming from RPN')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def main():
	global args
	print "Loading training set and testing set..."
	train_set = visual_genome(args.dataset_option, 'train')
	test_set = visual_genome(args.dataset_option, 'test')
	object_classes = test_set.object_classes
	print "Done."

	train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
	net = FasterRCNN(use_kmeans_anchors=args.use_kmeans_anchors, n_classes=len(object_classes), model=args.base_model)
	if args.resume_model:
		print 'Resume training from: {}'.format(args.resume_model)
		if len(args.resume_model) == 0:
			raise Exception('[resume_model] not specified')
		network.load_net(args.detection_model, net)
		# optimizer = torch.optim.SGD([
		# 	{'params': list(net.parameters())},
		# ], lr=args.lr, momentum=args.momentum, weight_decay=0.0005)
	else:
		print 'Training from scratch...Initializing network...'
	optimizer = torch.optim.SGD(list(net.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

	# network.set_trainable(net.features, requires_grad=True)
	net.cuda()

	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)

	best_map = 0.0

	for epoch in range(0, args.max_epoch):
		# Training
		train(train_loader, net, optimizer, epoch)

		# update learning rate
		if epoch%args.step_size == args.step_size-1:
			args.clip_gradient = False
			args.lr /= 5
			for param_group in optimizer.param_groups:
				param_group['lr'] = args.lr

		save_name = os.path.join(args.output_dir, '{}_epoch_{}.h5'.format(args.model_name, epoch))
		network.save_net(save_name, net)
		print('save model: {}'.format(save_name))

		try:
		# Testing
			map = evaluate(test_loader, net, object_classes)
			print('Epoch[{epoch:d}]: '
				  'Recall: '
				  'object: {map: .3f}%% (Best: {best_map: .3f}%%)'.format(
				epoch=epoch, map=map*100, best_map=best_map*100))
			if map > best_map:
				best_map = map
				save_name = os.path.join(args.output_dir, '{}_best.h5'.format(args.model_name, epoch))
				network.save_net(save_name, net)
		except:
			continue


def train(train_loader, target_net, optimizer, epoch):
	batch_time = network.AverageMeter()
	data_time = network.AverageMeter()
	train_loss = network.AverageMeter()
	train_loss_box_rpn = network.AverageMeter()
	train_loss_entropy_rpn = network.AverageMeter()
	train_loss_box_det = network.AverageMeter()
	train_loss_entropy_det = network.AverageMeter()
	tp, tf, fg, bg = 0., 0., 0, 0

	target_net.train()
	end = time.time()
	for i, (im_data, im_info, gt_boxes, gt_relationships) in enumerate(train_loader):
		# measure the data loading time
		data_time.update(time.time()-end)

		# Forward pass
		target_net(im_data, im_info.numpy(), gt_boxes.numpy()[0])
		# record loss
		loss = target_net.loss + target_net.rpn.loss
		# total loss
		train_loss.update(loss.data[0], im_data.size(0))
		train_loss_box_det.update(target_net.loss_box.data[0], im_data.size(0))
		train_loss_entropy_det.update(target_net.cross_entropy.data[0], im_data.size(0))
		train_loss_box_rpn.update(target_net.rpn.loss_box.data[0], im_data.size(0))
		train_loss_entropy_rpn.update(target_net.rpn.cross_entropy.data[0], im_data.size(0))

		tp += float(target_net.tp)
		tf += float(target_net.tf)
		fg += target_net.fg_cnt
		bg += target_net.bg_cnt

		# backward
		optimizer.zero_grad()
		loss.backward()
		if args.clip_gradient:
			network.clip_gradient(target_net, 10.)
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time()-end)
		end = time.time()


		if (i+1)%args.log_interval == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Batch_Time: {batch_time.avg:.3f}s\t'
				  'lr: {lr: f}\t'
				  'Loss: {loss.avg:.4f}\n'
				  '\t[rpn]: '
				  'cls_loss_rpn: {cls_loss_rpn.avg:.3f}\t'
				  'reg_loss_rpn: {reg_loss_rpn.avg:.3f}\n'
				  '\t[rcnn]: '
				  'cls_loss_rcnn: {cls_loss_rcnn.avg:.3f}\t'
				  'reg_loss_rcnn: {reg_loss_rcnn.avg:.3f}'
				.format(epoch, i+1, len(train_loader), batch_time=batch_time, lr=args.lr, data_time=data_time,
				loss=train_loss, cls_loss_rpn=train_loss_entropy_rpn, cls_loss_rcnn=train_loss_entropy_det,
				reg_loss_rpn=train_loss_box_rpn, reg_loss_rcnn=train_loss_box_det))

			print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)'%(tp/fg*100., tf/bg*100., fg, bg))


def evaluate(test_loader, target_net, object_classes, score_thresh=0.00, overlap_thresh=0.5, nms_thresh=0.5):
    print(object_classes)
    box_num, correct_cnt, total_cnt = 0, 0, 0
    # ipdb.set_trace()
    # store cls_scores, cls_tp, cls_gt_num of each cls and every image
    all_cls_gt_num = [0] * (len(object_classes) - 1)
    all_cls_scores = [np.array([])]*(len(object_classes) - 1)
    all_cls_tp = [np.array([])]*(len(object_classes) - 1)

    print('========== Evaluate =======')
    target_net.eval()

    batch_time = network.AverageMeter()
    end = time.time()
    for i, (im_data, im_info, gt_boxes, gt_relationships) in enumerate(test_loader):
        # get every class scores, tf array and gt number
        classes_scores, classes_tf, classes_gt_num, object_rois = \
            image_eval(target_net, im_data, im_info, gt_boxes.numpy()[0], object_classes,
                       max_per_image=100, score_thresh=score_thresh, overlap_thresh=overlap_thresh, nms_thresh=nms_thresh)

        for j in range(len(object_classes)-1):
            all_cls_scores[j] = np.append(all_cls_scores[j], classes_scores[j])
            all_cls_tp[j] = np.append(all_cls_tp[j], classes_tf[j])
            all_cls_gt_num[j] += classes_gt_num[j]


        box_num += object_rois.size(0)
        correct_cnt_t, total_cnt_t = check_recall(object_rois, gt_boxes.numpy()[0], 64, thresh=0.5)
        correct_cnt += correct_cnt_t
        total_cnt += total_cnt_t
        batch_time.update(time.time() - end)
        end = time.time()
        if (i+1)%500 == 0 and i > 0:
	        print('[{0}/{6}]  Time: {1:2.3f}s/img).'
	              '\t[object] Avg: {2:2.2f} Boxes/im, Top-64 recall: {3:2.3f} ({4:d}/{5:d})'.format(
		        i+1, batch_time.avg, box_num/float(i+1), correct_cnt/float(total_cnt)*100,
		        correct_cnt, total_cnt, len(test_loader)))

    all_aps = []
    for k, cls in enumerate(object_classes[1:]):
        # sort scores of all images
        cls_ap = cls_eval(all_cls_scores[k], all_cls_tp[k], all_cls_gt_num[k])
        all_aps += [cls_ap]
        print('AP for {} = {:.4f}'.format(cls, cls_ap))

	mean_ap = np.mean(all_aps)
    print('Mean AP = {:.4f}'.format(mean_ap))

    print('====== Done Testing ====')
    return mean_ap


if __name__ == '__main__':
	main()

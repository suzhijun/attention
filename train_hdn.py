import os
import shutil
import time
import random
import numpy as np
import numpy.random as npr
import argparse

import torch
import pdb

from faster_rcnn import network
from faster_rcnn.MSDN import Hierarchical_Descriptive_Model
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.config import cfg
from faster_rcnn.datasets.visual_genome_loader import visual_genome
from faster_rcnn.utils.HDN_utils import get_model_name, group_features, check_recall
from faster_rcnn.utils.map_eval import cls_eval

TIME_IT = False
parser = argparse.ArgumentParser('Options for training Hierarchical Descriptive Model in pytorch')

parser.add_argument('--gpu', type=str, default='0', help='GPU id')
# Training parameters
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='base learning rate for training')
parser.add_argument('--max_epoch', type=int, default=10, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--log_interval', type=int, default=1000, help='Interval for Logging')
parser.add_argument('--step_size', type=int, default=6, help='Step size for reduce learning rate')

# structure settings
parser.add_argument('--resume_model', action='store_true', help='Resume model from the entire model')
parser.add_argument('--HDN_model', default='./output/HDN/HDN_1_iters_all_vgg_2048_kmeans_RCNN_SGD_best.h5', help='The model used for resuming entire training')
parser.add_argument('--load_RCNN', default=True, help='Resume training from RCNN')
parser.add_argument('--RCNN_model', type=str, default = './output/detection/Faster_RCNN_vgg_12epoch_2048_best.h5', help='The Model used for resuming from RCNN')
parser.add_argument('--load_RPN', default=False, help='Resume training from RPN')
parser.add_argument('--RPN_model', type=str, default = './output/RPN/RPN_relationship_best_kmeans.h5', help='The Model used for resuming from RPN')

parser.add_argument('--enable_clip_gradient', action='store_true', help='Whether to clip the gradient')
parser.add_argument('--use_kmeans_anchors', default=True, help='Whether to use kmeans anchors')
parser.add_argument('--mps_feature_len', type=int, default=2048, help='The expected feature length of message passing')
parser.add_argument('--dropout', action='store_true', help='To enables the dropout')
parser.add_argument('--MPS_iter', type=int, default=1, help='Iterations for Message Passing')
parser.add_argument('--base_model', type=str, default='vgg', help='base model: vgg or resnet50 or resnet101')

# Environment Settings
parser.add_argument('--train_all', default=False, help='Train all the mode')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--dataset_option', type=str, default='all', help='The dataset to use (small | normal | fat | all)')
parser.add_argument('--output_dir', type=str, default='./output/HDN', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='HDN', help='The name for saving model.')
parser.add_argument('--nesterov', action='store_true', help='Set to use the nesterov for SGD')
parser.add_argument('--optimizer', type=int, default=0, help='which optimizer used for optimize model [0: SGD | 1: Adam | 2: Adagrad]')
parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
parser.add_argument('--use_rpn_scores', action='store_true', help='Use rpn scores to help to predict')
parser.add_argument('--use_gt_boxes', action='store_true', help='Use GT boxes')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# Overall loss logger
overall_train_loss = network.AverageMeter()
overall_train_rpn_loss = network.AverageMeter()

optimizer_select = 2
# normal_test = False

def main():
	global args, optimizer_select
	# To set the model name automatically
	print args
	lr = args.lr
	args = get_model_name(args)
	print 'Model name: {}'.format(args.model_name)

	# To set the random seed
	random.seed(args.seed)
	torch.manual_seed(args.seed + 1)
	torch.cuda.manual_seed(args.seed + 2)

	print("Loading training set and testing set...")
	train_set = visual_genome(args.dataset_option, 'train')
	test_set = visual_genome(args.dataset_option, 'test')
	print("Done.")

	train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

	net = Hierarchical_Descriptive_Model(nhidden=args.mps_feature_len,
				 n_object_cats=train_set.num_object_classes,
				 n_predicate_cats=train_set.num_predicate_classes,
				 MPS_iter=args.MPS_iter,
				 object_loss_weight=train_set.inverse_weight_object,
				 predicate_loss_weight=train_set.inverse_weight_predicate,
				 dropout=args.dropout,
				 use_kmeans_anchors=args.use_kmeans_anchors,
	             base_model=args.base_model) #True

	# params = list(net.parameters())
	# for param in params:
	#     print param.size()
	print net

	# Setting the state of the training model
	net.cuda()
	net.train()
	network.set_trainable(net, False)
	# network.weights_normal_init(net, dev=0.01)


	if args.resume_model:
		print 'Resume training from: {}'.format(args.HDN_model)
		if len(args.HDN_model) == 0:
			raise Exception('[resume_model] not specified')
		network.load_net(args.HDN_model, net)
		# network.load_net(args.RPN_model, net.rpn)
		args.train_all = True
		optimizer_select = 3

	elif args.load_RCNN:
		print 'Loading pretrained RCNN: {}'.format(args.RCNN_model)
		args.train_all = False
		network.load_net(args.RCNN_model, net.rcnn)
		optimizer_select = 2

	elif args.load_RPN:
		print 'Loading pretrained RPN: {}'.format(args.RPN_model)
		args.train_all = False
		network.load_net(args.RPN_model, net.rpn)
		net.reinitialize_fc_layers()
		optimizer_select = 1

	else:
		print 'Training from scratch.'
		net.rpn.initialize_parameters()
		net.reinitialize_fc_layers()
		optimizer_select = 0
		args.train_all = True

	# To group up the features
	# vgg_features_fix, vgg_features_var, rpn_features, hdn_features = group_features(net)
	basenet_features, rpn_features, rcnn_feature, hdn_features = group_features(net)
	optimizer = network.get_optimizer(lr, optimizer_select, args,
	                                  basenet_features, rpn_features, rcnn_feature, hdn_features)


	target_net = net
	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)


	top_Ns = [50, 100]
	best_recall = np.zeros(len(top_Ns))


	if args.evaluate:
		recall = test(test_loader, target_net, top_Ns, train_set.object_classes)
		print('======= Testing Result =======')
		for idx, top_N in enumerate(top_Ns):
			print('[Recall@{top_N:d}] {recall:2.3f}%% (best: {best_recall:2.3f}%%)'.format(
				top_N=top_N, recall=recall[idx] * 100, best_recall=best_recall[idx] * 100))

		print('==============================')
	else:
		for epoch in range(0, args.max_epoch):
			# Training
			train(train_loader, target_net, optimizer, epoch)
			# snapshot the state
			save_name = os.path.join(args.output_dir, '{}_epoch_{}.h5'.format(args.model_name, epoch))
			network.save_net(save_name, net)
			print('save model: {}'.format(save_name))

			recall = test(test_loader, target_net, top_Ns, train_set.object_classes)

			if np.all(recall > best_recall):
				best_recall = recall
				save_name = os.path.join(args.output_dir, '{}_best.h5'.format(args.model_name))
				network.save_net(save_name, net)
				print('\nsave model: {}'.format(save_name))

			print('Epoch[{epoch:d}]:'.format(epoch = epoch)),
			for idx, top_N in enumerate(top_Ns):
				print('\t[Recall@{top_N:d}] {recall:2.3f}%% (best: {best_recall:2.3f}%%)'.format(
					top_N=top_N, recall=recall[idx] * 100, best_recall=best_recall[idx] * 100))

			# updating learning policy
			if (epoch+1)%args.step_size==0 or (epoch+1)%(args.step_size+2)==0:
				lr /= 10
				args.lr = lr
				print '[learning rate: {}]'.format(lr)

				args.enable_clip_gradient = False
				args.train_all = False
				optimizer_select = 2
				# update optimizer and correponding requires_grad state
				optimizer = network.get_optimizer(lr, optimizer_select, args,
				                                  basenet_features, rpn_features, rcnn_feature, hdn_features)


def train(train_loader, target_net, optimizer, epoch):
	global args
	# Overall loss logger
	# global overall_train_loss
	# global overall_train_rpn_loss
	# global overall_train_region_caption_loss

	batch_time = network.AverageMeter()
	data_time = network.AverageMeter()
	# Total loss
	train_loss = network.AverageMeter()
	train_rpn_loss = network.AverageMeter()
	# object related loss
	train_rcnn_loss = network.AverageMeter()  # pre_mps_obj_cls_loss
	train_post_mps_obj_cls_loss = network.AverageMeter()
	# train_obj_box_loss = network.AverageMeter()
	# relationship cls loss
	train_pre_mps_pred_cls_loss = network.AverageMeter()
	train_post_mps_pred_cls_loss = network.AverageMeter()
	# train_pred_box_loss = network.AverageMeter()


	# accuracy
	accuracy_obj_pre_mps = network.AccuracyMeter()
	accuracy_pred_pre_mps = network.AccuracyMeter()
	accuracy_obj_post_mps = network.AccuracyMeter()
	accuracy_pred_post_mps = network.AccuracyMeter()

	target_net.train()
	end = time.time()
	for i, (im_data, im_info, gt_objects, gt_relationships) in enumerate(train_loader):
		data_time.update(time.time() - end)
		t0 = time.time()

		target_net(im_data, im_info, gt_objects.numpy()[0], gt_relationships.numpy()[0])

		if TIME_IT:
			t1 = time.time()
			print('forward time %.3fs')%(t1-t0)


		# Determine the loss function
		if args.train_all:
			loss = target_net.loss + target_net.rcnn.loss + target_net.rcnn.rpn.loss
		else:
			loss = target_net.loss


		train_loss.update(loss.data.cpu().numpy()[0], im_data.size(0))
		train_rcnn_loss.update(target_net.rcnn.loss.data.cpu().numpy()[0], im_data.size(0))
		train_post_mps_obj_cls_loss.update(target_net.post_mps_cross_entropy_object.data.cpu().numpy()[0], im_data.size(0))
		# train_obj_box_loss.update(target_net.loss_obj_box.data.cpu().numpy()[0], im_data.size(0))
		train_pre_mps_pred_cls_loss.update(target_net.pre_mps_cross_entropy_predicate.data.cpu().numpy()[0], im_data.size(0))
		train_post_mps_pred_cls_loss.update(target_net.post_mps_cross_entropy_predicate.data.cpu().numpy()[0], im_data.size(0))

		train_rpn_loss.update(target_net.rcnn.rpn.loss.data.cpu().numpy()[0], im_data.size(0))
		# overall_train_loss.update(target_net.loss.data.cpu().numpy()[0], im_data.size(0))
		# overall_train_rpn_loss.update(target_net.rpn.loss.data.cpu().numpy()[0], im_data.size(0))

		accuracy_obj_pre_mps.update(target_net.pre_mps_tp_obj, target_net.pre_mps_tf_obj, target_net.pre_mps_fg_cnt_obj, target_net.pre_mps_bg_cnt_obj)
		accuracy_pred_pre_mps.update(target_net.pre_mps_tp_pred, target_net.pre_mps_tf_pred, target_net.pre_mps_fg_cnt_pred, target_net.pre_mps_bg_cnt_pred)
		accuracy_obj_post_mps.update(target_net.post_mps_tp_obj, target_net.post_mps_tf_obj, target_net.post_mps_fg_cnt_obj, target_net.post_mps_bg_cnt_obj)
		accuracy_pred_post_mps.update(target_net.post_mps_tp_pred, target_net.post_mps_tf_pred, target_net.post_mps_fg_cnt_pred, target_net.post_mps_bg_cnt_pred)


		t2 = time.time()
		optimizer.zero_grad()
		loss.backward()
		if args.enable_clip_gradient:
			network.clip_gradient(target_net, 10.)
		optimizer.step()
		if TIME_IT:
			t3 = time.time()
			print('backward time %.3fs')%(t3-t2)

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		# Logging the training loss
		if  (i + 1) % args.log_interval == 0:
			print('Epoch: [{0}][{1}/{2}] [lr: {lr}] [Solver: {solver}]\n'
				  '\tBatch_Time: {batch_time.avg: .3f}s\t'
				  'TOTAL Loss: {loss.avg: .4f}\t'
				  'RPN Loss: {rpn_loss.avg: .4f}'.format(
				   epoch, i + 1, len(train_loader), batch_time=batch_time,lr=args.lr,
				   loss=train_loss, rpn_loss=train_rpn_loss, solver=args.solver))

			print('[pre mps][Loss]\tRCNN_loss: %.4f\t pre_mps_pred_cls_loss: %.4f' %
				 (train_rcnn_loss.avg, train_pre_mps_pred_cls_loss.avg))
			print('[Accuracy]\t pre_mps_tp: %.2f, \tpre_mps_tf: %.2f, \tfg/bg=(%d/%d)'%
				 (accuracy_obj_pre_mps.ture_pos*100., accuracy_obj_pre_mps.true_neg*100., accuracy_obj_pre_mps.foreground, accuracy_obj_pre_mps.background))

			print('[post mps][Loss]\tpost_mps_obj_cls_loss: %.4f\t post_mps_pred_cls_loss: %.4f'%
				 (train_post_mps_obj_cls_loss.avg,  train_post_mps_pred_cls_loss.avg))
			print('[Accuracy]\t post_mps_tp: %.2f, \tpost_mps_tf: %.2f, \tfg/bg=(%d/%d)\n'%
			     (accuracy_obj_post_mps.ture_pos*100., accuracy_obj_post_mps.true_neg*100., accuracy_obj_post_mps.foreground, accuracy_obj_post_mps.background))


def test(test_loader, net, top_Ns, object_classes):

	global args

	print '========== Testing ========'
	net.eval()
	# For efficiency inference
	# net.use_region_reg = True

	rel_cnt = 0.
	rel_cnt_correct = np.zeros(len(top_Ns))
	box_num, obj_correct_cnt, obj_total_cnt = 0, 0, 0
	batch_time = network.AverageMeter()
	end = time.time()

	all_cls_gt_num = [0]*(len(object_classes)-1)
	all_cls_scores = [np.array([])]*(len(object_classes)-1)
	all_cls_tp = [np.array([])]*(len(object_classes)-1)

	for i, (im_data, im_info, gt_objects, gt_relationships) in enumerate(test_loader):
		# Forward pass
		# pdb.set_trace()
		total_cnt_t, rel_cnt_correct_t, object_rois, classes_scores, classes_tf, classes_gt_num = net.evaluate(
			im_data, im_info, gt_objects.numpy()[0], gt_relationships.numpy()[0],
			top_Ns = top_Ns, use_gt_boxes=args.use_gt_boxes, nms=False, nms_thresh=0.4, thresh=0.5, use_rpn_scores=args.use_rpn_scores)
		box_num += object_rois.size(0)
		obj_correct_cnt_t, obj_total_cnt_t = check_recall(object_rois/im_info[0][2], gt_objects.numpy()[0], 64)
		obj_correct_cnt += obj_correct_cnt_t
		obj_total_cnt += obj_total_cnt_t
		rel_cnt += total_cnt_t
		rel_cnt_correct += rel_cnt_correct_t

		for j in range(len(object_classes)-1):
			all_cls_scores[j] = np.append(all_cls_scores[j], classes_scores[j])
			all_cls_tp[j] = np.append(all_cls_tp[j], classes_tf[j])
			all_cls_gt_num[j] += classes_gt_num[j]

		batch_time.update(time.time() - end)
		end = time.time()
		if (i + 1) % args.log_interval == 0 and i > 0:
			print('Time: {0:2.3f}s/img.'
			      '\t[object] Avg: {1:2.2f} Boxes/im, Top-64 recall: {2:2.3f} ({3:d}/{4:d})'.format(
				batch_time.avg, box_num/float(i+1), obj_correct_cnt/float(obj_total_cnt)*100,
				obj_correct_cnt, obj_total_cnt))
			for idx, top_N in enumerate(top_Ns):
				print '[%d/%d][Evaluation] Top-%d Recall: %2.3f%%' % (
					i+1, len(test_loader), top_N, rel_cnt_correct[idx] / float(rel_cnt) * 100)

	all_aps = []
	for k, cls in enumerate(object_classes[1:]):
		# sort scores of all images
		cls_ap = cls_eval(all_cls_scores[k], all_cls_tp[k], all_cls_gt_num[k])
		all_aps += [cls_ap]
		print('AP for {} = {:.4f}'.format(cls, cls_ap))
	print('Mean AP = {:.4f}'.format(np.mean(all_aps)))

	recall = rel_cnt_correct / rel_cnt
	print '====== Done Testing ===='

	return recall


if __name__ == '__main__':
	main()

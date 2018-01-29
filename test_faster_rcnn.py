import os
import torch
import numpy as np
import time
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN  # Hierarchical_Descriptive_Model

from faster_rcnn.datasets.visual_genome_loader import visual_genome
from faster_rcnn.utils.map_eval import cls_eval, image_eval
from faster_rcnn.utils.HDN_utils import check_recall
import argparse


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser('Options for training RPN in pytorch')
parser.add_argument('--use_kmeans_anchors', default=True, help='Whether to use kmeans anchors')
## Environment Settings

parser.add_argument('--dataset_option', type=str, default='small', help='The dataset to use (small | normal | fat)')
parser.add_argument('--output_dir', type=str, default='./output/detection', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='Faster_RCNN', help='model name for snapshot')
parser.add_argument('--base_model', type=str, default='vgg', help='base model: vgg or resnet50 or resnet101')
parser.add_argument('--resume_training', default=True, help='Resume training from the model [resume_model]')
parser.add_argument('--resume_model', type=str, default='./output/detection/Faster_RCNN_small_vgg_12epoch_epoch_11.h5', help='The model we resume')
args = parser.parse_args()


def main():
    global args
    print "Loading training set and testing set..."
    # train_set = visual_genome(args.dataset_option, 'train')
    test_set = visual_genome(args.dataset_option, 'test')
    object_classes = test_set.object_classes
    print "Done."

    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    net = FasterRCNN(use_kmeans_anchors=args.use_kmeans_anchors, n_classes=len(object_classes), model=args.base_model)
    network.load_net(args.resume_model, net)
    # network.set_trainable(net.features, requires_grad=False)
    net.cuda()
    # Testing
    evaluate(test_loader, net, test_set.object_classes)


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
            # print('[{0}/{1})] Time: {2:2.3f}s/img').format(
            #     i+1, len(test_loader), batch_time.avg)

    all_aps = []
    for k, cls in enumerate(object_classes[1:]):
        # sort scores of all images
        cls_ap = cls_eval(all_cls_scores[k], all_cls_tp[k], all_cls_gt_num[k])
        all_aps += [cls_ap]
        print('AP for {} = {:.4f}'.format(cls, cls_ap))
    print('Mean AP = {:.4f}'.format(np.mean(all_aps)))

    print('====== Done Testing ====')


if __name__ == '__main__':
    main()

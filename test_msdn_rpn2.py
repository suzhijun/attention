import os
import torch
import numpy as np
import numpy.random as npr
import time
from faster_rcnn import network
from faster_rcnn.RPN import RPN  # Hierarchical_Descriptive_Model
from faster_rcnn.utils.timer import Timer
from faster_rcnn.utils.HDN_utils import check_recall

from faster_rcnn.datasets.visual_genome_loader import visual_genome
from faster_rcnn.fast_rcnn.config import cfg
from faster_rcnn.utils.cython_bbox import bbox_overlaps
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser('Options for training RPN in pytorch')

## training settings
parser.add_argument('--lr', type=float, default=0.01, help='To disable the Lanuage Model ')
parser.add_argument('--max_epoch', type=int, default=5, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--log_interval', type=int, default=500, help='Interval for Logging')
parser.add_argument('--disable_clip_gradient', action='store_true', help='Whether to clip the gradient')
parser.add_argument('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors')
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


def check_msdn_rpn(object_rois, gt_objects, gt_relationships, num_images=1):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """

    # In training stage use gt_obj to choose proper predict obj_roi, here obj_roi got by rpn and concat with gt_box
    overlaps = bbox_overlaps(
        np.ascontiguousarray(object_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_objects[:, :4], dtype=np.float))

    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    # print('gt', gt_objects.shape[0])
    # print('recall', np.sum(overlaps.max(axis=0)>=cfg.TRAIN.FG_THRESH))
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= 0.5)[0]
    gt_union_boxes_num = np.where(gt_relationships > 0)[0].size
    # print('fg_inds', fg_inds.size)

    #### prepare relationships targets

    # gt_assignment is the correct gt_box that predict box belong to.
    if fg_inds.size > 1:
        # fg_inds is the index of object_rois and labels to get fg_predict_rois
        # Grouping the input object rois
        id_i, id_j = np.meshgrid(range(fg_inds.size), range(fg_inds.size), indexing='ij')
        id_i = id_i.reshape(-1)
        id_j = id_j.reshape(-1)
        # use gt_relationships to choose pair sbj and obj box and relation type.
        pair_labels = gt_relationships[gt_assignment[fg_inds[id_i]], gt_assignment[fg_inds[id_j]]]

        if np.all(pair_labels==0):
            return fg_inds.size, 0, gt_union_boxes_num
        sub_list = gt_assignment[fg_inds[id_i]][np.where(pair_labels>0)]
        obj_list = gt_assignment[fg_inds[id_j]][np.where(pair_labels>0)]
        predicate_list = pair_labels[pair_labels > 0]
        phrase_list = np.vstack((sub_list, predicate_list, obj_list)).T

        # if np.unique(phrase_list, axis=0).shape[0] != np.where(gt_relationships > 0)[0].size:
        #     print(gt_objects.shape[0], np.unique(gt_assignment))
        #     print(np.array_equal(np.unique(gt_assignment), np.arange(gt_objects.shape[0])))
        #     print(np.where(gt_relationships > 0)[0].size)
        #     print('gt_relationships')
        #     print(gt_relationships[np.where(gt_relationships > 0)])
        #     print('sub', np.where(gt_relationships > 0)[0])
        #     print('obj', np.where(gt_relationships > 0)[1])
        #     print('gt_assignment', gt_assignment)
        #     print(np.unique(predicate_list))
        #     print(sub_list)
        #     print(predicate_list)
        #     print(obj_list)
        #
        #     print(phrase_list)
        #     # # print(phrase_list)
        #     # # print(np.unique(phrase_list, axis=0))
        #     print('unique')
        #     print(np.unique(phrase_list, axis=0))
        return fg_inds.size, np.unique(phrase_list, axis=0).shape[0], gt_union_boxes_num
    else:
        return fg_inds.size, 0, gt_union_boxes_num


def main():
    global args
    print "Loading training set and testing set..."
    # train_set = visual_genome(args.dataset_option, 'train')
    test_set = visual_genome('small', 'test')
    print test_set.num_object_classes
    print test_set.num_predicate_classes
    print "Done."

    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    net = RPN(not args.use_normal_anchors)
    network.load_net('./output/RPN/RPN_region_full_best.h5', net)
    # network.set_trainable(net.features, requires_grad=False)
    net.cuda()

    best_recall = np.array([0.0, 0.0])

    # Testing
    recall = test(test_loader, net)

    print('Recall: '
          'object: {recall[0]: .3f}%'
          'relationship: {recall[1]: .3f}%'.format(recall=recall * 100))


def test(test_loader, target_net):
    box_num = np.array([0, 0])
    correct_cnt, total_cnt = np.array([0, 0]), np.array([0, 0])
    print '========== Testing ======='
    target_net.eval()

    batch_time = network.AverageMeter()
    cover_cnt = 0
    cnt_gt = 0
    object_cnt = 0

    num = 0
    end = time.time()
    for i, (im_data, im_info, gt_objects, gt_relationships, gt_boxes_relationship) in enumerate(test_loader):
        # num += 1
        # if num > 320:
        #     break
        correct_cnt_t, total_cnt_t = np.array([0, 0]), np.array([0, 0])
        # Forward pass

        object_rois, relationship_rois = target_net(im_data, im_info.numpy(), gt_objects.numpy(),
                                                    gt_boxes_relationship.numpy())[1:]
        # all_rois = object_rois.data.cpu().numpy()
        # zeros = np.zeros((gt_objects.numpy().shape[1], 1), dtype=gt_objects.numpy().dtype)
        # # add gt_obj to predict_rois
        # all_rois = np.vstack(
        #     (all_rois, np.hstack((zeros, gt_objects.numpy()[0][:, :4])))
        # )
        # object_rois = network.np_to_variable(all_rois, is_cuda=True)
        # TODO: add rules
        # img_shape = im_info[0][:2]
        # object_rois = object_rois[:, 1:5]
        # relationship_rois = enlarge_rois_clip(relationship_rois[:, 1:5], 1.2, img_shape)
        # obj_in_predicate(object_rois, relationship_rois, 9)
        object_cnt_t, cover_cnt_t, cnt_gt_t = check_msdn_rpn(
            object_rois.data.cpu().numpy(), gt_objects.numpy()[0], gt_relationships.numpy()[0])
        object_cnt += object_cnt_t
        cover_cnt += cover_cnt_t
        cnt_gt += cnt_gt_t
        # gt_num = gt_boxes_relationship[0].size()[0]
        box_num[0] += object_rois.size(0)
        box_num[1] += relationship_rois.size(0)
        correct_cnt_t[0], total_cnt_t[0] = check_recall(object_rois, gt_objects[0].numpy(), object_rois.size(0),
                                                        thres=0.5)

        correct_cnt_t[1], total_cnt_t[1] = check_recall(relationship_rois, gt_boxes_relationship[0].numpy(), 64,
                                                        thres=0.5)
        correct_cnt += correct_cnt_t
        total_cnt += total_cnt_t
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 100 == 0 and i > 0:
            print('[{0}/{10}]  Time: {1:2.3f}s/img).'
                  '\t[object] Avg: {2:2.2f} Boxes/im, Top-2000 recall: {3:2.3f} ({4:d}/{5:d})'
                  '\t[relationship] Avg: {6:2.2f} Boxes/im, Top-500 recall: {7:2.3f} ({8:d}/{9:d})'.format(
                i + 1, batch_time.avg,
                box_num[0] / float(i + 1), correct_cnt[0] / float(total_cnt[0]) * 100, correct_cnt[0], total_cnt[0],
                box_num[1] / float(i + 1), correct_cnt[1] / float(total_cnt[1]) * 100, correct_cnt[1], total_cnt[1],
                len(test_loader)))
            print('fg object number:{0:d}'
                  '\tcover number:{1:d}'
                  '\tcover & sub & obj vs gt_relationship_boxes average recall: {2:.3f}').format(
                object_cnt / i, cover_cnt / i,
                cover_cnt / float(cnt_gt) * 100)
            # print('\n')

    recall = correct_cnt / total_cnt.astype(np.float)
    print('====== Done Testing ====')
    return recall


if __name__ == '__main__':
    main()

#!/usr/bin/env python
import argparse
import mxnet as mx
import os
import sys
from train.train_net_mobi import train_net


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Single-shot detection network')
    parser.add_argument('--train-path', dest='train_path', help='train record to use',
                        default=os.path.join(os.getcwd(), 'data', 'train.rec'), type=str)
    parser.add_argument('--train-list', dest='train_list', help='train list to use',
                        default="", type=str)
    parser.add_argument('--val-path', dest='val_path', help='validation record to use',
                        default=os.path.join(os.getcwd(), 'data', 'val.rec'), type=str)
    parser.add_argument('--val-list', dest='val_list', help='validation list to use',
                        default="", type=str)
    parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
                        help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--resume', dest='resume', type=int, default=-1,
                        help='resume training from epoch n')
    parser.add_argument('--finetune', dest='finetune', type=int, default=-1,
                        help='finetune from epoch n, rename the model before doing this')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'vgg16_reduced'), type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=1, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'output', 'voc0712', 'ssd'), type=str)
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0', type=str)
    parser.add_argument('--begin-epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end-epoch', dest='end_epoch', help='end epoch of training',
                        default=240, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300,
                        help='set image shape')
    parser.add_argument('--label-width', dest='label_width', type=int, default=350,
                        help='force padding label width to sync across train and validation')
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='sgd',
                        help='Whether to use a different optimizer or follow the original code with sgd')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.004,
                        help='learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0.0005,
                        help='weight decay')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--lr-steps', dest='lr_refactor_step', type=str, default='80, 160',
                        help='refactor learning rate at specified epochs')
    parser.add_argument('--lr-factor', dest='lr_refactor_ratio', type=str, default=0.1,
                        help='ratio to refactor learning rate')
    parser.add_argument('--freeze', dest='freeze_pattern', type=str, default="^(conv1_|conv2_).*",
                        help='freeze layer pattern')
    parser.add_argument('--log', dest='log_file', type=str, default="train.log",
                        help='save training log to file')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    parser.add_argument('--pattern', dest='monitor_pattern', type=str, default=".*",
                        help='monitor parameter pattern, as regex')
    parser.add_argument('--num-class', dest='num_class', type=int, default=20,
                        help='number of classes')
    parser.add_argument('--num-example', dest='num_example', type=int, default=16551,
                        help='number of image examples')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='aeroplane, bicycle, bird, boat, bottle, bus, \
                        car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                        person, pottedplant, sheep, sofa, train, tvmonitor',
                        help='string of comma separated names, or text filename')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.45,
                        help='non-maximum suppression threshold')
    parser.add_argument('--nms_topk', dest='nms_topk', type=int, default=400,
                        help='final number of detections')
    parser.add_argument('--overlap', dest='overlap_thresh', type=float, default=0.5,
                        help='evaluation overlap threshold')
    parser.add_argument('--force', dest='force_nms', type=bool, default=False,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--use-difficult', dest='use_difficult', type=bool, default=False,
                        help='use difficult ground-truths in evaluation')
    parser.add_argument('--voc07', dest='use_voc07_metric', type=bool, default=True,
                        help='use PASCAL VOC 07 11-point metric')
    parser.add_argument('--tensorboard', dest='tensorboard', type=bool, default=False,
                        help='save metrics into tensorboard readable files')
    parser.add_argument('--min_neg_samples', dest='min_neg_samples', type=int, default=0,
                        help='min number of negative samples taken in hard mining.')

    args = parser.parse_args()
    return args

def parse_class_names(args):
    """ parse # classes and class_names if applicable """
    num_class = args.num_class
    if len(args.class_names) > 0:
        if os.path.isfile(args.class_names):
            # try to open it to read class names
            with open(args.class_names, 'r') as f:
                class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in args.class_names.split(',')]
        assert len(class_names) == num_class, str(len(class_names))
        for name in class_names:
            assert len(name) > 0
    else:
        class_names = None
    return class_names

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx
    # class names if applicable
    class_names = parse_class_names(args)
    # start training


    print('args.network = {}'.format(args.network))
    print('args.train_path = {}'.format(args.train_path))
    print('args.num_class = {}'.format(args.num_class))
    print('args.batch_size = {}'.format(args.batch_size))
    print('args.data_shape = {}'.format(args.data_shape))
    print('args.mean_r = {}'.format(args.mean_r))
    print('args.mean_g = {}'.format(args.mean_g))
    print('args.mean_b = {}'.format(args.mean_b))
    print('args.resume = {}'.format(args.resume))
    print('args.finetune = {}'.format(args.finetune))
    print('args.pretrained = {}'.format(args.pretrained))
    print('args.epoch = {}'.format(args.epoch))
    print('args.prefix = {}'.format(args.prefix))
    print('ctx = {}'.format(ctx))
    print('args.begin_epoch = {}'.format(args.begin_epoch))
    print('args.end_epoch = {}'.format(args.end_epoch))
    print('args.frequent = {}'.format(args.frequent))
    print('args.learning_rate = {}'.format(args.learning_rate))
    print('args.momentum = {}'.format(args.momentum))
    print('args.weight_decay = {}'.format(args.weight_decay))
    print('args.lr_refactor_step = {}'.format(args.lr_refactor_step))
    print('args.lr_refactor_ratio = {}'.format(args.lr_refactor_ratio))
    print('ctx = {}'.format(ctx))
    print('args.val_path = {}'.format(args.val_path))
    print('args.min_neg_samples = {}'.format(args.min_neg_samples))
    print('args.num_example = {}'.format(args.num_example))
    print('class_names = {}'.format(class_names))
    print('args.label_width = {}'.format(args.label_width))
    print('args.freeze_pattern = {}'.format(args.freeze_pattern))
    print('args.monitor = {}'.format(args.monitor))
    print('args.monitor_pattern = {}'.format(args.monitor_pattern))
    print('args.log_file = {}'.format(args.log_file))
    print('args.nms_thresh = {}'.format(args.nms_thresh))
    print('args.nms_topk = {}'.format(args.nms_topk))
    print('args.force_nms = {}'.format(args.force_nms))
    print('args.overlap_thresh = {}'.format(args.overlap_thresh))
    print('args.use_difficult = {}'.format(args.use_difficult))
    print('args.use_voc07_metric = {}'.format(args.use_voc07_metric))
    print('args.optimizer = {}'.format(args.optimizer))
    print('args.tensorboard = {}'.format(args.tensorboard))



    train_net(args.network, args.train_path,
              args.num_class, args.batch_size,
              args.data_shape, [args.mean_r, args.mean_g, args.mean_b],
              args.resume, args.finetune, args.pretrained,
              args.epoch, args.prefix, ctx, args.begin_epoch, args.end_epoch,
              args.frequent, args.learning_rate, args.momentum, args.weight_decay,
              args.lr_refactor_step, args.lr_refactor_ratio,
              val_path=args.val_path,
              min_neg_samples=args.min_neg_samples,
              num_example=args.num_example,
              class_names=class_names,
              label_pad_width=args.label_width,
              freeze_layer_pattern=args.freeze_pattern,
              iter_monitor=args.monitor,
              monitor_pattern=args.monitor_pattern,
              log_file=args.log_file,
              nms_thresh=args.nms_thresh,
              nms_topk=args.nms_topk,
              force_nms=args.force_nms,
              ovp_thresh=args.overlap_thresh,
              use_difficult=args.use_difficult,
              voc07_metric=args.use_voc07_metric,
              optimizer=args.optimizer,
              tensorboard=args.tensorboard)

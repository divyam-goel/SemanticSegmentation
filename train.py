import argparse
import os
import os.path as osp
# import pickle
# import random
# import sys

# import cv2
# import matplotlib.pyplot as plt
import numpy as np
# import pickle
# import scipy.misc
# from packaging import version

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data, model_zoo

from model.deeplab import DeepLab
from model.discriminator import Discriminator
from dataset.voc_dataset import VOCDataSet, VOCGTDataSet
from utils.loss import CrossEntropy2d, BCEWithLogitsLoss2d


# net = BasicNeuralNet()
# BatchNorm = nn.BatchNorm2d
# net = build_aspp(16, BatchNorm)
# net = build_backbone(16, BatchNorm)
# net = build_decoder(21, BatchNorm)
# net = DeepLab()
# net = Discriminator(21)
# print(net)

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

INPUT_SIZE = '321,321'
# MODEL = 'DeepLab'
BATCH_SIZE = 10
ITER_SIZE = 1
NUM_WORKERS = 4
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 20000
POWER = 0.9
RANDOM_SEED = 1234
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_ADV_PRED = 0.1

DATA_DIRECTORY = './dataset/VOC2012'
DATA_LIST_PATH = './dataset/voc_list/train_aug.txt'
SNAPSHOT_DIR = './snapshots/'

RESTORE_FROM = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'

# PARTIAL_DATA=0.5
IGNORE_LABEL = 255

SEMI_START=5000
LAMBDA_SEMI=0.1
MASK_T=0.2

LAMBDA_SEMI_ADV=0.001
SEMI_START_ADV=0
D_REMAIN=True


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    # parser.add_argument("--model", type=str, default=MODEL,
    #                     help="available options : DeepLab/DRN")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    # parser.add_argument("--partial-data", type=float, default=PARTIAL_DATA,
    #                     help="The index of the label to ignore during the training.")
    # parser.add_argument("--partial-id", type=str, default=None,
    #                     help="restore partial id list")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-pred", type=float, default=LAMBDA_ADV_PRED,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-semi", type=float, default=LAMBDA_SEMI,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--lambda-semi-adv", type=float, default=LAMBDA_SEMI_ADV,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--mask-T", type=float, default=MASK_T,
                        help="mask T for semi adversarial training.")
    parser.add_argument("--semi-start", type=int, default=SEMI_START,
                        help="start semi learning after # iterations")
    parser.add_argument("--semi-start-adv", type=int, default=SEMI_START_ADV,
                        help="start semi learning after # iterations")
    parser.add_argument("--D-remain", type=bool, default=D_REMAIN,
                        help="Whether to train D with unlabeled data")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    # parser.add_argument("--gpu", type=int, default=0,
    #                     help="choose gpu device.")
    return parser.parse_args()

args = get_arguments()

def main():

    # parse input size
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    # cudnn.enabled = True
    # gpu = args.gpu

    # create segmentation network
    model = DeepLab(num_classes=args.num_classes)

    # load pretrained parameters
    # if args.restore_from[:4] == 'http' :
    #     saved_state_dict = model_zoo.load_url(args.restore_from)
    # else:
    #     saved_state_dict = torch.load(args.restore_from)

    # only copy the params that exist in current model (caffe-like)
    # new_params = model.state_dict().copy()
    # for name, param in new_params.items():
    #     if name in saved_state_dict and param.size() == saved_state_dict[name].size():
    #         new_params[name].copy_(saved_state_dict[name])
    # model.load_state_dict(new_params)

    model.train()
    model.cpu()
    # model.cuda(args.gpu)
    # cudnn.benchmark = True


    # create discriminator network
    model_D = Discriminator(num_classes=args.num_classes)
    # if args.restore_from_D is not None:
    #     model_D.load_state_dict(torch.load(args.restore_from_D))
    model_D.train()
    model_D.cpu()
    # model_D.cuda(args.gpu)


    # Create directory to save snapshots of the model
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    # Load train data and ground truth labels
    # train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size,
    #                 scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
    # train_gt_dataset = VOCGTDataSet(args.data_dir, args.data_list, crop_size=input_size,
    #                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    # trainloader = data.DataLoader(train_dataset,
    #                 batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=False)
    # trainloader_gt = data.DataLoader(train_gt_dataset,
    #                 batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=False)

    # trainloader_iter = enumerate(trainloader)
    # trainloader_gt_iter = enumerate(trainloader_gt)


    # optimizer for segmentation network
    optimizer = optim.SGD(model.optim_parameters(args),
                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D.zero_grad()

    print(optimizer)
    print(optimizer_D)


    # loss/ bilinear upsampling
    bce_loss = BCEWithLogitsLoss2d()
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)


    # labels for adversarial training
    pred_label = 0
    gt_label = 1





if __name__ == '__main__':
    main()

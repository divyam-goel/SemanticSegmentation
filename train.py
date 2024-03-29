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
from dataset.temp_dataset import MyCustomDataset
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

def loss_calc(pred, label): #, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cpu()
    # label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cpu()
    # criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)

def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape)*label
    D_label[ignore_mask] = 255
    D_label = Variable(torch.FloatTensor(D_label)).cpu()
    # D_label = Variable(torch.FloatTensor(D_label)).cuda(args.gpu)

    return D_label

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

    
    # MILESTONE 1
    print("Printing MODELS ...")
    print(model)
    print(model_D)


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

    train_dataset = MyCustomDataset()
    train_gt_dataset = MyCustomDataset()

    trainloader = data.DataLoader(train_dataset, batch_size=5, shuffle=True)
    trainloader_gt = data.DataLoader(train_gt_dataset, batch_size=5, shuffle=True)

    trainloader_iter = enumerate(trainloader)
    trainloader_gt_iter = enumerate(trainloader_gt)

    
    # MILESTONE 2
    print("Printing Loaders")
    print(trainloader_iter)
    print(trainloader_gt_iter)


    # optimizer for segmentation network
    optimizer = optim.SGD(model.optim_parameters(args),
                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D.zero_grad()


    # MILESTONE 3
    print("Printing OPTIMIZERS ...")
    print(optimizer)
    print(optimizer_D)


    # loss/ bilinear upsampling
    bce_loss = BCEWithLogitsLoss2d()
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)


    # labels for adversarial training
    pred_label = 0
    gt_label = 1


    for i_iter in range(args.num_steps):

        loss_seg_value = 0
        loss_adv_pred_value = 0
        loss_D_value = 0
        loss_semi_value = 0
        loss_semi_adv_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # do semi first
            # if (args.lambda_semi > 0 or args.lambda_semi_adv > 0 ) and i_iter >= args.semi_start_adv :
            #     try:
            #         _, batch = next(trainloader_remain_iter)
            #     except:
            #         trainloader_remain_iter = enumerate(trainloader_remain)
            #         _, batch = next(trainloader_remain_iter)

            #     # only access to img
            #     images, _, _, _ = batch
            #     images = Variable(images).cuda(args.gpu)


            #     pred = interp(model(images))
            #     pred_remain = pred.detach()

            #     D_out = interp(model_D(F.softmax(pred)))
            #     D_out_sigmoid = F.sigmoid(D_out).data.cpu().numpy().squeeze(axis=1)

            #     ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool)

            #     loss_semi_adv = args.lambda_semi_adv * bce_loss(D_out, make_D_label(gt_label, ignore_mask_remain))
            #     loss_semi_adv = loss_semi_adv/args.iter_size

            #     #loss_semi_adv.backward()
            #     loss_semi_adv_value += loss_semi_adv.data.cpu().numpy()/args.lambda_semi_adv

            #     if args.lambda_semi <= 0 or i_iter < args.semi_start:
            #         loss_semi_adv.backward()
            #         loss_semi_value = 0
            #     else:
            #         # produce ignore mask
            #         semi_ignore_mask = (D_out_sigmoid < args.mask_T)

            #         semi_gt = pred.data.cpu().numpy().argmax(axis=1)
            #         semi_gt[semi_ignore_mask] = 255

            #         semi_ratio = 1.0 - float(semi_ignore_mask.sum())/semi_ignore_mask.size
            #         print('semi ratio: {:.4f}'.format(semi_ratio))

            #         if semi_ratio == 0.0:
            #             loss_semi_value += 0
            #         else:
            #             semi_gt = torch.FloatTensor(semi_gt)

            #             loss_semi = args.lambda_semi * loss_calc(pred, semi_gt, args.gpu)
            #             loss_semi = loss_semi/args.iter_size
            #             loss_semi_value += loss_semi.data.cpu().numpy()/args.lambda_semi
            #             loss_semi += loss_semi_adv
            #             loss_semi.backward()

            # else:
            #     loss_semi = None
            #     loss_semi_adv = None

            # train with source

            try:
                _, batch = next(trainloader_iter)
            except:
                trainloader_iter = enumerate(trainloader)
                _, batch = next(trainloader_iter)

            images, labels, _, _ = batch
            images = Variable(images).cpu()
            # images = Variable(images).cuda(args.gpu)
            ignore_mask = (labels.numpy() == 255)
            
            # segmentation prediction
            pred = interp(model(images))
            # (spatial multi-class) cross entropy loss
            loss_seg = loss_calc(pred, labels)
            # loss_seg = loss_calc(pred, labels, args.gpu)

            # discriminator prediction
            D_out = interp(model_D(F.softmax(pred)))
            # adversarial loss
            loss_adv_pred = bce_loss(D_out, make_D_label(gt_label, ignore_mask))

            # multi-task loss
            # lambda_adv - weight for minimizing loss
            loss = loss_seg + args.lambda_adv_pred * loss_adv_pred

            # loss normalization
            loss = loss/args.iter_size
            
            # back propagation
            loss.backward()
            
            loss_seg_value += loss_seg.data.cpu().numpy()/args.iter_size
            loss_adv_pred_value += loss_adv_pred.data.cpu().numpy()/args.iter_size


            # train D

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with pred
            pred = pred.detach()

            # if args.D_remain:
            #     pred = torch.cat((pred, pred_remain), 0)
            #     ignore_mask = np.concatenate((ignore_mask,ignore_mask_remain), axis = 0)

            D_out = interp(model_D(F.softmax(pred)))
            loss_D = bce_loss(D_out, make_D_label(pred_label, ignore_mask))
            loss_D = loss_D/args.iter_size/2
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()


            # train with gt
            # get gt labels
            try:
                _, batch = next(trainloader_gt_iter)
            except:
                trainloader_gt_iter = enumerate(trainloader_gt)
                _, batch = next(trainloader_gt_iter)

            _, labels_gt, _, _ = batch
            D_gt_v = Variable(one_hot(labels_gt)).cpu()
            # D_gt_v = Variable(one_hot(labels_gt)).cuda(args.gpu)
            ignore_mask_gt = (labels_gt.numpy() == 255)

            D_out = interp(model_D(D_gt_v))
            loss_D = bce_loss(D_out, make_D_label(gt_label, ignore_mask_gt))
            loss_D = loss_D/args.iter_size/2
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()



        optimizer.step()
        optimizer_D.step()

        print('exp = {}'.format(args.snapshot_dir))
        print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.3f}, loss_semi = {5:.3f}, loss_semi_adv = {6:.3f}'.format(i_iter, args.num_steps, loss_seg_value, loss_adv_pred_value, loss_D_value, loss_semi_value, loss_semi_adv_value))

        if i_iter >= args.num_steps-1:
            print ('save model ...')
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(args.num_steps)+'.pth'))
            torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(args.num_steps)+'_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print ('taking snapshot ...')
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(i_iter)+'.pth'))
            torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(i_iter)+'_D.pth'))

    end = timeit.default_timer()
    print (end-start,'seconds')



if __name__ == '__main__':
    main()

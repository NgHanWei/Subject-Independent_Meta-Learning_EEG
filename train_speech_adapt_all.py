#!/usr/bin/env python
# coding: utf-8
'''Subject-adaptative classification with KU Data,
using Deep ConvNet model from [1].

References
----------
.. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
   Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
   Deep learning with convolutional neural networks for EEG decoding and
   visualization.
   Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
'''
import argparse
import json
import logging
import sys
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from speech_braindecode.models.deep4 import Deep5Net
from speech_braindecode.torch_ext.optimizers import AdamW
from speech_braindecode.torch_ext.util import set_random_seeds
from torch import nn

from sklearn.utils import shuffle

from Inner_Speech_Dataset.Python_Processing.Data_extractions import  Extract_data_from_subject
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Select_time_window, Transform_for_classificator, Split_trial_in_time

# python train_adapt.py -scheme 5 -trfrate 10 -subj $subj

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
set_random_seeds(seed=2022, cuda=True)

parser = argparse.ArgumentParser(
    description='Subject-adaptative classification with Inner Speech')
parser.add_argument('rootdir', type=str, help='Path to the root directory containing data')
parser.add_argument('modelpath', type=str,
                    help='Path to the base model folder')
parser.add_argument('outpath', type=str, help='Path to the result folder')
parser.add_argument('--meta',default=False, help='Training Mode', action='store_true')
parser.add_argument('-scheme', type=int, help='Adaptation scheme', default=1)
parser.add_argument(
    '-trfrate', type=int, help='The percentage of data for adaptation', default=40)
parser.add_argument('-lr', type=float, help='Learning rate', default=10)
parser.add_argument('-gpu', type=int, help='The gpu device to use', default=0)

args = parser.parse_args()
outpath = args.outpath
modelpath = args.modelpath
scheme = args.scheme
rate = args.trfrate
lr = args.lr
meta = args.meta
torch.cuda.set_device(args.gpu)
set_random_seeds(seed=2022, cuda=True)
BATCH_SIZE = 16
TRAIN_EPOCH = 100

root_dir = args.rootdir

# Data Type
datatype = "EEG"

# Sampling rate
fs = 256

# Select the useful par of each trial. Time in seconds
t_start = 1.5
t_end = 3.5

all_loss = 0
for subj in range(1,11):

    # Target Subject
    targ_subj = subj #[1 to 10]

    X_train = np.array([])
    Y_train = np.array([])
    for i in range(1,11):

        # Subject number
        N_S = i   #[1 to 10]

        #@title Data extraction and processing

        # Load all trials for a sigle subject
        X, Y = Extract_data_from_subject(root_dir, N_S, datatype)

        # Cut usefull time. i.e action interval
        X = Select_time_window(X = X, t_start = t_start, t_end = t_end, fs = fs)

        # print("Data shape: [trials x channels x samples]")
        # print(X.shape) # Trials, channels, samples

        # print("Labels shape")
        # print(Y.shape) # Time stamp, class , condition, session

        # Conditions to compared
        Conditions = [["Inner"],["Inner"],["Inner"],["Inner"]]
        # The class for the above condition
        Classes    = [  ["Up"] ,["Down"],["Left"],["Right"] ]

        # Transform data and keep only the trials of interes
        X , Y =  Transform_for_classificator(X, Y, Classes, Conditions)

        print("Final data shape")
        print(X.shape)

        print("Final labels shape")
        print(Y.shape)

        # Normalize Data
        Max_val = 500
        norm = np.amax(abs(X))
        X = Max_val * X/norm

        if i == targ_subj:
            print(X.shape)
            # p = np.random.permutation(len(Y))
            # X = X[p]
            # Y = Y[p]
            X, Y = shuffle(X, Y, random_state=0)
            X_test = X
            Y_test = Y
        else:
            X_train = np.concatenate((X_train, X),axis=0) if len(X_train) > 0 else X
            Y_train = np.concatenate((Y_train, Y),axis=0) if len(Y_train) > 0 else Y

    print("Training Data shape")    
    print(X_train.shape)
    print(Y_train.shape)

    print("Test Data shape")
    print(X_test.shape)
    print(Y_test.shape)

    X_train2 = X_train.astype(np.float32)
    Y_train2 = Y_train.astype(np.int64)
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.int64)


    n_classes = 4
    in_chans = X.shape[1]
    # final_conv_length = auto ensures we only get a single output in the time dimension
    model = Deep5Net(in_chans=in_chans, n_classes=n_classes,
                    input_time_length=X.shape[2],
                    final_conv_length='auto').cuda()

    # Deprecated.


    def reset_conv_pool_block(network, block_nr):
        suffix = "_{:d}".format(block_nr)
        conv = getattr(network, 'conv' + suffix)
        kernel_size = conv.kernel_size
        n_filters_before = conv.in_channels
        n_filters = conv.out_channels
        setattr(network, 'conv' + suffix,
                nn.Conv2d(
                    n_filters_before,
                    n_filters,
                    kernel_size,
                    stride=(1, 1),
                    bias=False,
                ))
        setattr(network, 'bnorm' + suffix,
                nn.BatchNorm2d(
                    n_filters,
                    momentum=0.1,
                    affine=True,
                    eps=1e-5,
                ))
        # Initialize the layers.
        conv = getattr(network, 'conv' + suffix)
        bnorm = getattr(network, 'bnorm' + suffix)
        nn.init.xavier_uniform_(conv.weight, gain=1)
        nn.init.constant_(bnorm.weight, 1)
        nn.init.constant_(bnorm.bias, 0)


    def reset_model(checkpoint):
        # Load the state dict of the model.
        model.network.load_state_dict(checkpoint['model_state_dict'])

        # # Resets the last conv block
        # reset_conv_pool_block(model.network, block_nr=4)
        # reset_conv_pool_block(model.network, block_nr=3)
        # reset_conv_pool_block(model.network, block_nr=2)
        # # Resets the fully-connected layer.
        # # Parameters of newly constructed modules have requires_grad=True by default.
        # n_final_conv_length = model.network.conv_classifier.kernel_size[0]
        # n_prev_filter = model.network.conv_classifier.in_channels
        # n_classes = model.network.conv_classifier.out_channels
        # model.network.conv_classifier = nn.Conv2d(
        #     n_prev_filter, n_classes, (n_final_conv_length, 1), bias=True)
        # nn.init.xavier_uniform_(model.network.conv_classifier.weight, gain=1)
        # nn.init.constant_(model.network.conv_classifier.bias, 0)

        if scheme != 5:
            # Freeze all layers.
            for param in model.network.parameters():
                param.requires_grad = False

            if scheme in {1, 2, 3, 4}:
                # Unfreeze the FC layer.
                for param in model.network.conv_classifier.parameters():
                    param.requires_grad = True

            if scheme in {2, 3, 4}:
                # Unfreeze the conv4 layer.
                for param in model.network.conv_4.parameters():
                    param.requires_grad = True
                for param in model.network.bnorm_4.parameters():
                    param.requires_grad = True

            if scheme in {3, 4}:
                # Unfreeze the conv3 layer.
                for param in model.network.conv_3.parameters():
                    param.requires_grad = True
                for param in model.network.bnorm_3.parameters():
                    param.requires_grad = True

            if scheme == 4:
                # Unfreeze the conv2 layer.
                for param in model.network.conv_2.parameters():
                    param.requires_grad = True
                for param in model.network.bnorm_2.parameters():
                    param.requires_grad = True

        # Only optimize parameters that requires gradient.
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.network.parameters()),
                        lr=lr, weight_decay=0.5*0.001)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.compile(loss=F.nll_loss, optimizer=optimizer,
                    iterator_seed=2022, )

    cutoff = int(rate * 50 / 100)
    # Use only session 1 data for training
    assert(cutoff <= 50)

    total_loss = []

    checkpoint = torch.load(pjoin(modelpath, 'model_subj' + str(targ_subj) + '.pt'),
                            map_location='cuda:' + str(args.gpu))
    reset_model(checkpoint)

    X, Y = X_test, Y_test

    X_train, Y_train = X[:cutoff], Y[:cutoff]

    X_val, Y_val = X[50:], Y[50:]
    # X_val, Y_val = X, Y
    model.fit(X_train, Y_train, epochs=TRAIN_EPOCH,
                batch_size=BATCH_SIZE, scheduler='cosine',
                validation_data=(X_val, Y_val), remember_best_column='valid_misclass',meta=meta)
    model.epochs_df.to_csv(pjoin(outpath, 'epochs' + str(targ_subj) + '.csv'))
    test_loss = model.evaluate(X_test[50:], Y_test[50:])
    total_loss.append(test_loss["misclass"])
    with open(pjoin(outpath, 'test' + str(targ_subj) + '.json'), 'w') as f:
        json.dump(test_loss, f)

    print(total_loss[0])
    all_loss += total_loss[0]

print(all_loss/10)
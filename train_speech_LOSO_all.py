import os
import mne 
import pickle
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

from speech_braindecode.models.deep4 import Deep5Net
from speech_braindecode.datautil.signal_target import SignalAndTarget
from speech_braindecode.torch_ext.optimizers import AdamW
from speech_braindecode.torch_ext.util import set_random_seeds
import torch.nn.functional as F
import torch
from os.path import join as pjoin
import argparse
import json
import logging
import sys

from sklearn.utils import shuffle

from Inner_Speech_Dataset.Python_Processing.Data_extractions import  Extract_data_from_subject
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Select_time_window, Transform_for_classificator, Split_trial_in_time

mne.set_log_level(verbose='warning') #to avoid info at terminal
warnings.filterwarnings(action = "ignore", category = DeprecationWarning ) 
warnings.filterwarnings(action = "ignore", category = FutureWarning ) 

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
set_random_seeds(seed=2022, cuda=True)

parser = argparse.ArgumentParser(
    description='Subject-independent classification with Inner Speech')
parser.add_argument('rootdir', type=str, help='Path to the root directory containing data')
parser.add_argument('--meta',default=False, help='Training Mode', action='store_true')

args = parser.parse_args()

### Hyperparameters

# The root dir have to point to the folder that cointains the database
root_dir = args.rootdir

# Data Type
datatype = "EEG"
# Sampling rate
fs = 256
# Select the useful par of each trial. Time in seconds
t_start = 1.5
t_end = 3.5

meta = args.meta

for targ_subj in range(1,11):

    X_train = np.array([])
    Y_train = np.array([])
    X_val = np.array([])
    Y_val = np.array([])
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

        subjs = [1,2,3,4,5,6,7,8,9,10]

        if i == targ_subj:
            X, Y = shuffle(X, Y, random_state=0)
            X_test = X[50:]
            Y_test = Y[50:]
        elif i == subjs[targ_subj-2]:
            X_val = X[:].astype(np.float32)
            Y_val = Y[:].astype(np.int64)
        else:
            X, Y = shuffle(X, Y, random_state=0)
            X_train = np.concatenate((X_train, X),axis=0) if X_train != [] else X
            Y_train = np.concatenate((Y_train, Y),axis=0) if Y_train != [] else Y

    print("Training Data shape")    
    print(X_train.shape)
    print(Y_train.shape)

    print("Test Data shape")
    print(X_test.shape)
    print(Y_test.shape)

    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.int64)
    X_val = X_val.astype(np.float32)
    Y_val = Y_val.astype(np.int64)
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.int64)

    ## Augmentation - Random Masking
    # X_aug = X_train
    # Y_aug = Y_train
    # for i in range(0,X_train.shape[0]):
    #     for j in range(0,X_train.shape[1]):
    #         rand_int = random.randint(0,450)
    #         X_aug[i,j,rand_int:rand_int+50] = 0
    # X_train = np.concatenate((X_train, X_aug),axis=0)
    # Y_train = np.concatenate((Y_train, Y_aug),axis=0)

    ### Training Details
    TRAIN_EPOCH = 200
    BATCH_SIZE = 16

    train_set = SignalAndTarget(X_train, y=Y_train)
    valid_set = SignalAndTarget(X_val, y=Y_val)
    test_set = SignalAndTarget(X_test, y=Y_test)
    n_classes = 4
    in_chans = train_set.X.shape[1]

    model = Deep5Net(in_chans=in_chans, n_classes=n_classes,
                        input_time_length=train_set.X.shape[2],
                        final_conv_length='auto').cuda()

    optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001)
    model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, )

    exp = model.fit(train_set.X, train_set.y, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, scheduler='cosine',
                        validation_data=(valid_set.X, valid_set.y), remember_best_column='valid_loss', meta=meta)

    rememberer = exp.rememberer
    base_model_param = {
        'epoch': rememberer.best_epoch,
        'model_state_dict': rememberer.model_state_dict,
        'optimizer_state_dict': rememberer.optimizer_state_dict,
        'loss': rememberer.lowest_val
    }

    if meta == False:

        torch.save(base_model_param, pjoin(
            './speech_results_baseline/', 'model_subj{}.pt'.format(targ_subj)))
        model.epochs_df.to_csv(
            pjoin('./speech_results_baseline/', 'LOSO_epochs_subj{}.csv'.format(targ_subj)))

        test_loss = model.evaluate(test_set.X, test_set.y)
        with open(pjoin('./speech_results_baseline/', 'test_LOSO_subj{}.json'.format(targ_subj)), 'w') as f:
            json.dump(test_loss, f)

    else:

        torch.save(base_model_param, pjoin(
            './speech_results_meta/', 'model_subj{}.pt'.format(targ_subj)))
        model.epochs_df.to_csv(
            pjoin('./speech_results_meta/', 'Meta_epochs_subj{}.csv'.format(targ_subj)))

        test_loss = model.evaluate(test_set.X, test_set.y)
        with open(pjoin('./speech_results_meta/', 'test_meta_subj{}.json'.format(targ_subj)), 'w') as f:
            json.dump(test_loss, f)

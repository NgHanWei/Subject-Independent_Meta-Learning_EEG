
# Subject-Independent Meta-Learning for EEG-based Motor Imagery and Inner Speech Classification

## Results Overview

### Motor Imagery

### Inner Speech

## Resources

Motor Imagery Raw Dataset: [Link](http://gigadb.org/dataset/100542)

The pre-processing code of the dataset is provided by Kaishuo et al.: [Link](https://github.com/zhangks98/eeg-adapt)

Inner Speech Dataset: [Link](https://openneuro.org/datasets/ds003626/versions/2.1.2)

Original Inner Speech Github: [Link](https://github.com/N-Nieto/Inner_Speech_Dataset)

## Dependencies

## Run

### Motor Imagery

Conditions = Motor Imagery

Classes = "Left", "Right"

#### Obtain the raw dataset

Download the motor imagery raw dataset from the resources above, and save them to the same `$source` folder. To conserve space, you may only download files that ends with `EEG_MI.mat`.

#### Pre-process the raw dataset

The following command will read the raw dataset from the `$source` folder, and output the pre-processed data `KU_mi_smt.h5` into the `$target` folder.

```
python preprocess_h5_smt.py $source $target
```

#### Training the classifier

Run `train_motor_LOSO.py`
```
usage: python train_motor_LOSO.py [--META] [DATAPATH] [OUTPATH] [-gpu GPU] [-fold FOLD]

Training a subject-indepdendent meta-learning baseline model with cross validation for a single subject.

Positional Arguments:
    DATAPATH                            Path for the pre-processed EEG signals file
    OUTPATH                             Path to folder for saving the trained model and results in

Optional Arguments:
    --meta                              Set to enable meta-learning, default meta-learning is switched off
    -gpu GPU                            Set gpu to use, default is 0
    -fold FOLD                          Set the fold number to determine subject for training a binary-class motor imagery classification model
```

To obtain baseline for all subjects, run `train_motor_LOSO_all.py`:
```
usage: python train_motor_LOSO_all.py [--META] [DATAPATH] [OUTPATH] [-gpu GPU]

Trains subject-indepdendent meta-learning baseline model with cross validation for all subjects from the KU dataset.

Positional Arguments:
    DATAPATH                            Path for the pre-processed EEG signals file
    OUTPATH                             Path to folder for saving the trained model and results in

Optional Arguments:
    --meta                              Set to enable meta-learning, default meta-learning is switched off
    -gpu GPU                            Set gpu to use, default is 0
```

#### Evaluate baseline models overall performance, and prepare baseline models for adaptation

Obtaining baseline models for subsequent adaptation, run `eval_motor_base.py`:
```
usage: python eval_motor_base.py [DATAPATH] [MODELPATH] [OUTPATH] [-gpu GPU]

Evaluate performance of models for each subject and selects models for subsequent transfer-learning adaptation.

Positional Arguments:
    DATAPATH                            Path for the pre-processed EEG signals file
    MODELPATH                           Path to folder containing the baseline models
    OUTPATH                             Path to folder for saving the selected models and results in

Optional Arguments:
    -gpu GPU                            Set gpu to use, default is 0
```

#### Subject-adaptive transfer learning

To perform adaptation on the selected baseline models, run `train_motor_adapt_all.py`:
```
usage: python train_motor_adapt_all.py [DATAPATH] [MODELPATH] [OUTPATH] [-scheme SCHEME] [-trfrate TRFRATE] [-lr LR] [-gpu GPU]

Evaluate performance of models for each subject and selects models for subsequent transfer-learning adaptation.

Positional Arguments:
    DATAPATH                            Path for the pre-processed EEG signals file
    MODELPATH                           Path to folder containing the selected baseline models for adaptation
    OUTPATH                             Path to folder for saving the adaptation results in

Optional Arguments:
    -scheme SCHEME                      Set scheme which determines layers of the model to be frozen
    -trfrate TRFRATE                    Set amount of target subject data to be used for subject-adaptive transfer learning
    -lr LR                              Set the learning rate of the transfer learning
    -gpu GPU                            Set gpu to use, default is 0
```
-----

### Inner Speech

Conditions = Inner Speech

Classes = "Arriba/Up", "Abajo/Down", "Derecha/Right", "Izquierda/Left"

#### Obtain the raw dataset

Download the inner speech raw dataset from the resources above, save them to the save directory as the main folder.

#### Training the classifier

To perform subject-independent meta-learning on chosen subject, run `train_speech_LOSO.py`. Meta learning may also be utilized.
```
usage: python train_speech_LOSO.py [ROOTDIR] [--META] [-subj SUBJ]

Trains a baseline classifier for a chosen subject either using baseline backpropagation or meta-learning.

Positional Arguments:
    ROOTDIR                             Root directory path to the folder containing the EEG data

Optional Arguments:
    --meta                              Set to enable meta-learning, default meta-learning is switched off
    -subj SUBJ                          Set subject to perform transfer learning adaptation on
```

To train classifiers for all subjects, run `train_speech_LOSO_all.py`. Contains the same arguments as `train_speech_LOSO.py` except without the subject argument. Example usage to train meta-learning for all subjects:
```
python train_speech_LOSO_all.py ROOTDIR --meta
```

#### Subject-adaptive transfer learning

To perform subject-adaptive transfer learning on chosen subject, run `train_speech_adapt.py`. Meta learning may also be utilized.
```
usage: python train_adapt_all.py [ROOTDIR] [MODELPATH] [OUTPATH] [--META] [-scheme SCHEME] [-trfrate TRFRATE] [-lr LR] [-gpu GPU] [-subj SUBJ]

Performs subject-adaptive transfer learning on a subject with or without meta-learning.

Positional Arguments:
    ROOTDIR                             Root directory path to the folder containing the EEG data
    MODELPATH                           Path to folder containing the baseline models for adaptation
    OUTPATH                             Path to folder for saving the adaptation results in

Optional Arguments:
    --meta                              Set to enable meta-learning, default meta-learning is switched off
    -scheme SCHEME                      Set scheme which determines layers of the model to be frozen
    -trfrate TRFRATE                    Set amount of target subject data to be used for subject-adaptive transfer learning
    -lr LR                              Set the learning rate of the transfer learning
    -gpu GPU                            Set gpu to use, default is 0
    -subj SUBJ                          Set subject to perform transfer learning adaptation on
```

To perform subject-adaptive transfer learning on all subjects, run `train_speech_adapt_all.py`. Meta learning may also be utilized. Contains the same arguments as `train_speech_adapt.py` except without the subject argument.

To recreate for meta-learning subject-adaptation, run:
```
python train_speech_adapt_all.py ROOTDIR MODELPATH OUTPATH --meta -scheme 1 -trfrate 40
```

For normal subject-adaptation:
```
python train_speech_adapt_all.py ROOTDIR MODELPATH OUTPATH -scheme 1 -trfrate 40
```

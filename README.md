
# Subject-Independent Meta-Learning for EEG-based Motor Imagery and Inner Speech Classification

## Results Overview

### Motor Imagery

### Inner Speech

## Resources

Motor Imagery Raw Dataset: [Link](http://gigadb.org/dataset/100542)

The pre-processing code of the dataset and a variation of training of the baseline models is provided by Kaishuo et al.: [Link](https://github.com/zhangks98/eeg-adapt)

Inner Speech Dataset: [Link](https://openneuro.org/datasets/ds003626/versions/2.1.2)

Original Inner Speech Github: [Link](https://github.com/N-Nieto/Inner_Speech_Dataset)

## Dependencies

## Run

### Motor Imagery

#### Obtain the raw dataset

Download the motor imagery raw dataset from the resources above, and save them to the same `$source` folder. To conserve space, you may only download files that ends with `EEG_MI.mat`.

#### Pre-process the raw dataset

The following command will read the raw dataset from the `$source` folder, and output the pre-processed data `KU_mi_smt.h5` into the `$target` folder.

```
python preprocess_h5_smt.py $source $target
```
#### Training the classifier

===================

### Inner Speech

#### Obtain the raw dataset

Download the inner speech raw dataset from the resources above, save them to the save directory as the main folder.

#### Training the classifier

Inner Speech Dataset.

Download from inner speech dataset link, put all files into the same folder

Conditions = Inner Speech

Classes = "Arriba/Up", "Abajo/Down", "Derecha/Right", "Izquierda/Left"

Run train_LOSO.py -subj X --meta or no --meta accordinly

Run train_adapt.py following trained subjcts of meta

To recreate

run train_adapt_all.py
meta: on shuffle_meta, scheme 1 transfer rate 40 with meta learning
transfer-learning: shuffle_LOSO, scheme 1 transfer rate 40 without meta learning

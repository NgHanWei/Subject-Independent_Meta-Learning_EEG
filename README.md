
# Subject-Independent Meta-Learning for EEG-based Motor Imagery and Inner Speech Classification
Code for
### Subject-independent meta-learning framework towards optimal training of EEG-based classifiers

## Results Overview

### Motor Imagery

| Methodology | Mean (SD) | Median | Range (Max-Min) |
|-|-|-|-|
| Subject-Independent CNN| 74.15 (±15.83) | 75.00 | 60.00 (100.00-40.00) |
| Subject-Independent Deep CNN| 84.19 (±9.98) | 84.50 | 47.50 (99.50-52.00) |
| Subject-Adaptive Deep CNN | 86.89 (±11.41) | 88.50 | 44.00 (100.00-56.00) |
| **Subject-Independent Meta-Learning** | 87.20 (±10.54) | 89.50 | 41.00 (100.00-59.00) |

### Inner Speech

| Subject     | EEGNet | DeepConvNet | Transfer Learning | Meta-Learning | **Transfer Meta-Learning** |
|-------------|--------|-------------|-------------------|---------------|----------------------------|
| 1           |  30.00 |    23.33    |       32.00       |     24.67     |            30.00           |
| 2           |  30.41 |    26.32    |       29.47       |     24.74     |            32.11           |
| 3           |  30.56 |    26.15    |       32.31       |     26.92     |            35.38           |
| 4           |  23.75 |    26.84    |       27.89       |     23.16     |            30.00           |
| 5           |  30.00 |    25.79    |       30.00       |     25.79     |            30.00           |
| 6           |  27.78 |    22.89    |       32.53       |     21.08     |            32.53           |
| 7           |  30.41 |    24.74    |       28.42       |     27.89     |            28.95           |
| 8           |  32.11 |    26.67    |       28.00       |     26.00     |            29.33           |
| 9           |    -   |    23.16    |       27.37       |     28.95     |            31.58           |
| 10          |    -   |    26.32    |       34.21       |     26.32     |            31.58           |
| **Average** |  29.67 |    24.40    |       30.22       |     25.55     |            31.15           |

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

Please cite
```
@article{ng2024subject,
  title={Subject-independent meta-learning framework towards optimal training of EEG-based classifiers},
  author={Ng, Han Wei and Guan, Cuntai},
  journal={Neural Networks},
  pages={106108},
  year={2024},
  publisher={Elsevier}
}
```

CBCR License 1.0

Copyright 2023 Centre for Brain Computing Research (CBCR)

Redistribution and use for non-commercial purpose in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

4. In the event that redistribution and/or use for commercial purpose in source or binary forms, with or without modification is required, please contact the contributor(s) of the work.

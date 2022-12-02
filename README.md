
# Subject-Independent Meta-Learning for EEG-based Motor Imagery and Inner Speech Classification

## Results Overview

### Motor Imagery

### Inner Speech

## Resources

Inner Speech Dataset: [Link](https://openneuro.org/datasets/ds003626/versions/2.1.2)

Original Inner Speech Github: [Link](https://github.com/N-Nieto/Inner_Speech_Dataset)

## Dependencies

## Run

### Motor Imagery

### Inner Speech

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

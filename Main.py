import random

from Training import train
from dataloader import load_data, balanced_sampler
from Settings import DATASET_TYPE, CATEGORIES

data_dir = "./" + DATASET_TYPE + "/"
dataset, cnt = load_data(data_dir)

# test with happiness and anger
images = balanced_sampler(dataset, cnt, emotions=CATEGORIES)
display_index = 0

X = []
y = []
for i, category in enumerate(CATEGORIES):
    images_in_cat = images[category]
    random.shuffle(images_in_cat)
    X += images_in_cat
    y += [i] * len(images[category])

# get my feature - label pairs zipped together
all_data = list(zip(X, y))

# randomize the dataset so I can fold properly

train(all_data)

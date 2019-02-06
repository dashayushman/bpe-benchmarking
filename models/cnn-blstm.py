import csv

import numpy as np
from keras import backend as K
from tqdm import tqdm
from bpemb import BPEmb
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D


batch_size = 32
embedding_dims = 50
filters = 300
kernel_size = 3

hidden_dims = 500
epochs = 1000
bpemb_dims = 50


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


bpemb_en = BPEmb(lang="en", dim=bpemb_dims)
le = preprocessing.LabelEncoder()

with open("../datasets/WebApplication/train.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    max_len = 0
    y = []
    for row in tqdm(reader):
        y.append(row[1])
        sample_len = len(bpemb_en.encode(row[0]))
        max_len = sample_len if sample_len > max_len else max_len
max_len = max_len + 1
print(max_len)
encoded_labels = le.fit_transform(y)
print(le.classes_)

y = to_categorical(encoded_labels, num_classes=len(le.classes_))


x = None

with open("../datasets/WebApplication/train.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in tqdm(reader):
        embeddings = bpemb_en.embed(row[0])
        padding_vec = np.zeros((max_len - embeddings.shape[0], bpemb_dims))
        padded = np.vstack((embeddings, padding_vec))
        padded = np.expand_dims(padded, axis=0)
        if x is not None:
            x = np.vstack((x, padded))
        else:
            x = padded


x_test = None
y_test = []
with open("../datasets/WebApplication/test.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in tqdm(reader):
        embeddings = bpemb_en.embed(row[0])
        y_test.append(row[1])
        padding_vec = np.zeros((max_len - embeddings.shape[0], bpemb_dims))
        padded = np.vstack((embeddings, padding_vec))
        padded = np.expand_dims(padded, axis=0)
        if x_test is not None:
            x_test = np.vstack((x_test, padded))
        else:
            x_test = padded
y_test_enc = le.transform(y_test)
y_test = to_categorical(y_test_enc, num_classes=len(le.classes_))
print(x_test.shape, y_test.shape)
print('Build model...')
model = Sequential()

model.add(Dropout(0.5))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(len(le.classes_)))
model.add(Activation('softmax'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[f1, "accuracy"])
model.fit(x, y,
          batch_size=batch_size,
          epochs=epochs, validation_data=(x_test, y_test))

import random
import csv

import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

import cv2 as cv

import numpy as np




IMG_WIDTH = 256
IMG_HEIGHT = 256


get_image = lambda file_name: cv.resize(
    cv.imread(
        file_name,
        cv.IMREAD_GRAYSCALE
    ),
    (IMG_WIDTH, IMG_HEIGHT)
)



dec_row = lambda enc_row: (
    get_image("./dataset/" + str(enc_row[:1][0])),
    np.array(
        list(
            map(
                lambda r: float(r),
                enc_row[1:]
            )
        )
    )
)



def read_csv_data(f_name):
    with open(f_name) as f:
        reader = csv.reader(f)

        return list(map(
                lambda r: dec_row(r),
                reader
        ))




get_column = lambda rows, column: list(map(
    lambda r: r[column],
    rows
))

get_X = lambda row: get_column(row, 0)
get_y = lambda row: get_column(row, 1)



def get_train_test(rows, split_percentage):
    random.shuffle(rows)

    split_index = int(len(rows) * split_percentage)

    test = rows[:split_index]

    train = rows[split_index:]

    return get_X(train), get_y(train), get_X(test), get_y(test)



rows = read_csv_data("./dataset/data.csv")

X_train, y_train, X_test, y_test = get_train_test(rows, 0.2)

X_train = np.array(X_train).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
y_train = np.array(y_train)
X_test = np.array(X_test).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
y_test = np.array(y_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



model = Sequential([
    Conv2D(filters=64, kernel_size=(4, 4), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(filters=32, kernel_size=(4, 4), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(filters=64, kernel_size=(4, 4), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(filters=64, kernel_size=(4, 4), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(300, activation='sigmoid')
])


print(model.summary())


model.compile(
    optimizer='adam',
    loss='poisson',
    metrics=['accuracy']
)


model.fit(X_train, y_train, epochs=3)


import random
import csv

import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

import numpy as np

dec_row = lambda enc_row: (
    enc_row[:1][0],
    list(
        map(
            lambda r: float(r),
            enc_row[1:]
        )
    )
)



def read_csv_data(f_name):
    with open(f_name) as f:
        reader = csv.reader(f)

        return list(
            map(
                lambda r: dec_row(r),
                reader
            )
        )



get_image = lambda file_name: (
    image.img_to_array(
        image.load_img(
            file_name,
            target_size=(256, 256, 3)
        )
    ) / 256.0
)


populate_imgs = lambda rows: list(map(
    lambda r: (
        get_image("./dataset/" + str(r[0])),
        r[1]
    ),
    rows
))


# NUMPY CODs
get_column = lambda rows, column: list(map(
    lambda r: np.array(r[column]),
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

rows = populate_imgs(rows)

X_train, y_train, X_test, y_test = get_train_test(rows, 0.2)


img_width = 256
img_height = 256

model = Sequential([
    Conv2D(filters=16, kernel_size=(4, 4), activation="relu", input_shape=(img_width, img_height, 3)),
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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=2)


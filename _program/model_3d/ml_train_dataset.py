import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image


import random
import csv
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



enc_row = lambda dec_row: (
    get_image("./dataset/" + str(dec_row[:1][0])),
    np.array(
        list(
            map(
                lambda r: float(r),
                dec_row[1:]
            )
        )
    )
)



def read_csv_data(f_name):
    with open(f_name) as f:
        reader = csv.reader(f)

        return list(map(
                lambda r: enc_row(r),
                reader
        ))




get_column = lambda rows, column: list(map(
    lambda r: r[column],
    rows
))

get_X = lambda row: get_column(row, 0)
get_y = lambda row: get_column(row, 1)




def shuffle_split(rows, split_percentage):
    random.shuffle(rows)

    split_index = int(len(rows) * split_percentage)

    test = rows[:split_index]

    train = rows[split_index:]

    return train, test



def get_train_test(rows, split_percentage):
    train, test = shuffle_split(rows, split_percentage)

    X_train, y_train, X_test, y_test = get_X(train), get_y(train), get_X(test), get_y(test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)


    X_train = np.array(X_train).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
    X_test = np.array(X_test).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)

    return X_train, y_train, X_test, y_test





rows = read_csv_data("./dataset/data.csv")

X_train, y_train, X_test, y_test = get_train_test(rows, 0.1)


get_c = lambda col: np.array(list(map(
    lambda y: y[:3],
    col
)))
y_train = get_c(y_train)
y_test = get_c(y_test)



# Improve machine learning architecture
model = Sequential([
    Conv2D(filters=256, kernel_size=(8, 8), activation="sigmoid", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    MaxPooling2D(pool_size=(4, 4)),
    Dropout(0.25),
    Conv2D(filters=128, kernel_size=(8, 8), activation="sigmoid"),
    MaxPooling2D(pool_size=(4, 4)),
    Dropout(0.25),
    Conv2D(filters=64, kernel_size=(4, 4), activation="sigmoid"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(filters=32, kernel_size=(4, 4), activation="sigmoid"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation="sigmoid"),
    Dropout(0.5),
    Dense(128, activation="sigmoid"),
    Dropout(0.5),
    Dense(3, activation="sigmoid")
])
#print(model.summary())

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredLogarithmicError(reduction="auto"),
    metrics=[
        tf.keras.metrics.MeanSquaredError()
    ]
)


model.fit(x=X_train, y=y_train, epochs=500, validation_data=(X_test, y_test,))
model_dir = '../my_model'
model.save(model_dir)






"""
def get_camera_data_prediction(model, image):
    return list(model.predict(
        np.array([image])
    )[0])

new_model = tf.keras.models.load_model(model_dir)
#new_model.summary()
pred1 = get_camera_data_prediction(new_model, X_test[0])
print(pred1)
"""


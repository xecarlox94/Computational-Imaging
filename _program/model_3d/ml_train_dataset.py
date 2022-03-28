import tensorflow as tf
import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image


import datetime
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



def shuffle_split(rows, split_percentage):
    random.shuffle(rows)

    split_index = int(len(rows) * split_percentage)

    test = rows[:split_index]

    train = rows[split_index:]

    return train, test



get_X = lambda row: get_column(row, 0)
get_y = lambda row: get_column(row, 1)

get_image_arr = lambda lst: np.array(lst).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)



def get_train_test(rows, split_percentage):
    train, test = shuffle_split(rows, split_percentage)

    X_train, y_train, X_test, y_test = get_X(train), get_y(train), get_X(test), get_y(test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train = get_image_arr(X_train)
    X_test = get_image_arr(X_test)

    return X_train, y_train, X_test, y_test




rows = read_csv_data("./dataset/data.csv")



X_train, y_train, X_test, y_test = get_train_test(rows, 0.1)



gx = lambda r: r[3:]
gy = lambda r: r[3:]


get_Xx = lambda y_array: np.array(list(map(
    gx,
    y_array
)))

get_yy = lambda y_array: np.array(list(map(
    gy,
    y_array
)))


Xx_test = get_Xx(y_test)

Xx_train = get_Xx(y_train)

y_train = get_yy(y_train)

y_test = get_yy(y_test)



layers = [
    Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    Conv2D(filters=20, kernel_size=(2, 2), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),
    Conv2D(filters=20, kernel_size=(2, 2), activation="relu"),
    MaxPooling2D(pool_size=(4, 4)),
    Dropout(0.5),
    Flatten(),
    Dense(1024, activation="sigmoid"),
    Dropout(0.5),
]

out = Dense(90, activation="sigmoid")


more = True
more = False

if more == True:
    model_sec = Sequential([
        Input(shape=(3,)),
        Dense(3, activation="sigmoid"),
    ])

    model = Sequential(layers)

    concat = concatenate([model.output, model_sec.output])

    m = out(concat)

    model = Model(inputs=[model.input, model_sec.input], outputs=m)


else:
    model = Sequential(
        layers + [out]
    )


#print(model.summary())


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredLogarithmicError(reduction="auto"),
    metrics=[
        tf.keras.metrics.MeanSquaredError(),
        #tf.keras.metrics.MeanAbsoluteError()
    ]
)


log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.fit(
    x=[
        X_train,
        #Xx_train
    ],
    y=y_train,
    epochs=100,
    validation_data=(
        [
            X_test,
            #Xx_test
        ],
        y_test
    ),
    callbacks=[
        #tensorboard_callback
    ]
)


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


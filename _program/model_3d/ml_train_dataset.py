
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






def train_model(data, params):

    X_train, y_train, X_test, y_test = data

    (
        model_name,
        epochs,
        output_size,
        Xx_input_len,
        num_filters_1,
        kernel_size_1,
        poolsize_1,
        dropout_1,
        num_filters_2,
        kernel_size_2,
        poolsize_2,
        dropout_2,
        num_dense_3,
        dropout_3,
        num_dense_4,
        dropout_4,
    ) = params


    convolution_layers = [
        Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
        Conv2D(filters=num_filters_1, kernel_size=(kernel_size_1, kernel_size_1), activation="relu"),
        MaxPooling2D(pool_size=(poolsize_1, poolsize_1)),
        Dropout(dropout_1),
        Conv2D(filters=num_filters_2, kernel_size=(kernel_size_2, kernel_size_2), activation="relu"),
        MaxPooling2D(pool_size=(poolsize_2, poolsize_2)),
        Dropout(dropout_2),
        Flatten(),
        Dense(num_dense_3, activation="sigmoid"),
        Dropout(dropout_3),
    ]

    out = Dense(output_size, activation="sigmoid")


    if Xx_input_len > 0:

        model_sec = Sequential([
            Input(shape=(Xx_input_len,)),
            Dense(Xx_input_len, activation="sigmoid"),
        ])

        m = Sequential(convolution_layers + [
                Dense(num_dense_4, activation="sigmoid"),
                Dropout(dropout_4),
            ]
        )

        model = Model(
            inputs=[
                m.input,
                model_sec.input
            ],
            outputs=out(
                concatenate([
                    m.output,
                    model_sec.output
                ])
            )
        )

        get_Xx = lambda y_array: np.array(list(map(
            lambda r: r[: Xx_input_len],
            y_array
        )))

        model_fit_x = [
            get_Xx(y_train)
        ]

        model_fit_y = [
            get_Xx(y_test)
        ]


    else:

        model = Sequential(
            convolution_layers + [out]
        )

        model_fit_x = []
        model_fit_y = []


    get_yy = lambda y_array: np.array(list(map(
        lambda r: r[Xx_input_len : output_size + Xx_input_len ],
        y_array
    )))

    y_train = get_yy(y_train)
    y_test = get_yy(y_test)


    compile_fit_save_model(
        model,
        (
            X_train, y_train, X_test, y_test, model_fit_x, model_fit_y
        ),
        epochs,
        model_name,
        (
            model_name + get_ml_arch(
                Xx_input_len,
                num_filters_1,
                kernel_size_1,
                poolsize_1,
                dropout_1,
                num_filters_2,
                kernel_size_2,
                poolsize_2,
                dropout_2,
                num_dense_3,
                dropout_3,
                num_dense_4,
                dropout_4,
            ) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )




def get_params(
        model_name, epochs, output_size, Xx_input_len,
        # model params: Input 1
        num_filters_1=20,
        kernel_size_1=2,
        poolsize_1=2,
        dropout_1=0.1,
        num_filters_2=20,
        kernel_size_2=2,
        poolsize_2=4,
        dropout_2=0.1,
        num_dense_3=1024,
        dropout_3=0.5,
        # model params: 2 input
        num_dense_4=512,
        dropout_4=0.5
    ):
    return model_name, epochs, output_size, Xx_input_len, num_filters_1, kernel_size_1, poolsize_1, dropout_1, num_filters_2, kernel_size_2, poolsize_2, dropout_2, num_dense_3, dropout_3, num_dense_4, dropout_4





def get_ml_arch(
        Xx_input_len,
        num_filters_1,
        kernel_size_1,
        poolsize_1,
        dropout_1,
        num_filters_2,
        kernel_size_2,
        poolsize_2,
        dropout_2,
        num_dense_3,
        dropout_3,
        num_dense_4=0,
        dropout_4=0,
    ):
    arch='__num_filters_1:_' + str(num_filters_1) + ',_kernel_size_1:_' + str(kernel_size_1) + ',_poolsize_1:_' + str(poolsize_1) + ',_dropout_1:_' + str(dropout_1) + ',_num_filters_2:_' + str(num_filters_2) + ',_kernel_size_2:_' + str(kernel_size_2) + ',_poolsize_2:_' + str(poolsize_2) + ',_dropout_2:_' + str(dropout_2) + ',_num_dense_3:_' + str(num_dense_3) + ',_dropout_3:_' + str(dropout_3)
    if Xx_input_len > 0:
        arch = arch + ',_Xx_input_len:_' + str(Xx_input_len) + ',_num_dense_4:_' + str(num_dense_4) + ',_dropout_4:_' + str(dropout_4)
    return arch + '__'




def compile_fit_save_model(
        model,
        data,
        epochs,
        model_name,
        tensor_board_name
    ):

    X_train, y_train, X_test, y_test, model_fit_x, model_fit_y = data

    #print(model.summary())

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredLogarithmicError(reduction="auto"),
        metrics=[
            tf.keras.metrics.MeanSquaredError(),
            #tf.keras.metrics.MeanAbsoluteError()
        ]
    )

    log_dir = "./logs/fit/" + tensor_board_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        x=[X_train] + model_fit_x,
        y=y_train,
        epochs=epochs,
        validation_data=(
            [X_test] + model_fit_y,
            y_test
        ),
        callbacks=[
            tensorboard_callback
        ]
    )

    model.save('../models/' + model_name)





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




rows = read_csv_data("./dataset/data.csv")






train_model(
    get_train_test(rows, 0.1),
    get_params(
        #model_name,
        "my_model",
        #epochs,
        10,
        #output_size,
        12,
        #Xx_input_len,
        3,
        # model params: Input 1
        num_filters_1=20,
        kernel_size_1=2,
        poolsize_1=2,
        dropout_1=0.1,
        num_filters_2=20,
        kernel_size_2=2,
        poolsize_2=4,
        dropout_2=0.1,
        num_dense_3=1024,
        dropout_3=0.5,
        # model params: 2 input
        num_dense_4=512,
        dropout_4=0.5
    )
)

train_model(
    get_train_test(rows, 0.1),
    get_params(
        #model_name,
        "my_model",
        #epochs,
        10,
        #output_size,
        12,
        #Xx_input_len,
        3,
        # model params: Input 1
        num_filters_1=20,
        kernel_size_1=2,
        poolsize_1=2,
        dropout_1=0.1,
        num_filters_2=20,
        kernel_size_2=2,
        poolsize_2=4,
        dropout_2=0.1,
        num_dense_3=1024,
        dropout_3=0.5,
        # model params: 2 input
        num_dense_4=512,
        dropout_4=0.5
    )
)

train_model(
    get_train_test(rows, 0.1),
    get_params(
        #model_name,
        "my_model",
        #epochs,
        10,
        #output_size,
        78,
        #Xx_input_len,
        15,
        # model params: Input 1
        num_filters_1=20,
        kernel_size_1=2,
        poolsize_1=2,
        dropout_1=0.1,
        num_filters_2=20,
        kernel_size_2=2,
        poolsize_2=4,
        dropout_2=0.1,
        num_dense_3=1024,
        dropout_3=0.5,
        # model params: 2 input
        num_dense_4=512,
        dropout_4=0.5
    )
)


train_model(
    get_train_test(rows, 0.1),
    get_params(
        #model_name,
        "my_model",
        #epochs,
        10,
        #output_size,
        3,
        #Xx_input_len,
        0,
        # model params: Input 1
        num_filters_1=20,
        kernel_size_1=2,
        poolsize_1=2,
        dropout_1=0.1,
        num_filters_2=20,
        kernel_size_2=2,
        poolsize_2=4,
        dropout_2=0.1,
        num_dense_3=1024,
        dropout_3=0.5,
        # model params: 2 input
        num_dense_4=512,
        dropout_4=0.5
    )
)

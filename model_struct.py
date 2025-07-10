import tensorflow as tf 
from keras.layers import Input, Dense,Conv2D,Flatten,MaxPool2D,Dropout
from keras.models import Model
from keras.losses import Huber

def build_model():

    k_size =(3,3)
    inputs = Input(shape=(96,96,3))

    x = Conv2D(32,k_size,activation ='relu',padding = 'same')(inputs)
    x = Conv2D(32,k_size,activation ='relu',padding = 'same')(x)
    x = Conv2D(32,k_size,activation ='relu',padding= 'same')(x)
    x = MaxPool2D()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64,k_size,activation ='relu',padding = 'same')(x)
    x = Conv2D(64,k_size,activation ='relu',padding= 'same')(x)
    x = Conv2D(64,k_size,activation ='relu',padding= 'same')(x)
    x = MaxPool2D()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128,k_size,activation ='relu',padding = 'same')(x)
    x = Conv2D(128,k_size,activation ='relu',padding= 'same')(x)
    x = Conv2D(128,k_size,activation ='relu',padding = 'same')(x)
    x = MaxPool2D()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256,k_size,activation ='relu',padding = 'same')(x)
    x = Conv2D(256,k_size,activation ='relu',padding = 'same')(x)
    x = Conv2D(256,k_size,activation ='relu',padding = 'same')(x)
    x = MaxPool2D()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    shared = Dense(256, activation='relu')(x)
    shared = Dropout(0.3)(shared)

    # Gender branch
    g = Dense(128, activation='relu')(shared)
    g = Dropout(0.3)(g)
    gender_output = Dense(1, activation='sigmoid', name='gender')(g)

    # Age branch
    a = Dense(128, activation='relu')(shared)
    a = Dropout(0.3)(a)
    age_output = Dense(1, activation='linear', name='age')(a)
     
    model = Model(inputs=inputs, outputs=[gender_output, age_output])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss={'gender': 'binary_crossentropy', 'age': Huber(delta=1.0)},
        loss_weights={'gender': 1.0, 'age': 5.0},
        metrics={'gender': 'accuracy', 'age': 'mae'}
    )

    return model


# Model: "model"
# __________________________________________________________________________________________________
#  Layer (type)                Output Shape                 Param #   Connected to
# ==================================================================================================
#  input_1 (InputLayer)        [(None, 96, 96, 3)]          0         []

#  conv2d (Conv2D)             (None, 96, 96, 32)           896       ['input_1[0][0]']

#  conv2d_1 (Conv2D)           (None, 96, 96, 32)           9248      ['conv2d[0][0]']

#  conv2d_2 (Conv2D)           (None, 96, 96, 32)           9248      ['conv2d_1[0][0]']

#  max_pooling2d (MaxPooling2  (None, 48, 48, 32)           0         ['conv2d_2[0][0]']
#  D)

#  dropout (Dropout)           (None, 48, 48, 32)           0         ['max_pooling2d[0][0]']

#  conv2d_3 (Conv2D)           (None, 48, 48, 64)           18496     ['dropout[0][0]']

#  conv2d_4 (Conv2D)           (None, 48, 48, 64)           36928     ['conv2d_3[0][0]']

#  conv2d_5 (Conv2D)           (None, 48, 48, 64)           36928     ['conv2d_4[0][0]']

#  max_pooling2d_1 (MaxPoolin  (None, 24, 24, 64)           0         ['conv2d_5[0][0]']
#  g2D)

#  dropout_1 (Dropout)         (None, 24, 24, 64)           0         ['max_pooling2d_1[0][0]']

#  conv2d_6 (Conv2D)           (None, 24, 24, 128)          73856     ['dropout_1[0][0]']

#  conv2d_7 (Conv2D)           (None, 24, 24, 128)          147584    ['conv2d_6[0][0]']

#  conv2d_8 (Conv2D)           (None, 24, 24, 128)          147584    ['conv2d_7[0][0]']

#  max_pooling2d_2 (MaxPoolin  (None, 12, 12, 128)          0         ['conv2d_8[0][0]']
#  g2D)

#  dropout_2 (Dropout)         (None, 12, 12, 128)          0         ['max_pooling2d_2[0][0]']

#  conv2d_9 (Conv2D)           (None, 12, 12, 256)          295168    ['dropout_2[0][0]']

#  conv2d_10 (Conv2D)          (None, 12, 12, 256)          590080    ['conv2d_9[0][0]']

#  conv2d_11 (Conv2D)          (None, 12, 12, 256)          590080    ['conv2d_10[0][0]']

#  max_pooling2d_3 (MaxPoolin  (None, 6, 6, 256)            0         ['conv2d_11[0][0]']
#  g2D)

#  dropout_3 (Dropout)         (None, 6, 6, 256)            0         ['max_pooling2d_3[0][0]']

#  flatten (Flatten)           (None, 9216)                 0         ['dropout_3[0][0]']

#  dense (Dense)               (None, 256)                  2359552   ['flatten[0][0]']

#  dropout_4 (Dropout)         (None, 256)                  0         ['dense[0][0]']

#  gender (Dense)              (None, 1)                    257       ['dropout_4[0][0]']

#  age (Dense)                 (None, 1)                    257       ['dropout_4[0][0]']

# ==================================================================================================
# Total params: 4316162 (16.46 MB)
# Trainable params: 4316162 (16.46 MB)
# Non-trainable params: 0 (0.00 Byte)
# __________________________________________________________________________________________________


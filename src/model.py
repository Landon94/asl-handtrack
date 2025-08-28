from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Flatten,Conv2D,MaxPooling2D,Dense,Input,BatchNormalization
from typing import Tuple
from tensorflow.keras import Model


def create_model(input_size: tuple[int,int,int], num_classes: int=27) -> Model:
    model = Sequential()

    model.add(Input(input_size))
    
    model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes,activation='softmax'))

    return model
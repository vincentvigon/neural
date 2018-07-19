import keras
from keras.models import *
from keras.layers import Input, Concatenate, concatenate, Conv3D, MaxPooling3D, UpSampling3D, Cropping3D,Dropout, \
    BatchNormalization, ZeroPadding3D, add, ELU, Activation, PReLU

from keras.optimizers import *
from keras.callbacks import *
from keras import backend as keras
from keras.utils import plot_model
from tensorflow.python.client import device_lib
from keras import regularizers

from perdro.A_generateData import dataGenerator




def Model_Vnet3D_keras(patch_size:int,dropout:float,nbCouche:int):

    input= Input(batch_shape=(None,patch_size,patch_size,patch_size, 1))
    print("input.get_shape():",input.get_shape())
    output=Model_Vnet3D_2(patch_size,dropout)(input)
    model = Model(inputs=input, outputs=output)
    model.summary()
    #plot_model(model, show_shapes=True, to_file='./out/VNetBN.png')
    return model



class Model_Vnet3D:

    def __init__(self,patch_size:int,dropout:float):

        self.patch_size = patch_size
        self.dropout=dropout

    def __call__(self,X):

        print("X.get_shape():",X.get_shape())

        deph = 16

        Y = Conv3D(deph, 5, activation='relu', padding='same', kernel_initializer='he_normal')(X)
        Y = Dropout(self.dropout)(Y)
        Y = Conv3D(deph * 2, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        jump1 = Dropout(self.dropout)(Y)
        Y = Conv3D(deph * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                   strides=(2, 2, 2))(jump1)
        Y = Conv3D(deph * 2, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Dropout(self.dropout)(Y)
        jump2 = Conv3D(deph * 2, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Conv3D(deph * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                   strides=(2, 2, 2))(jump2)
        Y = Conv3D(deph * 4, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Dropout(self.dropout)(Y)
        jump3 = Conv3D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Conv3D(deph * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                   strides=(2, 2, 2))(jump3)
        Y = Conv3D(deph * 16, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Conv3D(deph * 16, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Dropout(self.dropout)(Y)
        Y = Conv3D(deph * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling3D(size=2)(Y))
        Y = Concatenate()([jump3, Y])
        Y = Conv3D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Conv3D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Dropout(self.dropout)(Y)
        Y = Conv3D(deph * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling3D(size=2)(Y))
        Y = Concatenate()([jump2, Y])
        Y = Conv3D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Conv3D(deph * 2, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Dropout(self.dropout)(Y)
        Y = Conv3D(deph * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling3D(size=2)(Y))
        Y = Concatenate()([jump1, Y])
        Y = Conv3D(64, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Conv3D(64, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Dropout(self.dropout)(Y)
        Y = Conv3D(1, 1, activation='sigmoid')(Y)

        return Y



class Model_Vnet3D_1:

    def __init__(self,patch_size:int,dropout:float,nbCouche:int):

        self.patch_size = patch_size
        self.dropout=dropout
        self.nbCouche=nbCouche

    def __call__(self,X):

        print("X.get_shape():",X.get_shape())

        deph = 16
        Y=X

        """ (32,32) depth=16 puis 32 """
        Y = Conv3D(deph, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Dropout(self.dropout)(Y)
        Y = Conv3D(deph * 2, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        jump1= Y = Dropout(self.dropout,name="jump1")(Y)


        """ (16,16) depth= 32 """
        Y = Conv3D(deph * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal',strides=(2, 2, 2))(Y)

        Y = Conv3D(deph * 2, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Dropout(self.dropout)(Y)        
        jump2=Y = Conv3D(deph * 2, 5, activation='relu', padding='same', kernel_initializer='he_normal',name="jump2")(Y)
        
        """ (8,8) depth= 64 """
        Y = Conv3D(deph * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal',strides=(2, 2, 2))(Y)
        Y = Conv3D(deph * 4, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Dropout(self.dropout)(Y) 
        jump3=Y = Conv3D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal',name="jump3")(Y)

        """ (4,4) depth=128 """
        Y = Conv3D(deph * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal',strides=(2, 2, 2))(Y)
        Y = Conv3D(deph * 16, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y,name="bottle1")

        """ (4,4) depth=256 """
        Y = Conv3D(deph * 16, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Dropout(self.dropout,name="bottle2")(Y)

        """ (8,8) depth=  128 -> 192 -> 64 """
        Y = Conv3D(deph * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=2)(Y))
        Y = Concatenate()([jump3, Y])
        Y = Conv3D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Conv3D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Dropout(self.dropout)(Y)
        Y = Conv3D(deph * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=2)(Y))
        
        Y = Concatenate()([jump2, Y])
        Y = Conv3D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Conv3D(deph * 2, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Dropout(self.dropout)(Y)
        Y = Conv3D(deph * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=2)(Y))
        
        Y = Concatenate()([jump1, Y])
        Y = Conv3D(64, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Conv3D(64, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Dropout(self.dropout)(Y)
        Y = Conv3D(1, 1, activation='sigmoid')(Y)

        return Y


class Model_Vnet3D_2:

    def __init__(self, patch_size: int, dropout: float, nbCouche: int):
        self.patch_size = patch_size
        self.dropout = dropout
        self.nbCouche = nbCouche

    def __call__(self, X):
        print("X.get_shape():", X.get_shape())

        deph = 16
        Y = X

        """ (32,32) depth=16 puis 32 """
        Y = Conv3D(deph, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Dropout(self.dropout)(Y)
        Y = Conv3D(deph * 2, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        jump1 = Y = Dropout(self.dropout, name="jump1")(Y)

        """ (16,16) depth= 32 """
        Y = Conv3D(deph * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2, 2, 2))(Y)

        Y = Conv3D(deph * 2, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Dropout(self.dropout)(Y)
        jump2 = Y = Conv3D(deph * 2, 5, activation='relu', padding='same', kernel_initializer='he_normal',
                           name="jump2")(Y)

        """ (8,8) depth= 64 """
        Y = Conv3D(deph * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2, 2, 2))(Y)
        Y = Conv3D(deph * 4, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Dropout(self.dropout)(Y)
        jump3 = Y = Conv3D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal',
                           name="jump3")(Y)

        """ (4,4) depth=128 """
        Y = Conv3D(deph * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal', strides=(2, 2, 2))(Y)
        Y = Conv3D(deph * 16, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y, name="bottle1")

        """ (4,4) depth=256 """
        Y = Conv3D(deph * 16, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Dropout(self.dropout, name="bottle2")(Y)

        """ (8,8) depth=  128 -> 192 -> 64 """
        Y = Conv3D(deph * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=2)(Y))
        Y = Concatenate()([jump3, Y])
        Y = Conv3D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Conv3D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Dropout(self.dropout)(Y)
        Y = Conv3D(deph * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=2)(Y))

        Y = Concatenate()([jump2, Y])
        Y = Conv3D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Conv3D(deph * 2, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Dropout(self.dropout)(Y)
        Y = Conv3D(deph * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling3D(size=2)(Y))

        Y = Concatenate()([jump1, Y])
        Y = Conv3D(64, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Conv3D(64, 5, padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = Dropout(self.dropout)(Y)
        Y = Conv3D(1, 1, activation='sigmoid')(Y)

        return Y






import tensorflow as tf


if __name__ == '__main__':

    patch_size=32

    # for X,Y in dataGenerator(batch_size=3,patch_size=patch_size):
    #     break
    #
    # print("X.shape:",X.shape)
    # print("Y.shape:",Y.shape)
    #
    # _X=tf.constant(X)
    #
    # out = Model_Vnet3D(patch_size=patch_size,dropout=1)(_X)
    # print("out.get_shape():",out.get_shape())

    Model_Vnet3D_keras(32, dropout= 0.2,nbCouche=4)









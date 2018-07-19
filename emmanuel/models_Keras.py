import keras
from keras.models import *
from keras.layers import Input, Concatenate, Dropout, BatchNormalization, Conv2D, UpSampling2D

from emmanuel.A_generateData import dataGenerator_croix_sur_ile
from ne_ne2.VISU import draw16imgs


def get_keras_model(patch_size:int,dropout:float):

    input= Input(batch_shape=(None,patch_size,patch_size, 1))
    output=Model_interpolation_2D(patch_size,dropout)(input)
    model = Model(inputs=input, outputs=output)
    model.summary()
    #plot_model(model, show_shapes=True, to_file='./out/VNetBN.png')
    return model




class Model_interpolation_2D:

    def __init__(self,patch_size:int,dropout:float):

        self.patch_size = patch_size
        self.dropout=dropout

    def __call__(self,X):

        print("X.get_shape():",X.get_shape())

        deph = 16
        Y=X

        """ (s,s,d*2)  """
        Y = Conv2D(deph*2 , 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Conv2D(deph*2, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Dropout(self.dropout)(Y)
        jump1= Y = Dropout(self.dropout,name="jump1")(Y)


        """ (s/2,s/2,d*4) """
        Y = Conv2D(deph * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal',strides=(2, 2))(Y)
        Y = Conv2D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Dropout(self.dropout)(Y)
        jump2=Y = Conv2D(deph * 2, 5, activation='relu', padding='same', kernel_initializer='he_normal',name="jump2")(Y)


        """ (s/4,s/4,d*8)  """
        Y = Conv2D(deph * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal',strides=(2, 2))(Y)
        Y = Conv2D(deph * 8, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Dropout(self.dropout,name="bottle_neck")(Y)


        """ (s/2,s/2)   d*8 -> d*8+d*4 -> d*4 """
        Y = Conv2D(deph * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=2)(Y))
        Y = Concatenate()([jump2, Y])
        Y = Conv2D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Conv2D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Dropout(self.dropout)(Y)


        """ (s,s)   d*4 -> d*4+d*2-> d*2 """
        Y = Conv2D(deph * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=2)(Y))
        Y = Concatenate()([jump1, Y])
        Y = Conv2D(deph * 4, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = Conv2D(deph * 2, 5, activation='relu', padding='same', kernel_initializer='he_normal')(Y)
        Y = BatchNormalization()(Y)
        Y = Dropout(self.dropout)(Y)


        Y = Conv2D(1, 1)(Y)

        return Y



import tensorflow as tf
if __name__ == '__main__':

    patch_size=32

    X,Y=None,None
    for X,Y in dataGenerator_croix_sur_ile(batch_size=16, patch_size=patch_size):
        break

    print("X.shape:",X.shape)
    print("Y.shape:",Y.shape)

    _X=tf.constant(X)
    _out = Model_interpolation_2D(patch_size=patch_size,dropout=1)(_X)
    print("out.get_shape():",_out.get_shape())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out=sess.run(_out)


    print("out.shape:",out.shape)
    draw16imgs(out[:, :, :, 0])

    #get_keras_model(32, dropout= 0.2)









import tensorflow as tf
#from ne_ne2.KERAS_LIKE import Conv2D,MaxPooling2D,Concatenate,UpSampling2D
from keras.layers import Input, Concatenate,MaxPooling2D,Conv2D, UpSampling2D, Dropout, Cropping2D, ConvLSTM2D, RepeatVector, Lambda, TimeDistributed

import numpy as np
np.set_printoptions(linewidth=5000)


class ENCODER:

    def __init__(self,nbBlocks:int,filtersBegin):
        self.nbBlocks=nbBlocks
        self.filtersBegin=filtersBegin


    def __call__(self,X):


        with tf.name_scope("Encoder"):

            filters=self.filtersBegin
            Y=X
            self.out=[]

            for i in range(self.nbBlocks):

                with tf.name_scope("Block_go"+str(i)):

                    Y = Conv2D(filters, 3, activation='relu', padding='same', name='conv1')(Y)
                    print(Y.name, "\t\t\t", Y.shape)

                    Y = Conv2D(filters, 3, activation='relu', padding='same', name='conv2')(Y)
                    print(Y.name, "\t\t\t", Y.shape)

                    self.out.append(Y)

                    Y = MaxPooling2D(pool_size=(2, 2), name='pool')(Y)
                    print(Y.name, "\t\t\t", Y.shape)

                    filters*=2


            filters=128
            with tf.name_scope("Encoded1"):
                Y = Conv2D(filters, 1, activation='relu', padding='same', name='conv1')(Y)
                print(Y.name,"\t\t\t" ,Y.shape)
                Y = Conv2D(filters, 1, activation='relu', padding='same', name='conv2')(Y)
                print(Y.name,"\t\t\t" ,Y.shape)

            filters = 128
            with tf.name_scope("Encoded2"):
                Y = Conv2D(filters, 1, activation='relu', padding='same', name='conv1')(Y)
                print(Y.name,"\t\t\t" ,Y.shape)
                Y = Conv2D(filters, 1, activation='relu', padding='same', name='conv2')(Y)
                print(Y.name,"\t\t\t" ,Y.shape)


        return Y



class U_NET:


    def __init__(self,nbBlocks:int,filtersBegin:int,nbCategories:int):
        self.nbBlocks=nbBlocks
        self.filtersBegin=filtersBegin
        self.nbCategories=nbCategories

    def __call__(self,X:tf.Tensor):

        X_shape=X.get_shape().as_list()


        assert (X_shape[1]%(2**self.nbBlocks)==0) and X_shape[1]%(2**self.nbBlocks)==0,"img width and height must be divisible by 2^nbBlocks="+str(2**self.nbBlocks)


        self.encoder=ENCODER(self.nbBlocks,self.filtersBegin)
        Y=self.encoder(X)

        filters=self.filtersBegin*2**(self.nbBlocks-1)

        with tf.name_scope("Decoder"):

            for i in range(self.nbBlocks-1,-1,-1):

                with tf.name_scope("Block_back"+str(i)):
                    Y = Conv2D(filters, 2, activation='relu', padding='same', name='upconv')(UpSampling2D(size=(2, 2))(Y))
                    print(Y.name, "\t\t\t", Y.shape)
                    """ on peut aussi faire :  self.encoder.out[i]+ Y  """
                    Y =  Concatenate(axis=3, name='merge')([self.encoder.out[i], Y])
                    Y = Conv2D(filters, 3, activation='relu', padding='same', name='conv1')(Y)
                    print(Y.name, "\t\t\t", Y.shape)
                    Y = Conv2D(filters, 3, activation='relu', padding='same', name='conv2')(Y)
                    print(Y.name, "\t\t\t", Y.shape)

                    filters//=2


            with tf.name_scope("Decoded"):
                Y = Conv2D(self.nbCategories, 3, activation='relu', padding='same', name='conv1')(Y)
                print(Y.name, "\t\t\t", Y.shape)
                Y = Conv2D(self.nbCategories, 3, activation='relu', padding='same', name='conv2')(Y)
                print(Y.name, "\t\t\t", Y.shape)

        return Y





#
# def simple_taste():
#
#     X=np.ones([1,64,64,1],dtype=np.float32)
#     X=tf.constant(X)
#     encoder=U_NET(2, 10,4)(X)
#
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         res=sess.run(encoder.res)
#         print(res[0,:10,:10,0])


def place_holder_taste():


    X=np.ones([1,64,64,1],dtype=np.float32)
    _X=tf.placeholder(dtype=tf.float32,shape=[None,64,64,1])

    Y=U_NET( 2, 10,3)(_X)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res=sess.run(Y,feed_dict={_X:X})
        print(res[0,:10,:10,0])




if __name__=="__main__":

    place_holder_taste()





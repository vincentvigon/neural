import tensorflow as tf

from ne_ne2.KERAS_LIKE import Conv2D, MaxPooling2D


class LeNet_bottom:

    def __init__(self):
        pass


    def __call__(self,X:tf.Tensor)->tf.Tensor:

        Y=X
        filters=32

        with tf.name_scope("Block1"):
            Y = Conv2D(filters, 5, activation='relu', padding='same', name='conv1')(Y)
            print(Y.name, "\t\t\t", Y.shape)

            Y = MaxPooling2D(pool_size=(2, 2), name='pool')(Y)
            print(Y.name, "\t\t\t", Y.shape)


        filters = 64

        with tf.name_scope("Block2"):
            Y = Conv2D(filters, 5, activation='relu', padding='same', name='conv1')(Y)
            print(Y.name, "\t\t\t", Y.shape)


            Y = MaxPooling2D(pool_size=(2, 2), name='pool')(Y)
            print(Y.name, "\t\t\t", Y.shape)



        self.res=Y

        return self.res



import matplotlib

from ne_ne2.KERAS_LIKE import Conv2D, MaxPooling2D, UpSampling2D

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv

from ne_ne2.LOSSES import crossEntropy, crossEntropy_multiLabel
#from ne_ne2.bricks.U_NET import U_NET
from ne_ne2.VISU import colorize

import numpy as np
np.set_printoptions(linewidth=3000,precision=6,suppress=True)
import tensorflow as tf



class Structure_G:

    def __init__(self,nbCategories:int):
        self.nbCategories=nbCategories


    def __call__(self,X:tf.Tensor):
        """"""

        Y=X
        filters1=32
        with tf.name_scope("Block1"):
            """ 28,28 """
            Y = Conv2D(filters1, 5, activation='relu',padding="same", name='conv')(Y)
            print(Y.name, "\t", Y.shape)
            """ 14,14 """
            Y = MaxPooling2D(pool_size=(2, 2), name='pool')(Y)
            print(Y.name, "\t", Y.shape)
            out1=Y


        filters2 = 64
        with tf.name_scope("Block2"):
            """ 14,14"""
            Y = Conv2D(filters2, 5, activation='relu',padding="same", name='conv')(Y)
            print(Y.name, "\t", Y.shape)
            """ 7,7 """
            Y = MaxPooling2D(pool_size=(2, 2), name='pool')(Y)
            print(Y.name, "\t", Y.shape)



        """ DANS LA SUITE: on recopie leNet, mais en remplaçant les fully-connected par des convolutions 1*1  """

        filters3=128
        """ on élargit avec des conv 1*1"""
        with tf.variable_scope("smallConv0"):
            """7,7"""
            Y = Conv2D(filters3,1, activation='relu',padding="same",name="conv")(Y)
            print(Y.name, "\t", Y.shape)


        filters4=128
        with tf.variable_scope("smallConv0"):
            """7,7"""
            Y = Conv2D(filters4, 1, activation='relu',padding="same", name="conv")(Y)
            print(Y.name, "\t", Y.shape)



        filters5=64
        """  on dilate, on raccorde """
        with tf.variable_scope("dilate0"):
            """ 14 , 14, attention, le nombre de filtre doit être compatible pour le raccordement """
            Y = Conv2D(filters1,2,activation='relu',padding="same")(UpSampling2D(size=(2,2))(Y))
            print(Y.name, "\t", Y.shape)
            """  raccordement """
            Y = Y+out1
            Y = Conv2D(filters5,5,activation='relu',padding="same")(Y)
            print(Y.name, "\t", Y.shape)


        """ on ramène   """
        with tf.variable_scope("dilate1"):
            """ 28,28 """
            Y = Conv2D(filters1,2,activation='relu',padding="same")(UpSampling2D(size=(2,2))(Y))
            print(Y.name, "\t", Y.shape)
            Y = Conv2D(self.nbCategories,5,activation='relu',padding="same")(Y)
            print(Y.name, "\t", Y.shape)




        return Y





class Model_G:


    def __init__(self,h_img:int,w_img:int,nbChannels:int,nbCategories,nbBlocks,filtersBegin):

        self.nbConsecutiveOptForOneFit=1
        self.summaryEither_cat_proba=0

        (self.batch_size,self.h_img, self.w_img, self.nbChannels)=(None,h_img,w_img,nbChannels)

        self.nbCategories=nbCategories

        """ PLACEHOLDER """
        self._X = tf.placeholder(name="X", dtype=tf.float32,shape=(None,h_img,w_img,nbChannels))

        """les annotations : une image d'entier, chaque entier correspond à une catégorie"""
        self._Y_cat = tf.placeholder(dtype=tf.float32, shape=[None, h_img, w_img,nbCategories], name="Y_cat" )


        self._Y_background = tf.placeholder(dtype=tf.float32, shape=[None, h_img, w_img,2], name="Y_background" )

        self._itr = tf.placeholder(name="itr", dtype=tf.int32)

        self.keep_proba=tf.get_variable("keep_proba",initializer=1.,trainable=False)
        self.learning_rate=tf.get_variable("learning_rate",initializer=1e-2,trainable=False)


        resUNet=Structure_G(nbCategories)(self._X) #U_NET(nbBlocks=nbBlocks,filtersBegin=filtersBegin,nbCategories=nbCategories)(self._X)
        self._hat_Y_cat=tf.nn.sigmoid(resUNet)

        resUNet_b = Structure_G(2)(self._X)
        self._hat_Y_background=tf.nn.softmax(resUNet_b,dim=3)

        self._loss_cat =  crossEntropy_multiLabel(self._Y_cat,self._hat_Y_cat)
        self._loss_background=crossEntropy(self._Y_background,self._hat_Y_background)
        self._penalty=tf.constant(1.)

        self._loss=self._loss_cat#+self._loss_background
        tf.summary.scalar("loss", self._loss)


        """ optimizer, monitoring des gradients """
        adam_opt = tf.train.AdagradOptimizer(self.learning_rate)
        _grads_vars = adam_opt.compute_gradients(self._loss)


        # for index, grad in enumerate(_grads_vars):
        #     tf.summary.histogram("{}-grad".format(_grads_vars[index][0].name), _grads_vars[index][0])
        #     tf.summary.histogram("{}-var".format(_grads_vars[index][1].name), _grads_vars[index][1])
        #     if len(_grads_vars[index][0].get_shape().as_list())==4:
        #         ing.summarizeW_asImage(_grads_vars[index][0])


        """ la minimisation est faite via cette op:  """
        self.step_op = adam_opt.apply_gradients(_grads_vars)


        self.rien=tf.ones(1)



        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.verbose=True


        max_outputs=4
        tf.summary.image("input_image", self._X, max_outputs=max_outputs)


        self._Y_cat_sum=tf.reduce_sum(self._Y_cat,axis=3)
        self._hat_Y_cat_sum=tf.reduce_sum(self._hat_Y_cat,axis=3)


        tf.summary.image("Y cat sum", colorize(self._Y_cat_sum, vmin=0.0, vmax=self.nbCategories, cmap='plasma'),max_outputs=max_outputs)
        tf.summary.image("hat Y cat sum", colorize(self._hat_Y_cat_sum, vmin=0.0, vmax=self.nbCategories, cmap='plasma'),max_outputs=max_outputs)
        #tf.summary.image("hat Y background ", colorize(tf.argmax(self._hat_Y_background,axis=3), vmin=0.0, vmax=2, cmap='plasma'),max_outputs=max_outputs)



        self._summary=tf.summary.merge_all()




    def __fit_or_validate(self, X, Y_cat,Y_background, itr:int, fit: bool):


        assert X.shape[0] == Y_cat.shape[0], "x and y must have the same number of lines"

        feed_dict = {self._X: X, self._Y_cat: Y_cat,self._Y_background:Y_background,self._itr:itr}

        if fit:
            step_opt=self.step_op
        else:
            step_opt=self.rien


        if not fit:
            print("_"*20)
            nbStep=1
        else:
            nbStep=self.nbConsecutiveOptForOneFit

        for i in range (nbStep):
            [self.summary,
             _,
             self.loss,
             self.loss_cat,
             self.loss_background,
             self.penalty
             ]=self.sess.run([
                               self._summary,
                               step_opt,
                               self._loss,
                               self._loss_cat,
                               self._loss_background,
                               self._penalty],
                                feed_dict=feed_dict)
            if self.verbose:
                print("loss: %f, loss cat: %f, loss background : %f, penalty: %f" % (self.loss, self.loss_cat,self.loss_background, self.penalty))

        if not fit: print("_"*20)



    def fit(self, X_train, Y_cat_train,Y_background,itr:int):
        if self.verbose: print("fit")
        self.__fit_or_validate(X_train, Y_cat_train,Y_background, itr ,True)


    def validate(self, X_valid, Y_cat_valid,Y_background,itr):
        if self.verbose: print("validate")
        self.__fit_or_validate(X_valid, Y_cat_valid,Y_background,itr, False)


    def predict(self, X_test):
        return self.sess.run([self._hat_Y_cat,self._hat_Y_cat_sum], feed_dict={self._X: X_test})


    def close(self):
        self.sess.close()
        tf.reset_default_graph()






def testBasic():


    nbCat=2
    model = Model_G(28, 28, 1, nbCat,nbBlocks=2,filtersBegin=10)
    model.verbose = True

    X_batch=np.ones(shape=[1,28,28,1])
    Y_batch=np.random.random(size=[1,28,28,nbCat])
    model.fit(X_batch, Y_batch,Y_batch,1)


if __name__=="__main__":
    testBasic()

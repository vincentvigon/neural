
import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv

from ne_ne2.LOSSES import crossEntropy_multiLabel
from ne_ne2.bricks.U_NET import U_NET
from ne_ne2.VISU import colorize

import numpy as np
np.set_printoptions(linewidth=3000,precision=6,suppress=True)
import tensorflow as tf



class Model_H:


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

        self.hat=U_NET(self._X,nbBlocks=nbBlocks,filtersBegin=filtersBegin,nbCategories=nbCategories)

        self._hat_Y_cat=tf.sigmoid(self.hat.res)

        self._loss_cat =  crossEntropy_multiLabel(self._Y_cat,self._hat_Y_cat)
        self._loss_background=tf.constant(1.)
        self._penalty=tf.constant(1.)

        self._loss=self._loss_cat
        tf.summary.scalar("loss", self._loss)



        """ optimizer, monitoring des gradients """
        adam_opt = tf.train.AdamOptimizer(self.learning_rate)
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


        tf.summary.image("Y hat", colorize(self._Y_cat_sum, vmin=0.0, vmax=self.nbCategories, cmap='plasma'),max_outputs=max_outputs)
        tf.summary.image("Y hat", colorize(self._hat_Y_cat_sum, vmin=0.0, vmax=self.nbCategories, cmap='plasma'),max_outputs=max_outputs)



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
        Y_cat,Y_cat_sum = self.sess.run([self._hat_Y_cat,self._hat_Y_cat_sum], feed_dict={self._X: X_test})
        return Y_cat,Y_cat_sum


    def close(self):
        self.sess.close()
        tf.reset_default_graph()






def testBasic():


    nbCat=2
    model = Model_H(50, 50, 1, nbCat,nbBlocks=2,filtersBegin=10)
    model.verbose = True

    X_batch=np.ones(shape=[1,28,28,1])
    Y_batch=np.random.random(size=[1,28,28,nbCat])
    model.fit(X_batch, Y_batch,Y_batch,1)


if __name__=="__main__":
    testBasic()

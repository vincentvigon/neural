
import matplotlib


#TODO: décommentez cette ligne et commenter la suivante pour passer en vrai KERAS.
# sur ce test cela fonctionne vraiment mieux en KERAS_LiKE, Youpi
#from keras.layers import  Concatenate,MaxPooling2D,Conv2D, UpSampling2D
from ne_ne2.KERAS_LIKE import Concatenate,Conv2D, MaxPooling2D, UpSampling2D



from ne_ne2.LOSSES import crossEntropy

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import numpy as np
np.set_printoptions(linewidth=3000,precision=2,suppress=True)
import tensorflow as tf



class Structure_C:

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




class Model_C:


    def __init__(self,h_img:int,w_img:int,nbChannels:int,nbCategories,favoritism):

        self.verbose=True



        (self.batch_size,self.h_img, self.w_img, self.nbChannels)=(None,h_img,w_img,nbChannels)

        self.nbCategories=nbCategories

        self._X = tf.placeholder(name="X", dtype=tf.float32,shape=(None,h_img,w_img,nbChannels))

        """les annotations : une image d'entier, chaque entier correspond à une catégorie"""
        self._Y = tf.placeholder(dtype=tf.int32, shape=[None, h_img, w_img], name="Y")


        self.learning_rate=tf.get_variable("learning_rate",initializer=1e-3,trainable=False)

        res=Structure_C(nbCategories)(self._X)

        self._hat_Y_proba=tf.nn.softmax(res)

        if favoritism is None: favoritism=1

        self._hat_Y=tf.cast(tf.argmax(self._hat_Y_proba*favoritism,axis=3),dtype=tf.int32)


        self._loss =crossEntropy(tf.one_hot(self._Y, self.nbCategories), self._hat_Y_proba)

        self._accuracy=tf.reduce_mean(tf.cast(tf.equal(self._Y, self._hat_Y), tf.float32))
        """la cat 0 est la plus présente (c'est le fond de l'image). 
        le classement trivial revient à classer tous les pixels en 0"""
        self._accuracy_trivial=tf.reduce_mean(tf.cast(tf.equal(0, self._hat_Y), tf.float32))


        """ optimizer, monitoring des gradients """
        self._optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self._grads_vars = self._optimizer.compute_gradients(self._loss)
        for index, grad in enumerate(self._grads_vars):
            tf.summary.histogram("{}-grad".format(self._grads_vars[index][1].name), self._grads_vars[index][0])
            tf.summary.histogram("{}-var".format(self._grads_vars[index][1].name), self._grads_vars[index][1])


        """ la minimisation est faite via cette op:  """
        self._minimizer = self._optimizer.apply_gradients(self._grads_vars)



        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())



        tf.summary.scalar("loss",self._loss)
        tf.summary.scalar("accuracy",self._accuracy)



        self._summary=tf.summary.merge_all()



    def __fit_or_validate(self, X, Y_cat, fit: bool):

        X = self.preprocessingX(X)
        Y_cat = self.preprocessingY(Y_cat)


        assert X.shape[0] == Y_cat.shape[0], "x and y must have the same number of lines"


        """en mode 'validation' pas d'optimisation"""
        trainProcess = self._minimizer if fit else tf.constant(0.)

        feed_dict = {self._X: X, self._Y: Y_cat}
        _, self.loss, self.accuracy,self.accuracy_trivial,self.summary=\
            self.sess.run([trainProcess, self._loss, self._accuracy,self._accuracy_trivial,self._summary],feed_dict=feed_dict)

        if self.verbose:
            print("loss:", self.loss)
            print("accuracy:", self.accuracy)
            print("accuracy_trivial:", self.accuracy_trivial)



    def fit(self, X_train, Y_cat_train):
        if self.verbose: print("fit")
        self.__fit_or_validate(X_train, Y_cat_train, True)

    def validate(self, X_valid, Y_cat_valid):
        if self.verbose: print("validate")
        self.__fit_or_validate(X_valid, Y_cat_valid, False)

    def predict(self, X_test):
        X_test = self.preprocessingX(X_test)

        return self.sess.run([self._hat_Y,self._hat_Y_proba], feed_dict={self._X: X_test})


    def preprocessingX(self, X):
        if not isinstance(X, np.ndarray): raise ValueError("x must be numpy array")
        if len(X.shape)!=4: raise ValueError("x must be a tensor of order 4 which is (batch_size,img_width,img_height,nb_channel)")
        if X[0, :, :, :].shape!=(self.h_img, self.w_img, self.nbChannels): raise ValueError("each img must be of shape (img_width,img_height,nb_channel)")
        return X.astype(np.float32)


    def preprocessingY(self, Y_cat):
        if not isinstance(Y_cat, np.ndarray): raise ValueError("y must be numpy array")
        if len(Y_cat.shape)!=3: raise ValueError("y must be matrix with shape: (batch-size,h_img,w_img)")
        if Y_cat[0,:,:].shape!=(self.h_img, self.w_img):
            raise ValueError("y must be matrix with shape: (batch-size,h_img,w_img)")
        return Y_cat.astype(np.int32)


    def close(self):
        self.sess.close()
        tf.reset_default_graph()



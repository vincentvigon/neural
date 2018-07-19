
import matplotlib
from ne_ne.TasteExample.E_instances.sobel_filter import sobel_penalty

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import numpy as np
np.set_printoptions(linewidth=3000,precision=6,suppress=True)
import tensorflow as tf
import ne_ne.INGREDIENTS as ing


class Encoder:

    def __init__(self,X:tf.Tensor,nbChannels:int):

        self.nbChannels=nbChannels
        nbSummaryOutput=4

        """"""
        ''' couche de convolution 1'''
        with tf.variable_scope("conv1"):
            W_conv1 = ing.weight_variable([5, 5, self.nbChannels, 32],name="W")
            b_conv1 = ing.bias_variable([32],name="b")

            self.filtred1=tf.nn.relu(ing.conv2d_basic(X,W_conv1,b_conv1))
            """ shape=(?,14*14,nbChannels)  """
            self.pool1 =ing.max_pool_2x2(self.filtred1)

            ing.summarizeW_asImage(W_conv1)
            tf.summary.image("filtred", self.filtred1[:, :, :, 0:1], max_outputs=nbSummaryOutput)




        ''' couche de convolution 2'''
        with tf.variable_scope("conv2"):

            W_conv2 = ing.weight_variable([5, 5, 32, 64],name="W")
            b_conv2 = ing.bias_variable([64],name="b")

            self.filtred2=tf.nn.relu(ing.conv2d_basic(self.pool1, W_conv2, b_conv2))
            """ shape=(?,7*7,nbChannels)  """
            self.pool2 =ing.max_pool_2x2(self.filtred2)

            ing.summarizeW_asImage(W_conv2)
            tf.summary.image("filtred",self.filtred2[:,:,:,0:1],max_outputs=12)




        """un alias pour la sortie"""
        self.Y=self.pool2






""" dans cette classe: nous construisons notre sortie supposée.  """
class Decoder:

    def __init__(self, encoder:Encoder, Y_last_dim:int, keep_prob:float, favoritism:tuple, depth0:int, depth1:int):
        """"""


        """ on transforme le layer d'avant en un volume 7*7*depth0 par des conv 1*1"""
        with tf.variable_scope("smallConv0"):
            W = ing.weight_variable([1, 1, 64, depth0], name="W")
            b= ing.bias_variable([depth0], name="b")

            conv = tf.nn.conv2d(encoder.Y, W, strides=[1, 1, 1, 1], padding="SAME") + b
            relu = tf.nn.relu(conv, name="relu")
            relu_dropout = tf.nn.dropout(relu, keep_prob=keep_prob,name="dropout")


        """ on transforme le layer d'avant en un volume 7*7*nbCategories par des conv 1*1"""
        with tf.variable_scope("smallConv1"):
            W = ing.weight_variable([1, 1, depth0,depth1], name="W")
            b = ing.bias_variable([depth1], name="b")

            conv = tf.nn.conv2d(relu_dropout, W, strides=[1, 1, 1, 1], padding="SAME") + b
            relu = tf.nn.relu(conv, name="relu")
            relu_dropout = tf.nn.dropout(relu, keep_prob=keep_prob, name="dropout")



        """  DANS LA SUITE : on dilate les images 7*7 pour revenir à la résolution initiale 28*28  """


        """ 7*7*depth1 ---> 14*14*32 """
        with tf.variable_scope("dilate0"):

            """  [height, width, output_channels, in_channels=nbCategories] """
            W = tf.Variable(initial_value=ing.get_bilinear_initial_tensor([4, 4, 32, depth1],2),name='W')
            b = ing.bias_variable([32], name="b")
            upConv0 = ing.up_convolution(relu_dropout, W, 2, 2) + b
            """on y ajoute le milieu de leNet (14*14*32 aussi)"""
            fuse_1 = upConv0 + encoder.pool1

            ing.summarizeW_asImage(W)



        """on dilate maintenant fuse_1 pour atteindre la résolution des images d'origine
           14*14*32 ----> 28*28*nbCategories
        """
        with tf.variable_scope("dilate1"):
            W = tf.Variable(initial_value=ing.get_bilinear_initial_tensor([4, 4, Y_last_dim, 32], 2), name='W')
            b = ing.bias_variable([Y_last_dim], name="b")

            ing.summarizeW_asImage(W)

        """ les logits (on y applique pas le softmax car plus loin on peut éventuellement utiliser tf.nn.sparse_softmax_cross_entropy_with_logits) """
        self.Y_logits = ing.up_convolution(fuse_1,W,2,2) +b





        self.Y_proba= tf.nn.softmax(self.Y_logits)

        if favoritism is None: favoritism=1
        """ chaque pixel reçoit la catégorie qui a la plus forte probabilité, en tenant compte du favoritisme."""
        self.Y_cat = tf.cast(tf.argmax(self.Y_proba*favoritism, dimension=3, name="prediction"),tf.int32)





class Model_fullyConv_regCat:


    def __init__(self,h_img:int,w_img:int,nbChannels:int,nbCategories,nbRegressor,favoritism,depth0,depth1):

        self.nbConsecutiveOptForOneFit=1
        self.summaryEither_cat_proba=0
        self.nbRegressor=nbRegressor

        (self.batch_size,self.h_img, self.w_img, self.nbChannels)=(None,h_img,w_img,nbChannels)

        self.nbCategories=nbCategories

        """ PLACEHOLDER """
        self._X = tf.placeholder(name="X", dtype=tf.float32,shape=(None,h_img,w_img,nbChannels))
        """les annotations : une image d'entier, chaque entier correspond à une catégorie"""
        self._Y_cat = tf.placeholder(dtype=tf.int32, shape=[None, h_img, w_img], name="Y_cat" )
        self._Y_reg = tf.placeholder(dtype=tf.float32, shape=[None, h_img, w_img,nbRegressor], name="Y_cat" )

        self._itr = tf.placeholder(name="itr", dtype=tf.float32)


        self.keep_proba=tf.get_variable("keep_proba",initializer=1.,trainable=False)
        self.learning_rate=tf.get_variable("learning_rate",initializer=1e-2,trainable=False)



        """la sorties est un volume 7*7*64.  """
        encoder = Encoder(self._X, nbChannels)

        self.hat=Decoder(encoder,  nbCategories, self.keep_proba, favoritism, depth0, depth1)
        self.hat_reg=Decoder(encoder, self.nbRegressor, self.keep_proba, favoritism, depth0, depth1)



        """ les loss qu'on suivra sur le long terme. Les coef, c'est juste pour avoir des grandeurs faciles à lire  """
        self.where=tf.cast((self._Y_cat!=0),dtype=tf.float32)
        self._loss_reg = 0.1 * tf.reduce_mean(self.where*(self._Y_reg-self.hat_reg.Y_logits)**2)

        self._loss_cat =tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.hat.Y_logits, labels=self._Y_cat)))

        self._penalty=10*sobel_penalty(self.hat.Y_proba,self.nbCategories)

        """ si le coef devant la _loss_background est trop grand, la loss_instance reste bloquée à 0.
            mais s'il est trop petit le background se transforme en damier !"""


        self._loss=self._loss_cat + self._loss_reg


        tf.summary.scalar("log loss", tf.log(self._loss))
        tf.summary.scalar("log loss reg", tf.log(self._loss_reg))
        tf.summary.scalar("log loss cat", tf.log(self._loss_cat))
        tf.summary.scalar("log penalty", tf.log(self._penalty))


        """ optimizer, monitoring des gradients """
        adam_opt = tf.train.AdamOptimizer(self.learning_rate)
        _grads_vars = adam_opt.compute_gradients(self._loss)
        # for index, grad in enumerate(_grads_vars):
        #     tf.summary.histogram("{}-grad".format(_grads_vars[index][0].name), _grads_vars[index][0])
        #     tf.summary.histogram("{}-var".format(_grads_vars[index][1].name), _grads_vars[index][1])
        #     if len(_grads_vars[index][0].get_shape().as_list())==4:
        #         ing.summarizeW_asImage(_grads_vars[index][0])


        self._summary = tf.summary.merge_all()

        """ la minimisation est faite via cette op:  """
        self.step_op = adam_opt.apply_gradients(_grads_vars)




        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.verbose=True


        max_outputs=8
        tf.summary.image("input_image", self._X, max_outputs=max_outputs)


        if self.summaryEither_cat_proba==0:
            output = tf.expand_dims(tf.cast(self.hat.Y_cat,dtype=tf.float32),3)
            output_color = ing.colorize(output, vmin=0.0, vmax=self.nbCategories, cmap='plasma') #'viridis', 'plasma', 'inferno', 'magma'
            tf.summary.image("Y_hat",output_color)


            for i in range(self.nbRegressor):
                output2 = tf.expand_dims(tf.cast(self.hat_reg.Y_logits[:,:,:,i], dtype=tf.float32), 3)
                """   maxNbStrate depend of the size of cells. """
                output_color2 = ing.colorize(output2, vmin=None, vmax=None,cmap='plasma')  # 'viridis', 'plasma', 'inferno', 'magma'
                tf.summary.image("Y_reg_hat"+str(i), output_color2)

        else :
            for cat in range(0,self.nbCategories):
                tf.summary.image("hat_proba cat"+str(cat), tf.expand_dims(self.hat.Y_proba[:,:,:,cat],3), max_outputs=max_outputs)


        self._summary=tf.summary.merge_all()




    def __fit_or_validate(self, X, Y_cat,Y_reg, itr:int, fit: bool):


        assert X.shape[0] == Y_cat.shape[0]==Y_reg.shape[0], "x and y must have the same number of lines"

        feed_dict = {self._X: X, self._Y_cat: Y_cat,self._Y_reg:Y_reg,self._itr:itr}

        if fit:
            step_opt=self.step_op
        else:
            step_opt=tf.constant(1.)


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
             self.loss_reg,
             self.penalty
             ]=self.sess.run([
                               self._summary,
                               step_opt,
                               self._loss,
                               self._loss_cat,
                               self._loss_reg,
                               self._penalty],
                                feed_dict=feed_dict)
            if self.verbose:
                print("loss: %f, loss cat: %f, loss_reg: %f, penalty: %f" % (self.loss, self.loss_cat,self.loss_reg, self.penalty))

        if not fit: print("_"*20)



    def fit(self, X_train, Y_cat_train,Y_reg_train,itr:int):
        if self.verbose: print("fit")
        self.__fit_or_validate(X_train, Y_cat_train,Y_reg_train, itr ,True)

    def validate(self, X_valid, Y_cat_valid,Y_reg_valid,itr):
        if self.verbose: print("validate")
        self.__fit_or_validate(X_valid, Y_cat_valid,Y_reg_valid,itr, False)


    def predict(self, X_test):
        #X_test = self.preprocessingX(X_test)

        Y_cat,Y_reg = self.sess.run([self.hat.Y_cat,self.hat_reg.Y_logits], feed_dict={self._X: X_test})
        return Y_cat,Y_reg

    #
    # def preprocessingX(self, X):
    #     if not isinstance(X, np.ndarray): raise ValueError("x must be numpy array")
    #     if len(X.shape)!=4: raise ValueError("x must be a tensor of order 4 which is (batch_size,img_width,img_height,nb_channel)")
    #     if X[0, :, :, :].shape!=(self.h_img, self.w_img, self.nbChannels): raise ValueError("each img must be of shape (img_width,img_height,nb_channel)")
    #     return X.astype(np.float32)
    #
    #
    # def preprocessingY(self, Y_proba):
    #     if not isinstance(Y_proba, np.ndarray): raise ValueError("y must be numpy array")
    #     if len(Y_proba.shape)!=4: raise ValueError("y must be matrix with shape: (batch-size,h_img,w_img,nb_cat)")
    #     if Y_proba[0, :, :,:].shape!=(self.h_img, self.w_img,self.nbCategories) :
    #         raise ValueError("y must be matrix with shape: (batch-size,h_img,w_img,nb categories)")
    #     return Y_proba.astype(np.float32)


    def close(self):
        self.sess.close()
        tf.reset_default_graph()


#
# def testBasic():
#
#
#     nbCat=2
#     model = Model_fullyConv_reg(28, 28, 1, nbCat,favoritism=None,depth0=128,depth1=64)
#     model.verbose = True
#
#     X_batch=np.ones(shape=[1,28,28,1])
#     Y_batch=np.random.random(size=[1,28,28,2])
#     model.fit(X_batch, Y_batch,0)
#
#
# if __name__=="__main__":
#     testBasic()

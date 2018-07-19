
import matplotlib

from ne_ne.dataGen.dataContour.data_getter import get_both_data_withGT, get_one_data_withGT

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import numpy as np
np.set_printoptions(linewidth=3000,precision=2,suppress=True)
import matplotlib.pyplot as plt
import tensorflow as tf
import ne_ne.BRICKS as bricks
import ne_ne.INGREDIENTS as ing



""" dans cette classe: nous construisons notre sortie supposée.  """
class Hat_fullyConv:

    def __init__(self,X,nbChannels:int,nbCategories:int,keep_prob,favoritism):
        """"""


        """on récupère le réseau très simple: leNet_bottom"""
        leNet= bricks.LeNet_bottom(X,nbChannels)
        """la sorties est un volume 7*7*64.  """

        """ DANS LA SUITE: on recopie leNet, mais en remplaçant les fully-connected par des convolutions 1*1  """


        """ on transforme le layer d'avant en un volume 7*7*1024 par des conv 1*1"""
        with tf.variable_scope("smallConv0"):
            W = ing.weight_variable([1, 1, 64, 1024], name="W")
            b= ing.bias_variable([1024], name="b")

            conv = tf.nn.conv2d(leNet.Y, W, strides=[1, 1, 1, 1], padding="SAME") + b
            relu = tf.nn.relu(conv, name="relu")
            relu_dropout = tf.nn.dropout(relu, keep_prob=keep_prob,name="dropout")




        """ on transforme le layer d'avant en un volume 7*7*nbCategories par des conv 1*1"""
        with tf.variable_scope("smallConv1"):
            W = ing.weight_variable([1, 1, 1024,nbCategories], name="W")

            b = ing.bias_variable([nbCategories], name="b")

            conv = tf.nn.conv2d(relu_dropout, W, strides=[1, 1, 1, 1], padding="SAME") + b
            relu = tf.nn.relu(conv, name="relu")
            relu_dropout = tf.nn.dropout(relu, keep_prob=keep_prob, name="dropout")



        """  DANS LA SUITE : on dilate les images 7*7 pour revenir à la résolution initiale 28*28  """


        """ 7*7*nbCategories ---> 14*14*32 """
        with tf.variable_scope("dilate0"):

            """  [height, width, output_channels, in_channels=nbCategories] """
            W = tf.Variable(initial_value=ing.get_bilinear_initial_tensor([4, 4, 32, nbCategories],2),name='W')
            b = ing.bias_variable([32], name="b")
            upConv0 = ing.up_convolution(relu_dropout, W, 2, 2) + b
            """on y ajoute le milieu de leNet (14*14*32 aussi)"""
            fuse_1 = upConv0 + leNet.pool1

            ing.summarizeW_asImage(W)




        """on dilate maintenant fuse_1 pour atteindre la résolution des images d'origine
           14*14*32 ----> 28*28*nbCategories
        """
        with tf.variable_scope("dilate1"):
            W = tf.Variable(initial_value=ing.get_bilinear_initial_tensor([4, 4, nbCategories, 32],2),name='W')
            b = ing.bias_variable([nbCategories], name="b")

            ing.summarizeW_asImage(W)



        """ les logits (on y applique pas le softmax car plus loin on utilisera la loss tf.nn.sparse_softmax_cross_entropy_with_logits) """
        self.Y_logits = ing.up_convolution(fuse_1,W,2,2) +b

        self.Y_proba= tf.nn.softmax(self.Y_logits)
        """ chaque pixel reçoit la catégorie qui a la plus forte probabilité, en tenant compte du favoritisme."""
        self.Y_cat = tf.cast(tf.argmax(self.Y_proba*favoritism, dimension=3, name="prediction"),tf.int32)





class Model_fullyConv:


    def __init__(self,h_img:int,w_img:int,nbChannels:int,nbCategories,favoritism):


        (self.batch_size,self.h_img, self.w_img, self.nbChannels)=(None,h_img,w_img,nbChannels)

        self.nbCategories=nbCategories

        self._X = tf.placeholder(name="X", dtype=tf.float32,shape=(None,h_img,w_img,nbChannels))

        """les annotations : une image d'entier, chaque entier correspond à une catégorie"""
        self._Y_cat = tf.placeholder(dtype=tf.int32, shape=[None, h_img, w_img], name="Y", )


        self.keep_proba=tf.get_variable("keep_proba",initializer=1.,trainable=False)
        self.learning_rate=tf.get_variable("learning_rate",initializer=1e-3,trainable=False)

        self.hat=Hat_fullyConv(self._X, nbChannels, nbCategories, self.keep_proba,favoritism)


        """ une version 'déjà' prête de la loss """
        #self._loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.hat.Y_logits,labels=self._Y_cat)))
        """ la version 'à la main'  """
        self._loss =ing.crossEntropy(tf.one_hot(self._Y_cat,self.nbCategories),self.hat.Y_proba)


        self._accuracy=tf.reduce_mean(tf.cast(tf.equal(self._Y_cat, self.hat.Y_cat), tf.float32))
        """la cat 0 est la plus présente (c'est le fond de l'image). 
        le classement trivial revient à classer tous les pixels en 0"""
        self._accuracy_trivial=tf.reduce_mean(tf.cast(tf.equal(0, self.hat.Y_cat), tf.float32))




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

        self.verbose=True


        tf.summary.scalar("loss",self._loss)
        tf.summary.scalar("accuracy",self._accuracy)

        max_outputs=5
        tf.summary.image("input_image", self._X, max_outputs=max_outputs)
        tf.summary.image("ground_truth", tf.expand_dims(tf.cast(self._Y_cat,tf.float32),3), max_outputs=max_outputs)
        for cat in range(1,self.nbCategories):
            tf.summary.image("hat_proba cat 1", tf.expand_dims(self.hat.Y_proba[:,:,:,cat],3), max_outputs=max_outputs)


        self._summary=tf.summary.merge_all()



    def __fit_or_validate(self, X, Y_cat, fit: bool):

        X = self.preprocessingX(X)
        Y_cat = self.preprocessingY(Y_cat)


        assert X.shape[0] == Y_cat.shape[0], "x and y must have the same number of lines"


        """en mode 'validation' pas d'optimisation"""
        trainProcess = self._minimizer if fit else tf.constant(0.)

        feed_dict = {self._X: X, self._Y_cat: Y_cat}
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

        Y_cat,Y_proba = self.sess.run([self.hat.Y_cat,self.hat.Y_proba], feed_dict={self._X: X_test})
        return Y_cat,Y_proba


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




def taste():

    logDir="/Users/vigon/GoogleDrive/permanent/python/neurones/ne_ne/log"
    if tf.gfile.Exists(logDir):
        tf.gfile.DeleteRecursively(logDir)


    nbImg = 500

    imgChoice=2


    if imgChoice==0:
        """la classe 1 est beaucoup plus présente que la classe 0. On introduit du favoritism pour calculer l'accuracy"""
        favoritism = (1,5)
        "blurCircles -> centre"
        X,Y=get_one_data_withGT(nbImg, "blurCircles")
        nbCat=2
    elif imgChoice==1:
        favoritism = (1,5)
        "rectangle -> contour"
        X, Y = get_one_data_withGT(nbImg, "rectangles")
        nbCat=2
    elif imgChoice==2:
        favoritism = (1,5)
        "les deux, sans dire qui est qui"
        X, Y = get_both_data_withGT(nbImg, False)
        nbCat=2
    else :
        "les deux,en disant qui est qui: bizarre, cela n'améliore pas ! "
        favoritism = (1,5,5)
        X, Y = get_both_data_withGT(nbImg, True)
        nbCat=3


    model = Model_fullyConv(28, 28, 1, nbCat,favoritism)
    model.verbose = True
    summary_writer_train = tf.summary.FileWriter(logDir, model.sess.graph)



    nbTrain = int(len(X) * 0.8)
    X_train, Y_train = X[:nbTrain, :, :], Y[:nbTrain, :]
    X_val, Y_val = X[nbTrain:, :, :], Y[nbTrain:, :]


    batchSize=50
    nbStep=2000

    accTrain=np.zeros(nbStep)
    lossTrain=np.zeros(nbStep)

    accValid = np.zeros(nbStep)
    lossValid = np.zeros(nbStep)
    accValid[:]=np.nan
    lossValid[:]=np.nan


    try:
        for step in range(nbStep):
            print("step:", step)

            shuffle = np.random.permutation(len(X_train))
            shuffle = shuffle[:batchSize]
            X_batch = X_train[shuffle]
            Y_batch = Y_train[shuffle]

            model.fit(X_batch, Y_batch)

            accTrain[step] = model.accuracy
            lossTrain[step] = model.loss
            summary_writer_train.add_summary(model.summary,global_step=step)


            if step%20==0:
                print("\nVALIDATION")
                model.validate(X_val, Y_val)
                accValid[step] = model.accuracy
                lossValid[step] = model.loss

                print("\n")


    except  KeyboardInterrupt:
        print("on a stoppé")


    plt.subplot(1, 2, 1)
    plt.plot(accTrain, label="train")
    plt.plot(accValid, '.', label="valide class")
    plt.title("accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(lossTrain, label="train")
    plt.plot(lossValid, '.',label="valid class")

    plt.title("loss")
    plt.legend()


    """ TEST"""
    X_test,Y_test = X_val[:16],Y_val[:16]
    hat_Y_test_cat,hat_Y_test_proba=model.predict(X_test)
    X_test=X_test.reshape([16,28,28])



    def drawOne(imgs,title,vmax):

        plt.figure().suptitle(title)
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(imgs[i],vmin=0,vmax=vmax)

        plt.subplots_adjust(bottom=0.1,top=0.9, left=0.1, right=0.8)
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(cax=cax)


    drawOne(X_test,"X",1)
    drawOne(Y_test,"Y",2)

    drawOne(hat_Y_test_proba[:, :, :,1],"proba cat 1",1)
    if nbCat==3:drawOne(hat_Y_test_proba[:, :, :,2],"proba cat 2",1)


    plt.show()

    model.close()



if __name__=="__main__":

    taste()


import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import numpy as np
np.set_printoptions(linewidth=3000,precision=2,suppress=True)
import matplotlib.pyplot as plt
import tensorflow as tf
import ne_ne.BRICKS as bricks
import ne_ne.INGREDIENTS as ing
import ne_ne.dataGen.data.data_getter as data_getter


""" décrit la transformation de X en Y, paramétrée par les weight et bias """
class Hat_bounding:

    def __init__(self,X,_keep_proba,nbChannels:int,nbCategories:int,nbRegressor:int):

        leNet_bottom= bricks.LeNet_bottom(X,nbChannels)


        """on récupère la sortie de leNet_bottom"""
        h_pool2=leNet_bottom.pool2


        W_fc1 = ing.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = ing.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, _keep_proba)


        ''' on connecte les 1024 neurons avec nbCategories neurones de sorties'''
        W_fc2 = ing.weight_variable([1024, nbCategories])
        b_fc2 = ing.bias_variable([nbCategories])


        self.Y_prob=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        self.Y_cat=tf.arg_max(self.Y_prob, dimension=1)

        ''' on connecte les 1024 neurons avec nbRegressor neurones de sorties'''
        W_fc_reg = ing.weight_variable([1024, nbRegressor])
        b_fc_reg = ing.bias_variable([nbRegressor])

        tf.summary.histogram("b_fc2", b_fc2)
        tf.summary.histogram("b_fc1", b_fc1)
        tf.summary.histogram("W_fc2",W_fc2)
        tf.summary.histogram("W_fc1",W_fc1)

        self.Y_bound= tf.matmul(h_fc1_drop, W_fc_reg) + b_fc_reg




class Model_bounding:


    def __init__(self,nbChannels:int,nbCategories:int,nbRegressor:int):

        h_img, w_img=28,28


        self.inputSize=(h_img,w_img,nbChannels)
        self.nbRegressor=nbRegressor
        self.nbCategories=nbCategories


        self._X = tf.placeholder(name="X", dtype=tf.float32,shape=(None,h_img,w_img,nbChannels))

        tf.summary.image("X",self._X,max_outputs=12)

        self._Y_class = tf.placeholder(name="Y_class", dtype=tf.float32,shape=(None,nbCategories))
        self._Y_reg = tf.placeholder(name="Y_reg", dtype=tf.float32,shape=(None,nbRegressor))


        self.learning_rate=tf.get_variable("learning_rate",initializer=1e-2,trainable=False)
        self.keep_proba=tf.get_variable("keep_proba",initializer=1.,trainable=False)


        self.hat=Hat_bounding(self._X,self.keep_proba,nbChannels,nbCategories,nbRegressor)

        #self._Y_class_hat, self._Y_reg_hat = xToY.getYclass_Yreg(self._X, self.keep_proba)


        self._loss_class = 20.*ing.crossEntropy(self._Y_class, self.hat.Y_prob,True)
        self._loss_reg=  tf.reduce_mean((self._Y_reg - self.hat.Y_bound) ** 2)


        self._accuracy=tf.reduce_mean(tf.cast(tf.equal(self.hat.Y_cat,tf.argmax(self._Y_class,dimension=1)),dtype=tf.float32))

        self._minimizer = tf.train.AdamOptimizer(1e-4).minimize(self._loss_class+self._loss_reg)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.verbose=True

        tf.summary.scalar("loss_class",self._loss_class)
        tf.summary.scalar("loss_reg",self._loss_reg)
        tf.summary.scalar("accuracy",self._accuracy)

        self._summary=tf.summary.merge_all()



    def _fit_or_validate(self, X, Y_prob, Y_bounds, fit: bool):

        X = self.preprocessingX(X)
        Y_prob = self.preprocessingY_class(Y_prob)
        Y_bounds=self.preprocessingY_reg(Y_bounds)

        assert X.shape[0] == Y_prob.shape[0] == Y_bounds.shape[0], "x and y must have the same number of lines"


        """en mode 'validation' pas d'optimisation"""
        trainProcess = self._minimizer if fit else tf.constant(0.)

        feed_dict = {self._X: X, self._Y_class: Y_prob, self._Y_reg: Y_bounds}
        _, self.loss_class,self.loss_reg, self.accuracy,self.summary=self.sess.run([trainProcess, self._loss_class, self._loss_reg, self._accuracy,self._summary],feed_dict=feed_dict)

        if self.verbose:
            print("loss class:", self.loss_class)
            print("loss reg:", self.loss_reg)
            print("accuracy:", self.accuracy)


    def fit(self, X_train, Y_train_prob, Y_train_bounds):
        if self.verbose: print("fit")
        self._fit_or_validate(X_train, Y_train_prob, Y_train_bounds, True)

    def validate(self, X_valid, Y_valid_prob, Y_valid_bounds):
        if self.verbose: print("validate")
        self._fit_or_validate(X_valid, Y_valid_prob, Y_valid_bounds, False)

    def predict(self, X_test):
        X_test = self.preprocessingX(X_test)
        Y_prob_hat,Y_cat_hat,Y_bounds_hat = self.sess.run([self.hat.Y_prob, self.hat.Y_cat, self.hat.Y_bound], feed_dict={self._X: X_test})
        return Y_prob_hat,Y_cat_hat,Y_bounds_hat




    def preprocessingX(self, X):
        if not isinstance(X, np.ndarray): raise ValueError("x must be numpy array")
        if len(X.shape)!=4: raise ValueError("x must be a tensor of order 4 which is (batch_size,img_width,img_height,nb_channel)")
        if X[0, :, :, :].shape!=self.inputSize: raise ValueError("each img must be of shape (img_width,img_height,nb_channel)=" + str(self.inputSize))

        return X.astype(np.float32)

    def preprocessingY_class(self, Y):
        if not isinstance(Y, np.ndarray): raise ValueError("y must be numpy array")
        if len(Y.shape)!=2: raise ValueError("y must be matrix with shape: (batch-size,nb_categories)")
        if Y.shape[1]!=self.nbCategories : raise ValueError("the number of columns of y must be nb_categories=" + str(self.nbCategories))
        return Y.astype(np.float32)

    def preprocessingY_reg(self, Y):

        if self.nbRegressor==0: raise ValueError("nbRegressor is 0 but a regression is asked")

        if not isinstance(Y, np.ndarray): raise ValueError("y must be numpy array")
        if len(Y.shape)!=2: raise ValueError("y must be matrix with shape: (batch-size,nb_regressor)")
        if Y.shape[1]!=self.nbRegressor : raise ValueError("the number of columns of y must be nb_regressor=" + str(self.nbRegressor))
        return Y.astype(np.float32)




    def close(self):
        self.sess.close()
        tf.reset_default_graph()




def taste():
    """
    Le modèle doit repérer des ronds et des carrés, et leur bounding box.
    Volontairement, les boundings box des ronds et des sont paramétrés de manière différente:
     - pour les rond on indique : centre/rayon
     - pour les rectangles on indique : coin gauche,haut et la longueur du côté.
      quand l'ordi se trompe de catégorie, il ne se trompe aussi dans la bounding box...
    """

    logDir="/Users/vigon/GoogleDrive/permanent/python/neurones/ne_ne/TasteExample/log"
    if tf.gfile.Exists(logDir):
        tf.gfile.DeleteRecursively(logDir)

    model = Model_bounding(1,2,3)
    model.verbose = True
    summary_writer_train = tf.summary.FileWriter(logDir+"/train",model.sess.graph)
    summary_writer_test = tf.summary.FileWriter(logDir+"/test",model.sess.graph)


    nbImg = 500

    X, Y, bounding = data_getter.get_data_circlesAndSquares(nbImg)

    shuffle = np.random.permutation(len(X))
    X = X[shuffle]
    Y = Y[shuffle]
    bounding=bounding[shuffle]

    nbTrain = int(len(X) * 0.8)
    X_train, Y_train,bounding_train = X[:nbTrain, :, :], Y[:nbTrain, :],bounding[:nbTrain,:]
    X_val, Y_val , bounding_val = X[nbTrain:, :, :], Y[nbTrain:, :],bounding[nbTrain:,:]


    batchSize=50

    nbStep=400

    accTrain=np.zeros(nbStep)
    lossClassTrain=np.zeros(nbStep)
    lossRegTrain=np.zeros(nbStep)

    accValid = np.zeros(nbStep)
    accValid[:]=np.nan
    lossClassValid = np.zeros(nbStep)
    lossClassValid[:]=np.nan
    lossRegValid = np.zeros(nbStep)
    lossRegValid[:]=np.nan




    try:
        for step in range(nbStep):
            print("step:", step)

            shuffle = np.random.permutation(len(X_train))
            shuffle = shuffle[:batchSize]
            X_batch = X_train[shuffle, :, :]
            Y_batch = Y_train[shuffle, :]
            Y_reg_batch = bounding_train[shuffle, :]

            model.fit(np.expand_dims(X_batch, 3), Y_batch, Y_reg_batch)

            accTrain[step] = model.accuracy
            lossClassTrain[step] = model.loss_class
            lossRegTrain[step] = model.loss_reg
            summary_writer_train.add_summary(model.summary,global_step=step)


            if step%20==0:
                print("\nVALIDATION")
                model.validate(np.expand_dims(X_val, 3), Y_val,bounding_val)
                accValid[step] = model.accuracy
                lossClassValid[step] = model.loss_class
                lossRegValid[step] = model.loss_reg
                summary_writer_test.add_summary(model.summary, global_step=step)

                print("\n")


    except  KeyboardInterrupt:
        print("on a stoppé")


    plt.subplot(1, 2, 1)
    plt.plot(accTrain, label="train")
    plt.plot(accValid, '.', label="valide class")
    plt.title("accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(lossClassTrain, label="train")
    plt.plot(lossClassValid, '.',label="valid class")
    plt.plot(lossRegValid, '.',label="valid reg")

    plt.title("loss")
    plt.legend()



    _,Y_cat_hat,bounding_hat=model.predict(np.expand_dims(X_val, 3))
    model.close()

    plt.figure()

    for i in range(16):
        plt.subplot(4,4,i+1)

        if Y_cat_hat[i]==0:
            data_getter.showSquare(X_val[i,:,:],*bounding_hat[i])
        else:
            data_getter.showCircle(X_val[i,:,:],*bounding_hat[i])
    plt.show()



if __name__=="__main__":

    taste()

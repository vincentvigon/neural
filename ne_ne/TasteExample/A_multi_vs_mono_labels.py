
import matplotlib

from ne_ne.dataGen.data.data_getter import get_data_superposed, get_data_circlesAndSquares

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import numpy as np
np.set_printoptions(linewidth=3000,precision=2,suppress=True)
import matplotlib.pyplot as plt
import tensorflow as tf
import ne_ne.BRICKS as bricks
import ne_ne.INGREDIENTS as ing




class Hat_multi_vs_mono:

    def __init__(self,X,multi_label:bool,nbChannels:int,nbCategories:int,_keep_proba):


        self.leNet_bottom= bricks.LeNet_bottom(X,nbChannels)


        """"""
        """on récupère la sortie de leNet_bottom"""
        h_pool2=self.leNet_bottom.pool2

        '''
          On connecte les 7*7*64 neurones à une nouvelle couche de 1024 neurons
          fc signifie : fully connected.
          '''
        W_fc1 = ing.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = ing.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        """dropout pour éviter le sur-apprentissage"""
        h_fc1_drop = tf.nn.dropout(h_fc1, _keep_proba)


        ''' on connecte les 1024 neurons avec nbCategories neurones de sorties'''
        W_fc2 = ing.weight_variable([1024, nbCategories])
        b_fc2 = ing.bias_variable([nbCategories])


        if multi_label:
            ''' sigmoid produit un vecteur dont chaque composante est une proba'''
            self.Y= tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        else:
            '''softmax produit un vecteur de probabilité (la somme des composantes fait 1)'''
            self.Y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


class Model_LeNet:


    def __init__(self,multi_label,nbChannels,nbCategories):

        self._X = tf.placeholder(name="X", dtype=tf.float32)
        self._Y = tf.placeholder(name="Y", dtype=tf.float32)


        self.learning_rate=tf.get_variable("learning_rate",initializer=1e-4,trainable=False)
        self.keep_proba=tf.get_variable("keep_proba",initializer=1.,trainable=False)
        self.threshold=tf.get_variable("threshold",initializer=0.5,trainable=False)


        self.hat=Hat_multi_vs_mono(self._X,multi_label,nbChannels,nbCategories,self.keep_proba)


        """la loss et l'accuracy sont calculée de manière très différentes quand c'est du multi-label"""
        if multi_label:
            self._loss=ing.crossEntropy_multiLabel(self._Y,self.hat.Y)
            Y_hat_binary=tf.cast(tf.greater(self.hat.Y, self.threshold), tf.float32) #ing.make_binary_with_threshold(self._Y_hat,self.threshold)
            """ l'accuracy dépend d'un threshold. Attention, une accuracy proche de 1 n'est pas forcément bonne: 
              Ex:  Y_hat_binary = [ 0,0,0,0,0,0,0,0,0,1]
                    Y_hat       = [ 1,0,0,0,0,0,0,0,0,0]
                    ---> accuracy = 80 % 
                    alors que le modèle n'a rien compris.
              """
            self._accuracy=tf.reduce_mean(tf.cast(tf.equal(Y_hat_binary, self._Y), tf.float32))
        else:
            self._loss=ing.crossEntropy(self._Y,self.hat.Y)
            Y_cat=tf.argmax(self._Y,dimension=1)
            Y_cat_hat=tf.argmax(self.hat.Y,dimension=1)
            self._accuracy=tf.reduce_mean(tf.cast(tf.equal(Y_cat,Y_cat_hat),tf.float32))


        self._minimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self._loss)


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.verbose=True


    def preprocessingX(self, X):
        return X.astype(np.float32)

    def preprocessingY(self, Y):
        return Y.astype(np.float32)


    def __fit_or_validate(self, X, Y, fit: bool):

        X = self.preprocessingX(X)
        Y = self.preprocessingY(Y)

        if X.shape[0] != Y.shape[0]: raise ValueError("x and y must have the same number of lines")


        """en mode 'validation' il ne faut surtout pas faire d'optimisation"""
        trainProcess = self._minimizer if fit else tf.constant(0.)

        _,self.loss,self.accuracy=self.sess.run([trainProcess, self._loss,self._accuracy], {self._X: X, self._Y: Y})

        if self.verbose:
            print("loss:", self.loss)
            print("accuracy:", self.accuracy)



    def fit(self, X_train, Y_train):
        if self.verbose: print("fit")
        self.__fit_or_validate(X_train, Y_train, True)

    def validate(self, X_valid, Y_valid):
        if self.verbose: print("validate")
        self.__fit_or_validate(X_valid, Y_valid, False)

    def predict(self, X_test):
        X_test = self.preprocessingX(X_test)

        Y_hat = self.sess.run([self.hat.Y], feed_dict={self._X: X_test})
        return Y_hat

    def close(self):
        self.sess.close()
        tf.reset_default_graph()



def taste(multi_label:bool):

    model = Model_LeNet(multi_label,1,2)
    model.verbose = True

    """ nbImgPerType doit être <= MAX_NB_IMG=500.  """
    nbImgPerType = 500

    if multi_label:
        X,Y=get_data_superposed(nbImgPerType)
    else:
        X, Y, _ = get_data_circlesAndSquares(nbImgPerType)

    shuffle = np.random.permutation(len(X))
    X = X[shuffle]
    Y = Y[shuffle]

    nbTrain = int(len(X) * 0.9)
    X_train, Y_train = X[:nbTrain, :, :], Y[:nbTrain, :]
    X_val, Y_val = X[nbTrain:, :, :], Y[nbTrain:, :]

    nbStep = 400
    accTrain = np.zeros(nbStep)
    accValid = np.zeros(nbStep)
    accValid[:]=np.nan
    lossTrain = np.zeros(nbStep)
    lossValid = np.zeros(nbStep)
    lossValid[:]=np.nan
    batchSize = 50

    try:
        for step in range(nbStep):
            print("step:", step)

            shuffle = np.random.permutation(len(X_train))
            shuffle = shuffle[:batchSize]
            X_batch = X_train[shuffle, :, :]
            Y_batch = Y_train[shuffle, :]

            model.fit(np.expand_dims(X_batch, 3), Y_batch)
            accTrain[step] = model.accuracy
            lossTrain[step] = model.loss

            if np.isnan(model.loss) :
                print("train loss is Nan")
                raise Exception()

            if step%20==0:
                print("\nVALIDATION")
                model.validate(np.expand_dims(X_val, 3), Y_val)
                accValid[step] = model.accuracy
                lossValid[step] = model.loss
                print("\n")


    except  KeyboardInterrupt:
        print("on a stoppé")

    plt.subplot(1, 2, 1)
    plt.plot(accTrain, label="train")
    plt.plot(accValid, '.',label="valide")
    plt.title("accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(lossTrain, label="train")
    plt.plot(lossValid,'.' ,label="valid")
    plt.title("loss")
    plt.legend()
    plt.show()

    model.close()


if __name__=="__main__":

    taste(True)

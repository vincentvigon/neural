
import tensorflow as tf

from ne_ne2.KERAS_LIKE import Dense
from ne_ne2.LOSSES import crossEntropy_multiLabel, crossEntropy
from ne_ne2.bricks.Le_Net import LeNet_bottom





class Hat_A:

    def __init__(self,multi_label:bool,nbCategories:int):
        self.multi_label=multi_label
        self.nbCategories=nbCategories


    def __call__(self,X:tf.Tensor)->tf.Tensor:

        """"""
        """on récupère la sortie de leNet_bottom"""
        Y=LeNet_bottom()(X)

        #Y_shape=Y.get_shape().as_list()

        '''
          On connecte les 7*7*64 neurones à une nouvelle couche de 1024 neurons
          fc signifie : fully connected.
          '''

        with tf.name_scope("full"):
            #Y=tf.reshape(Y, [-1, Y_shape[1]*Y_shape[2]*Y_shape[3]])

            Y=Dense(1024)(Y)
            print(Y.name,"\t",Y.shape)


            ''' on connecte les 1024 neurons avec nbCategories neurones de sorties'''
            Y=Dense(self.nbCategories)(Y)
            print(Y.name,"\t",Y.shape)



        if self.multi_label:
            ''' sigmoid produit un vecteur dont chaque composante est une proba'''
            return tf.nn.sigmoid(Y)
        else:
            '''softmax produit un vecteur de probabilité (la somme des composantes fait 1)'''
            return tf.nn.softmax(Y)





class Model_A:


    def __init__(self,multi_label:bool,img_h:int,img_w:int,img_chanels:int,nbCategories:int):

        self._X = tf.placeholder(name="X", dtype=tf.float32,shape=[None,img_h,img_w,img_chanels])
        self._Y = tf.placeholder(name="Y", dtype=tf.float32,shape=[None,nbCategories])


        self.learning_rate=tf.get_variable("learning_rate",initializer=1e-4,trainable=False)

        self.keep_proba=tf.get_variable("keep_proba",initializer=1.,trainable=False)
        self.threshold=tf.get_variable("threshold",initializer=0.5,trainable=False)
        self.verbose=True


        self._hat_Y=Hat_A(multi_label, nbCategories)(self._X)


        """la loss et l'accuracy sont calculée de manière très différentes quand c'est du multi-label"""
        if multi_label:
            self._loss=crossEntropy_multiLabel(self._Y, self._hat_Y)
            Y_hat_binary=tf.cast(tf.greater(self._hat_Y, self.threshold), tf.float32) #ing.make_binary_with_threshold(self._Y_hat,self.threshold)
            """ l'accuracy dépend d'un threshold. Attention, une accuracy proche de 1 n'est pas forcément bonne: 
              Ex:  Y_hat_binary = [ 0,0,0,0,0,0,0,0,0,1]
                    Y_hat       = [ 1,0,0,0,0,0,0,0,0,0]
                    ---> accuracy = 80 % 
                    alors que le modèle n'a rien compris.
              """
            self._accuracy=tf.reduce_mean(tf.cast(tf.equal(Y_hat_binary, self._Y), tf.float32))
        else:
            self._loss=crossEntropy(self._Y, self._hat_Y)
            Y_argmax=tf.argmax(self._Y,dimension=1)
            hat_Y_argmax=tf.argmax(self._hat_Y, dimension=1)
            self._accuracy=tf.reduce_mean(tf.cast(tf.equal(Y_argmax,hat_Y_argmax),tf.float32))





        """ optimizer, monitoring des gradients """
        adam_opt = tf.train.AdamOptimizer(self.learning_rate)
        _grads_vars = adam_opt.compute_gradients(self._loss)
        for index, grad in enumerate(_grads_vars):
            tf.summary.histogram("{}-grad".format(_grads_vars[index][0].name), _grads_vars[index][0])
            tf.summary.histogram("{}-var".format(_grads_vars[index][1].name), _grads_vars[index][1])
            # if len(_grads_vars[index][0].get_shape().as_list())==4:
            #     ing.summarizeW_asImage(_grads_vars[index][0])



        """ la minimisation est faite via cette op:  """
        self._step_op = adam_opt.apply_gradients(_grads_vars)


        tf.summary.scalar("loss",self._loss)
        tf.summary.scalar("accuracy",self._accuracy)

        self._summary = tf.summary.merge_all()





        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


        self._nothing=tf.constant(0.)




    def __fit_or_validate(self, X, Y, fit: bool):



        if X.shape[0] != Y.shape[0]: raise ValueError("x and y must have the same number of lines")


        """en mode 'validation' il ne faut surtout pas faire d'optimisation"""
        trainProcess = self._step_op if fit else self._nothing

        self.summary,_,self.loss,self.accuracy=self.sess.run([self._summary,trainProcess, self._loss,self._accuracy], {self._X: X, self._Y: Y})

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

        return  self.sess.run(self._hat_Y, feed_dict={self._X: X_test})


    def close(self):
        self.sess.close()
        tf.reset_default_graph()

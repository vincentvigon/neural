import matplotlib

matplotlib.use('TkAgg')  # truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

logDir = "/Users/vigon/GoogleDrive/permanent/python/neurones/ne_ne/log"


class Model_ridge:


    def __init__(self,savedir:str=None):

        """les 2 placeholder correspondant à l'entrée et à la sortie"""
        self._X = tf.placeholder(name="X", dtype=tf.float32)
        self._Y = tf.placeholder(name="Y", dtype=tf.float32)


        """ 2 variables non-trainable """
        self.learning_rate=tf.get_variable("learning_rate",initializer=1e-4,trainable=False)
        """pénalisation L2"""
        self.wei_decay=tf.get_variable("wei_decay",initializer=0.1,trainable=False)


        """ 2 variables trainable """
        self.a=tf.get_variable("a",initializer=2.)
        self.b=tf.get_variable("b",initializer=0.)



        """  3 INGREDIENTS PRINCIPAUX :
          1/ Y_hat : la sortie estimée
          2/ loss  : la quantité à minimisée
          3/ minimizer : l'algo de minimisation
           """
        self._Y_hat = self.a*self._X+self.b

        self._loss = tf.reduce_mean((self._Y - self._Y_hat) ** 2) + self.wei_decay * self.a ** 2
        summary_entropy=tf.summary.scalar("loss", self._loss)
        summary_a=tf.summary.scalar("a", self.a)
        summary_b=tf.summary.scalar("b", self.b)

        self._summary_op = tf.summary.merge([summary_entropy,summary_a,summary_b]) #ou bien tf.summary.merge_all()


        """ 
        Nous précisons l'algo de minimisation et la quantité à minimiser. Sela peux se faire en une ligne:
                self._minimizer =  tf.train.AdamOptimizer(self.learning_rate).minimize(self._loss)
        Mais nous le faisons en deux étapes pour pouvoir observer les gradients. 
        
        ATTENTION :  certain optimizer (tel Adam)  ne veulent pas que "global_variables_initializer" soient lancée 
        AVANT que l'optimizer soit entré dans le graph tf. """

        """ l'algo """
        self._opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        """ les gradients: une liste de paires [Gradient, Variable]. On récupère le nom des variables. """
        self._grads_vars = self._opt.compute_gradients(self._loss)
        self.varName=[grad[1].name for grad in self._grads_vars]
        # for index, grad in enumerate(grads):
        #     tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])

        """ la minimisation est faite via cette op:  """
        self._minimizer = self._opt.apply_gradients(self._grads_vars)




        """pour sauver les variables du modèle"""
        self._saver = tf.train.Saver()
        """la session tensorflow"""
        self.sess = tf.Session()


        """l'initialization des variables se fait une seule fois.
                 En particulier, un second a appel de self.fit() améliore le travail du précédent
                  en modifiant les variables trainable """
        if savedir is None: self.sess.run(tf.global_variables_initializer())
        else : self._saver.restore(self.sess, savedir)

        self.verbose=True



    def saveAndClose(self,savedir:str):
        save_path = self._saver.save(self.sess, savedir)
        print("Model saved in file: %s" % save_path)
        self.close()



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


        _, self.loss, self.grads_vars, self.summary_op=\
            self.sess.run([trainProcess, self._loss, self._grads_vars, self._summary_op], {self._X: X, self._Y: Y})


        if self.verbose:
            print("loss:", self.loss)
            for name,grad_var in zip(self.varName,self.grads_vars):
                print("variable ", name, "\tgrad:",grad_var[0], "\tvalue (after grad-subtraction): ",grad_var[1])


    def fit(self, X_train, Y_train):
        if self.verbose: print("fit")
        self.__fit_or_validate(X_train, Y_train, True)

    def validate(self, X_valid, Y_valid):
        if self.verbose: print("validate")
        self.__fit_or_validate(X_valid, Y_valid, False)

    def predict(self, X_test):
        X_test = self.preprocessingX(X_test)
        Y_hat = self.sess.run([self._Y_hat], feed_dict={self._X: X_test})
        return Y_hat

    def close(self):
        self.sess.close()
        tf.reset_default_graph()







def unitaryTaste():
    """
    On  entraîne (=fit) le modèle ridge, en changeant les paramètres.
    On observe la courbe de loss.

    Pour observer les différents summary sur le tensorboard:
    ouvrez une console. Taper tensorboard --logdir VOTRE_LOG_DIR
    puis mettez l'url proposer dans le navigateur.
    Attention : à chaque relance du programme python,
    - il faut vider le logdir (c'est fait automatiquement ci-dessous)
    et aussi relancer tensorboard (control+C).
    - ou bien vous pouvez mettre des nouveaux sous-répertoires dans le logdir, cela fait
    des tracés superposés: Pratique pour comparer des résultats.
    """


    """génération des données"""
    nbData = 1000
    dataX = np.linspace(0, 1, nbData)
    dataY = 2 * dataX + 1 + np.random.normal(loc=0, scale=0.1, size=nbData)

    losses = []
    tf.reset_default_graph()

    if tf.gfile.Exists(logDir): tf.gfile.DeleteRecursively(logDir)


    model = Model_ridge()
    summary_writer = tf.summary.FileWriter(logDir,model.sess.graph)

    for itr in range(60):
        print("itr:",itr)

        if itr==20:
            model.sess.run(model.learning_rate.assign(0.1))
            print("changing leaning rate to:",model.sess.run(model.learning_rate))
        elif itr==40:
            model.sess.run(model.wei_decay.assign(0))
            print("changing wei_decay to:",model.sess.run(model.wei_decay))

        model.fit(dataX, dataY)
        losses.append(model.loss)
        summary_writer.add_summary(model.summary_op, itr)

    model.saveAndClose("/tmp/model.ckpt")



    model=Model_ridge("/tmp/model.ckpt")


    for itr in range(60,100):

        model.fit(dataX, dataY)
        losses.append(model.loss)
        summary_writer.add_summary(model.summary_op, itr)



    plt.plot(losses)
    plt.annotate("lr = 1e-4 ", xy=(0, 0))
    plt.annotate("lr = 0.1 ", xy=(20, 0))
    plt.annotate("wd = 0. ", xy=(40, 0))
    plt.annotate("restor", xy=(60, 0))
    plt.plot([0, 0], [0, 2], "--")
    plt.plot([20, 20], [0, 2], "--")
    plt.plot([40, 40], [0, 2], "--")
    plt.plot([60, 60], [0, 2], "--")

    plt.show()


    model.close()





if __name__ == "__main__":
    unitaryTaste()



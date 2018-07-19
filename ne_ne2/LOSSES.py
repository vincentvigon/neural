import tensorflow as tf



""" ici :https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
ils indique que faire un clip sur Y_hay n'est pas une bonne idée car cela arrète la propagation du gradient... 
J'ai tester  :  un clip  "grossier"  ex : tf.clip_by_value(Y_hat, 0.0001, 0.9999) : cela tue vraiment l'apprentissage.
Un clip plus fin ne change pas grand chose, mais il faudrait faire des tests plus poussé.
   """
def crossEntropy(Y, Y_hat):
    return - tf.reduce_mean(Y*tf.log(Y_hat+1e-10))



def quadraticLoss(Y, Y_hat):
    return tf.reduce_mean((Y-Y_hat)**2)


"""
Attention ici au fameux  1+1e-10 -Y avec Y=1  ---> 0  et quand on passe au log ---> nan
"""
def crossEntropy_multiLabel(Y,Y_hat):
    return - tf.reduce_mean(  Y * tf.log(Y_hat+1e-10) + (1 - Y) * tf.log( (1 - Y_hat)+1e-10))



def accuracy(true_Y_proba, hat_Y_proba):
    return tf.reduce_mean(tf.cast(tf.equal(hat_Y_proba, true_Y_proba), tf.float32))

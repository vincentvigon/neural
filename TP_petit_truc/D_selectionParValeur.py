import tensorflow as tf
import numpy as np
np.random.seed(1234)




sess = tf.InteractiveSession()


def step0():


    x = tf.constant([1, 2, -3, -4])
    A=tf.where(x>0)
    B=tf.reshape(A,shape=[-1])
    print(B.eval())




    x=tf.where(x>0,x,tf.zeros_like(x))
    print(x.eval())


    x=tf.constant([1.,2,3,4])
    y=tf.constant([2.1,2.2,2.3,2.4])
    z=tf.where(x<=y,x,y)

    print(z.eval())

step0()

"""  exo écrire l'équivalent numpy  """



"""

https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
Merci mathguyjohn

imaginons qu'on veuille implémenter la fonction suivante

f(x) = { 1/x, x!=0
       { 0,   x=0

Une solution naive serait 

    tf.where(x_ok, f(x), safe_f(x))
mais cela ne marche pas. Pourquoi ?

La bonne solution est :

safe_x = tf.where(x_ok, x, safe_x)
tf.where(x_ok, f(safe_x), safe_f(x))


"""





def step1():
    def f(x):
        x_ok = tf.not_equal(x, 0.)
        f = lambda x: 1. / x
        safe_f = tf.zeros_like
        return tf.where(x_ok, f(x), safe_f(x))

    def safe_f(x):
        x_ok = tf.not_equal(x, 0.)
        f = lambda x: 1. / x
        safe_f = tf.zeros_like
        safe_x = tf.where(x_ok, x, tf.ones_like(x))
        return tf.where(x_ok, f(safe_x), safe_f(x))


    x = tf.constant([-1., 0, 1])
    print(tf.gradients(f(x), x)[0].eval())
    # ==> array([ -1.,  nan,  -1.], dtype=float32)
    #  ...bah! We have a NaN at the asymptote despite not having
    # an asymptote in the non-differentiated result.


    x = tf.constant([-1., 0, 1])
    print(tf.gradients(safe_f(x), x)[0].eval())
    # ==> array([-1.,  0., -1.], dtype=float32)
    # ...yay! double-where trick worked. Notice that the gradient
    # is now a constant at the asymptote (as opposed to being NaN)




""" on voit qu'en certain points, il ont choisi un sous-gradient. """
def step2():

    x = tf.constant([-2., 0, 2])

    print(tf.gradients(tf.abs(x), x)[0].eval())

step2()







def step100():

    sess=tf.InteractiveSession()

    A=np.random.randint(0,10,size=[5,5])
    _A=tf.constant(A)
    print(_A.eval())

    _mask=tf.less_equal(_A,3)
    print(_mask.eval())

    values_gt_3=tf.boolean_mask(_A,_mask)

    print(values_gt_3.eval())

    indices_gt_3=tf.where(_mask)

    print(indices_gt_3.eval())


    _B= tf.Variable(initial_value=np.zeros([5,2]))


    sess.run(tf.global_variables_initializer())


    print(_B.eval())


    tf.scatter_update(_B, [1,3], [[1.,2.],[1.,2.]]).eval()

    print(_B.eval())



















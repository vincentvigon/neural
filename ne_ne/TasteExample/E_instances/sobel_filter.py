import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(linewidth=10000,suppress=True)



def sobel_filter_test(x,show=False):


    X = np.reshape(x, [1, 28, 28, 1])


    _W_vert=tf.reshape([-1.,0,1,-2,0,2,-1,0,1 ] , [3,3])
    _W_hori=tf.reshape([-1.,-2,-1,0,0,0,1,2,1] , [3,3])

    _W=tf.reshape(tf.stack([_W_hori,_W_vert],axis=2),[3,3,1,2])


    _X_fil = tf.reduce_sum(tf.nn.conv2d(X, _W, strides=[1, 1, 1, 1], padding='SAME'),axis=3)
    _gradient=_X_fil**2/16

    with tf.Session() as sess:
        X_fil,W_vert,W_hori,W,gradient = sess.run([_X_fil,_W_vert,_W_hori,_W,_gradient])
        print(W_vert)
        print(W_hori)
        print(W[:,:,0,1])

        print(X_fil.shape)

    gradient=np.reshape(gradient,[28,28])
    print(gradient)

    if show:
        plt.subplot(1, 2, 1)
        plt.imshow(x)
        plt.subplot(1, 2, 2)
        plt.imshow(gradient)
        plt.show()

    return np.mean(gradient)




def sobel_penalty(X,nbChannel):
    """
    :param X: shape[batch_size,w,h,nbChannel]
    :return: une pénalité grande si les images on beaucoup de contour.
    Cette pénalité est renormalisée par la taille de l'image, la taille du batch et le nombre de channel.
    Ainsi 1 image, ou une image répétée ont la même pénalitée (cf testes)
    """

    _W_vert = tf.reshape([-1., 0, 1, -2, 0, 2, -1, 0, 1], [3, 3])
    _W_hori = tf.reshape([-1., -2, -1, 0, 0, 0, 1, 2, 1], [3, 3])
    _W = tf.reshape(tf.stack([_W_hori, _W_vert], axis=2), [3, 3, 1, 2])

    res=0
    for i in range(nbChannel):
        X_one_ch=tf.expand_dims(X[:,:,:,i],axis=3)# en numpy c'est: X[:,:,:,[i]]
        _X_fil = tf.reduce_sum(tf.nn.conv2d(X_one_ch, _W, strides=[1, 1, 1, 1], padding='SAME'), axis=3)
        _gradient = _X_fil ** 2 / 16.
        res+=tf.reduce_mean(_gradient)

    return res/nbChannel



if __name__=="__main__":

    x = np.zeros([28, 28], dtype=np.float32)
    x[5:10, 10:20] = 1

    res=sobel_filter_test(x,False)
    print("one channel",res)



    X=np.zeros([2,28, 28,3], dtype=np.float32)

    X[:,5:10, 10:20,:]=1
    X=tf.constant(X)
    _Res=sobel_penalty(X,3)

    with tf.Session() as sess:
        Res=sess.run(_Res)
        print("several Chanel:",Res)












import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



"""

conv2d_transpose écarte les pixels d'une image en introduisant stride-pixel de zéro entre chaque pixel.
Puis elle effectue une convolution ordinaire.

La fonction  'conv2d_transpose' est un peu mal fichue, car elle exige de spécifier la taille de sortie 
(alors que cette taille devrait être calculée à partir de l'image d'entrée, du stride et du padding)... 
En tout cas, en prenant out = in*stride cela marche toujours 

(cela marche aussi des tailles proches car tf joue avec le padding, mais je n'ai pas compris exactement quelles
sont les taille autorisée, l'aide étant pour l'instant trop imprécise). 

Donc je conseille de prendre toujours out = in*stride, puis ensuite de réduire l'image l'image (un peu) soi même. 

Pour finir, dernière bizarerie: les dimension du filtre W correpondent à
            [width, height, num_out_channels, num_in_channels]
            
(in et out sont inversés par rapport à un filtre de convolution usuel)
"""
def step0():

    x=np.zeros([28,28],dtype=np.float32)
    x[5:10,10:20]=1

    """essayez avec w_size=1,2,3,..."""
    w_size=1
    W=np.ones([w_size,w_size,1,1],dtype=np.float32)

    X=np.reshape(x,[1,28,28,1])

    """ essayer avec stride = 1,2,3... """
    stride=2 # 1 ou 2
    output=28*stride
    _X_fil=tf.nn.conv2d_transpose(X,W,output_shape=[1,output,output,1],strides=[1,stride,stride,1])

    with tf.Session() as sess:
        X_fil=sess.run(_X_fil)

    print("X.shape:",X.shape)
    print("X_fil.shape:",X_fil.shape)

    x_fil= np.reshape(X_fil,[output,output])

    plt.subplot(1,2,1)
    plt.imshow(x)
    plt.subplot(1,2,2)
    plt.imshow(x_fil)
    plt.show()




""" Autre problème : souvent la dim-0 du tenseur d'entrée n'est pas connue car cela correspond à la taille du batch.
Cette petite fonction règle le problème.
"""
def up_convolution(X, W, nbOutChanel, stride_h, stride_w):

        input_size_h = X.get_shape().as_list()[1]
        input_size_w = X.get_shape().as_list()[2]

        output_size_h=input_size_h*stride_h
        output_size_w=input_size_w*stride_w


        """tf.shape(input)[0] c'est le batch-size, qui n'est déterminée qu'au moment du sess.run. 
          Du coup c'est un tf.shape et pas tf.get_shape """
        output_shape = tf.stack([tf.shape(X)[0],
                                 output_size_h, output_size_w,
                                 nbOutChanel])

        upconv = tf.nn.conv2d_transpose(X, W, output_shape, [1, stride_h, stride_w, 1])

        # Now output.get_shape() is equal (?,?,?,?) which can become a problem in the
        # next layers. This can be repaired by reshaping the tensor to its shape:
        output = tf.reshape(upconv, output_shape)
        # now the shape is back to (?, H, W, C)

        return output






"""
Une bonne (?) manière d'initialiser le filtre de convolution-transposée : 
C'est les noyau classiques que l'on utilise pour agrandir une image
d'après http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/"""
def get_bilinear_initial_tensor(filter_shape, upscale_factor):

    assert filter_shape[0]==filter_shape[1], "only square filters are produced here"

    kernel_size = filter_shape[1]
    """point culminant """
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]],dtype=np.float32)
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            """calcul de l'interpolation"""
            value = (1 - abs((x - centre_location) / upscale_factor)) * (1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value

    """on recopie pour tous les chanel"""
    weights = np.zeros(filter_shape,dtype=np.float32)
    for i in range(filter_shape[2]):
        for j in range(filter_shape[3]):
            """j'ai divisé par nb_in-chanel."""
            weights[:, :, i, j] = bilinear/filter_shape[2]


    return weights




""" observons le filtre bilinéaire """
def step1():
    upscale_factor = 5
    w_size = 7
    W = get_bilinear_initial_tensor([w_size, w_size, 1, 1], upscale_factor)

    w=np.reshape(W,[w_size, w_size])
    plt.imshow(w)
    plt.show()






def step2():

    x=np.zeros([28,28],dtype=np.float32)
    x[5:10,10:20]=1
    X=np.reshape(x,[1,28,28,1])

    upscale_factor=3
    w_size=5

    tip_top=True
    if tip_top:    W=get_bilinear_initial_tensor([w_size,w_size,1,1],upscale_factor)
    else: W=np.ones([w_size,w_size,1,1],dtype=np.float32)


    _X_fil=up_convolution(tf.constant(X),W,1,upscale_factor,upscale_factor)

    with tf.Session() as sess:
        X_fil=sess.run(_X_fil)

    print("X.shape:",X.shape)
    print("X_fil.shape:",X_fil.shape)

    x_fil= np.reshape(X_fil,[28*upscale_factor,28*upscale_factor])

    plt.subplot(1,2,1)
    plt.imshow(x)
    plt.subplot(1,2,2)
    plt.imshow(x_fil)
    plt.show()


step2()






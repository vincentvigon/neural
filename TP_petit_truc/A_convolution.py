import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
"""supprime certain warning pénible de tf"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(linewidth=10000,suppress=True)


"""

On va convoler des tenseurs d'ordre 4.

Un tenseur d'images :

X[b,i,j,q]

b -> l'indice qui fait parcourir le batch (ex: le numéro d'un individu)

i,j -> les indices qui dont parcourir les images (height,width)

q  ->  le numéro du 'chanel' ( ex: couleur)



Un tenseur de noyaux :

W[i,j,q,k]

i,j -> les indices qui décrivent les pixels de chaque noyau (kernel)
q   -> l'indice du in-chanel, qui doit coller au chanel de l'image
l   -> l'indice du out-chanel, qui peut dé-multiplier les convolutions

"""




"""
Pour commencer concentrons nous sur une entrée X constituée d'une seule image qui n'a qu'un seul chanel (image en niveau de gris). 
Et produisons en sortie une seule image avec un seul chanel.

Ainsi
shape X[b,i,j,q]  =  (1,h_img,w_img,1)
shape W[i,j,q,k]  =  (h_ker,w_ker,1,1)

"""

def step0():

    x=np.zeros([28,28],dtype=np.float32)
    x[5:10,10:20]=1
    w=np.ones([5,5],dtype=np.float32)


    X=np.reshape(x,[1,28,28,1])
    W=np.reshape(w,[5,5,1,1])

    """ essayer avec stride = 1 ou 2 """
    stride=2 # 1 ou 2
    _X_fil=tf.nn.conv2d(X,W,strides=[1,stride,stride,1],padding='SAME')

    with tf.Session() as sess:
        X_fil=sess.run(_X_fil)


    x_fil=np.reshape(X_fil,[28//stride,28//stride])

    plt.subplot(1,2,1)
    plt.imshow(x)
    plt.subplot(1,2,2)
    plt.imshow(x_fil)
    plt.show()



""" 
Le stride c'est le pas de déplacement du noyau de convolution. 
Ainsi avec stride=2, le noyau saute un pixel sur deux. 
Le résultat de la convolution est deux fois plus petit. 



Le padding c'est le fait de rajouter des zéro autour de l'image.
L'option padding='SAME' rajoute des zéro autour de l'image de manière à ce que, avec stride=1, l'image de
sortie ai la même taille que l'image d'entrée (avec stride=2 elle est exactement deux fois plus petite). 

L'option padding='VALID', ne rajoute aucun zéro autour, ainsi l'image de sortie est forcément plus petite.



La formule générale pour connaître la dimension de l'output est :

                    output =  (input + padding - filter)/ stride + 1

Cette formule étant bien entendu vraie en hauteur ou en largeur. 


Tensorflow fait le choix de régler le padding automatiquement

 * En mode SAME, dans le cas simple ou input%stride==0 et stride<filter :
                    padding = filter - stride
                donc
                    output = input/stride
                    
 * En mode VALIDE, padding = 0, et il faut arrondir quand la formule générale ne donne pas un entier. 
 
 Pour plus de détail, lire l'aide (bien faite): 
  https://www.tensorflow.org/api_guides/python/nn#convolution


"""





"""
La formule générale est

conv(x,W) [b, i, j, k]     =    sum_{di, dj, q}  X[b,   i + stride[1]*di ,  j + stride[2]*dj, q] *  W[di, dj, q, k]





Simplifions en prenant les stride =1

Y = conv(x,W) [b, i, j, k]     =    sum_{di, dj, q}  X[b,   i + di ,  j + dj, q] *  W[di, dj, q, k]

Observons l'indice : b 
est reste inchangé entre l'entrée X et la sortie Y

Par contre on voit qu'il y a une multiplication matricielle au niveau de l'indices 'q'. 

Par exemple, supposons que X soit constitué d'une seule image RGB, et que W n'ai qu'un seul canal de sortie
shape X[b,i,j,q]  =  (1,h_img,w_img,3)
shape W[i,j,q,k]  =  (h_ker,w_ker,3,1)

Ainsi ce cas, les 3 canaux de X sont convolés respectivement avec les 3 couches de W, puis sont sommés
pour former en sortie une image avec 1 chanel. 


Si maintenant W a  K-canaux de sortie, l'opération précédente est effectuée K-fois, produisant en sortie une image
avec K-canaux

"""


def step1():
    """

    Une image RGB changée une image en niveau de gris, car le W n'a que 1 canal de sortie.
    On verra que dans les réseaux de neurones, W à beaucoup de canaux de sortie : on démultiplie l'image de départ.

    """

    x=np.zeros([28,28,3],dtype=np.float32)
    x[5:10,10:20,0]=1
    x[0:5, 10:20, 1] = 1
    x[5:10, 15:25, 2] = 1

    X=np.reshape(x,[1,28,28,3])
    W=np.ones([5,5,3,1],dtype=np.float32)


    _X_fil=tf.nn.conv2d(X,W,strides=[1,1,1,1],padding='SAME')

    with tf.Session() as sess:
        X_fil=sess.run(_X_fil)


    x_fil=np.reshape(X_fil,[28,28])

    plt.subplot(1,2,1)
    plt.imshow(x)
    plt.subplot(1,2,2)
    plt.imshow(x_fil)
    plt.show()





'''le filtre de Sobel renvoie les contours d'une image'''
def demiSobel():
    x = np.zeros([28, 28], dtype=np.float32)
    x[5:10, 10:20] = 1

    X = np.reshape(x, [1, 28, 28, 1])


    W=tf.reshape([-1.,0,1,-2,0,2,-1,0,1],[3,3,1,1])

    _X_fil = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

    with tf.Session() as sess:
        X_fil = sess.run(_X_fil)


    gradient=np.reshape(X_fil,[28,28])**2

    plt.subplot(1, 2, 1)
    plt.imshow(x)
    plt.subplot(1, 2, 2)
    plt.imshow(gradient)
    plt.show()








''' Nous n'avons fait ici qu'un demi-Sobel.
 Le vrai filtre de Sobel c'est : calculer de gradient selon l'axe des x et l'axe des y, puis
 sommer les carrés de ces deux gradients.

 Modifiez le programme pour que cela marche avec la contrainte suivante : n'utiliser qu'une seule fois
 la convolution de tensorflow. Pour cela, utilisez le fait que W
 peuvent avoir plusieurs canaux de sortie.
 '''


import tensorflow as tf
import numpy as np


""" permuter un tenseur selon un premier indice """
def step2():

    A=tf.constant([[0.,0,0],[2,2,2],[4,4,4],[6,6,6]])
    indices=tf.constant([2,0,0,3,0])


    """ l'équivalent numpy c'est 
            B=A[indices]
        ou encore :
           pour tout i : B[i,:] = A[indices[i],:]    
     """
    B=tf.nn.embedding_lookup(A,indices)

    """ Chaque élément 'i'  de  "indices_multiple" est remplacé par A[i]   """
    indices_multiple = tf.constant([[2,2],[1,3],[0,3]])

    C=tf.nn.embedding_lookup(A,indices_multiple)



    with tf.Session() as sess:
        print(sess.run(B))
        print("_"*20)
        print(sess.run(C))


step2()




def step3():

    mat_class=tf.constant([[1,0],[1,2]])
    eye=tf.eye(3,3)
    """ chaque classe (0,1 ou 2) est remplacér par la 'dirac' correspondant : (1,0,0), (0,1,0) ou (0,0,1) """
    mat_proba=tf.nn.embedding_lookup(eye,mat_class)
    """  avec une formule :
    mat_proba[i,j,:]=eye[ mat_class[i,j] , : ]
    """

    """ il y a une aussi fonction toute  faite pour cela"""
    one_hot=tf.one_hot(mat_class,3)


    with tf.Session() as sess:
        print(sess.run(mat_class))
        print(sess.run(eye))
        print("-"*20)
        print("avec embedding_lookup")
        print(sess.run(mat_proba))
        print("avec one hot")
        print(sess.run(one_hot))


    """ La même chose en numpy:"""
    mat_class = np.array([[1, 0], [1, 2]])
    eye = np.eye(3, 3)
    """ chaque classe (0,1 ou 2) est remplacér par la 'dirac' correspondant : (1,0,0), (0,1,0) ou (0,0,1) """
    mat_proba = eye[mat_class]
    print("avec numpy")
    print(mat_proba)



def step4():

    tf.InteractiveSession()
    indices=tf.constant([2,1])
    A = tf.constant([[0., 0, 0], [2, 2, 2], [4, 4, 4], [6, 6, 6]])

    A_perm=tf.gather(A,indices)
    #B=tf.diag_part(A_perm[:,:2])

    print(A.eval())
    print()
    print(A_perm.eval())

    # print(B.eval())
    # print(tf.gather_nd(A,[[0,0],[2,0]]).eval())
    # print(A[[0,0],[2,0]].eval())














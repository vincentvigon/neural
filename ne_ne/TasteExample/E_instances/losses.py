import tensorflow as tf
#import numpy as np
import matplotlib
import numpy as np

from ne_ne.TasteExample.E_instances.dataDealer import oneBatch_of_rect_instances

np.random.seed(1234132)

from tensorflow.contrib.image.python.ops import image_ops

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt


def norm1_tf(Y):
    return tf.reduce_sum(Y,axis=[0,1])

def norm1_tf_dec(Y):
    return tf.reduce_sum(Y,axis=[1,2])

def norm1_np(a):
    return np.sum(np.sum(a,axis=0),axis=0)





def invariant_IoU(Y, Y_hat, epsilon=1e-6):
    """

        calcul :
           1/n  sum_t=0^n-1  max_s (  f_IoU (Y[t],Y_hat[s]) )

       Avec facteur 1/n : le max est 1

       A priori : Quand les strates de Y sont toutes remplies, pas besoin d'epsilon.
       Mais quand on travaille par batch, les dernière strates de Y sont nulles
    """

    n=Y.get_shape().as_list()[2]

    norm = norm1_tf(Y)
    norm_hat = norm1_tf(Y_hat)


    Y_ext=tf.expand_dims(Y, 3)
    Y_hat_ext=tf.expand_dims(Y_hat, 2)

    """ sca[k,l] = sca( Y[:,:,k] Y_hat[:,:,l])      """
    sca = norm1_tf(Y_ext * Y_hat_ext)

    norm=tf.expand_dims(norm,1)
    norm_hat=tf.expand_dims(norm_hat,0)

    F_IoU =  sca / ( epsilon + norm +norm_hat)


    """ 1/n *   sum_k  max_l F_IoU[k,l] 
      = 1/n *  sum_k  max_l F_IoU(Y[:,:,k],Y_hat[:,:,l])   """
    return  tf.reduce_sum(tf.reduce_max(F_IoU, axis=1))



def invariant_IoU_batch(Y, Y_hat, epsilon=1e-6):

    """ Idem que  invariant_IoU, mais pour des batch d'image.
     L'indeice de batch étant en premier, il faut décaler tous les indices
     """
    n=Y.get_shape().as_list()[3]

    norm_a = norm1_tf_dec(Y)
    norm_b = norm1_tf_dec(Y_hat)


    Ya_ext=tf.expand_dims(Y, 4)
    Yb_ext=tf.expand_dims(Y_hat, 3)

    """ sca[b,k,l] = sca( Y[b,:,:,k] Y_hat[b,:,:,l])      """
    sca = norm1_tf_dec(Ya_ext * Yb_ext)

    norm_a=tf.expand_dims(norm_a,2)
    norm_b=tf.expand_dims(norm_b,1)

    F_IoU =  sca / ( epsilon + norm_a+norm_b-sca)

    """   sum_k  max_l F_IoU[b,k,l] =   sum_k  max_l F_IoU(Ya[b,k],Yb[b,l])   """
    correl_simple=  tf.reduce_sum(tf.reduce_max(F_IoU, axis=2))/n

    return correl_simple




def matching_IoU(Y, Y_hat, epsilon=1e-6):
    """
    :param Y: de shape[img_h,img_w,nb_cat]
    :param Y_hat: idem
    :param epsilon: pour éviter une div par zéro
    :return:
    """


    norm = norm1_tf(Y)

    """  j'ai essayer de supprimer les eventuelles categorie avec que des zeros, mais en vain. 
     (ça serait 2 lignes simple de numpy !!!) """
    # indexPos=tf.reshape(tf.where(norm>0),shape=[-1])
    # Y=tf.nn.embedding_lookup(tf.transpose(Y,perm=[2,0,1]),indexPos)
    # Y=tf.transpose(Y,perm=[1,2,0])
    # norm=tf.nn.embedding_lookup(norm,indexPos)

    n = Y.get_shape().as_list()[2]


    norm_hat = norm1_tf(Y_hat)


    Y_ext=tf.expand_dims(Y, 3)
    Y_hat_ext=tf.expand_dims(Y_hat, 2)

    """ sca[k,l] = sca( Y[:,:,k] Y_hat[:,:,l])      """
    sca = norm1_tf(Y_ext * Y_hat_ext)

    norm=tf.expand_dims(norm,1)
    norm_hat=tf.expand_dims(norm_hat,0)

    F_IoU =  sca / (epsilon+ norm +norm_hat)
    """ Attention,  bipartite_match cherche à minimiser """
    r2c, c2r = image_ops.bipartite_match(-F_IoU, n)

    A_perm = tf.gather(F_IoU, c2r)
    return tf.reduce_sum(tf.diag_part(A_perm))


def just_IoU_batch(Y,Y_hat,epsilon=1e-6):
    """
    :param Y: de shape [batch_size,img_h,img_w]
    :param Y_hat: idem
    :param epsilon: pour éviter une div par zéro
    :return:
    """

    """ sum_i,j Y[b,i,j] """
    norm=tf.reduce_sum(Y,axis=[1,2])
    norm_hat=tf.reduce_sum(Y_hat,axis=[1,2])

    """ sca = sum_i,j  Y[b,i,j] Y_hat[b,i,j]     """
    sca = tf.reduce_sum(Y * Y_hat,axis=[1,2])

    IoU=  sca / (epsilon + norm + norm_hat)
    """ somme sur le batch """
    return tf.reduce_mean(IoU)







def matching_IoU_batch(Ys, Ys_hat, epsilon=1e-6):
    """
    On applique le matching_IoU sur chaque image du batch
    """

    func=lambda Y_Y_hat: matching_IoU(Y_Y_hat[0],Y_Y_hat[1],epsilon)

    Ys_Ys_hat=tf.stack([Ys,Ys_hat],axis=1)

    return tf.reduce_mean(tf.map_fn(func,Ys_Ys_hat))









def invariant_crossEntropy(Y, Y_hat, epsilon=1e-6):
    """
    Comme précédemment, mais on remplace IoU par cross_entropy.
    C'est très lent !
    """
    n = Y.get_shape().as_list()[2]

    Y_ext = tf.expand_dims(Y, 3)
    Y_hat_ext = tf.expand_dims(Y_hat, 2)

    cross =  - Y_ext * tf.log(Y_hat_ext + epsilon) - (1-Y_ext) * tf.log( (1-Y_hat_ext) + epsilon)

    """ 1/n *   sum_k  max_l F_IoU[k,l] 
      = 1/n *  sum_k  max_l F_IoU(Y[:,:,k],Y_hat[:,:,l])   """

    return tf.reduce_sum(tf.reduce_max(cross, axis=1)) / n



def sumMax_IoU(Y, Y_hat, epsilon=1e-6):

    n=Y.get_shape().as_list()[2]

    norm_a = norm1_tf(Y)
    norm_b = norm1_tf(Y_hat)

    """ sca[k] = sca( Y[:,:,k] Y_hat[:,:,k])      """
    sca = norm1_tf(Y * Y_hat)

    F_IoU =  sca / ( epsilon + norm_a+norm_b-sca)

    """ 1/n *   sum_k  F_IoU(Y[:,:,k],Y_hat[:,:,k])   """
    correl_simple=  tf.reduce_sum(F_IoU) /n

    return correl_simple



def sumMax_IoU_batch(Y, Y_hat, epsilon=1e-6):

    n=Y.get_shape().as_list()[3]

    norm_a = norm1_tf_dec(Y)
    norm_b = norm1_tf_dec(Y_hat)

    """ sca[b,k] = sca( Y[b,:,:,k] Y_hat[b,:,:,k])      """
    sca = norm1_tf_dec(Y * Y_hat)

    F_IoU =  sca / ( epsilon + norm_a+norm_b-sca)

    """ 1/n *   sum_k  F_IoU(Y[:,:,k],Y_hat[:,:,k])   """
    return tf.reduce_sum(F_IoU) /n




import time
def test_optimization(withBackground:bool):
    img_size = 50
    nbInstances = 40


    Xs,Ys=oneBatch_of_rect_instances(1, img_size, nbInstances, withBackground)
    X,Y=Xs[0],Ys[0]
    X=np.reshape(X,[img_size,img_size])

    print("X.shape,Y.shape",X.shape,Y.shape)
    nbCat = Y.shape[2]


    plt.figure()
    """ attention,  cela masque la cat 0 """
    plt.imshow(np.argmax(Y,axis=2),cmap="jet")


    _Y = tf.constant(Y)

    init=np.random.normal(0,0.1,size=[img_size,img_size,nbCat]).astype(np.float32)
    pre_Y_hat = tf.Variable(initial_value=init)
    _Y_hat=tf.nn.softmax(pre_Y_hat,dim=2)

    _loss = - invariant_IoU(_Y,_Y_hat)



    opt=tf.train.AdamOptimizer(1e-2).minimize(_loss )

    plt.figure()
    plt.ion()

    Y_hat_cat=None
    begin=time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            itr=0
            contin=True
            nbConsecutive=0
            Y_hat_cat_prev=np.zeros(X.shape)

            while contin :  #contin ou True
                itr+=1
                sess.run(opt)

                if itr%10==0:

                    [Y_hat,loss]=\
                        sess.run([_Y_hat,_loss])
                    print("itr:%d loss %f" %(itr,loss))
                    Y_hat_cat=probaToImg(Y_hat,withBackground)
                    plt.imshow(Y_hat_cat,cmap="jet")
                    plt.pause(0.001)
                    plt.draw()
                    plt.clf()



                    """sans le background, au début rien ne bouge. Dès que cela bouge, on attend un palier pour s'arréter """
                    if norm1_np(np.abs(Y_hat_cat - Y_hat_cat_prev)) == 0 and norm1_np(Y_hat_cat) > 1:
                        nbConsecutive += 1
                    else:
                        nbConsecutive = 0

                    contin = nbConsecutive < 10

                    Y_hat_cat_prev = Y_hat_cat

            print("DURATION:", time.time() - begin)


            """ photo finish"""
            plt.imshow(Y_hat_cat, cmap="jet")
            plt.pause(100)
            plt.draw()
            plt.clf()



        except KeyboardInterrupt:
            pass


def test_optimization_batch(withBackground: bool):
    img_size = 50
    nbInstances = 12
    batch_size=2

    Xs, Ys = oneBatch_of_rect_instances(1, img_size, nbInstances, withBackground)
    X, Y = Xs[0], Ys[0]
    X = np.reshape(X, [img_size, img_size])

    print("X.shape,Y.shape", X.shape, Y.shape)
    nbCat = Y.shape[2]

    plt.figure()
    """ attention,  cela masque la cat 0 """
    plt.imshow(np.argmax(Y, axis=2), cmap="jet")

    _Ys = tf.constant(Ys)

    init = np.random.normal(0, 0.1, size=[batch_size,img_size, img_size, nbCat]).astype(np.float32)
    pre_Ys_hat = tf.Variable(initial_value=init)

    """ attention, dim+=1 """
    _Ys_hat = tf.nn.softmax(pre_Ys_hat, dim=3)

    _loss = - invariant_IoU_batch(_Ys, _Ys_hat)

    opt = tf.train.AdamOptimizer(1e-2).minimize(_loss)

    plt.figure()
    plt.ion()

    Y_hat_cat = None
    begin = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            itr = 0
            contin = True
            nbConsecutive = 0
            Y_hat_cat_prev = np.zeros(X.shape)

            while contin:  # contin ou True
                itr += 1
                sess.run(opt)

                if itr % 10 == 0:

                    [Ys_hat, loss] = \
                        sess.run([_Ys_hat, _loss])
                    print("itr:%d loss %f" % (itr, loss))
                    Y_hat_cat = probaToImg(Ys_hat[0], withBackground)
                    plt.imshow(Y_hat_cat, cmap="jet")
                    plt.pause(0.001)
                    plt.draw()
                    plt.clf()

                    """sans le background, au début rien ne bouge. Dès que cela bouge, on attend un palier pour s'arréter """
                    if norm1_np(np.abs(Y_hat_cat - Y_hat_cat_prev)) == 0 and norm1_np(Y_hat_cat) > 1:
                        nbConsecutive += 1
                    else:
                        nbConsecutive = 0

                    contin = nbConsecutive < 10

                    Y_hat_cat_prev = Y_hat_cat

            print("DURATION:", time.time() - begin)

            """ photo finish"""
            plt.imshow(Y_hat_cat, cmap="jet")
            plt.pause(100)
            plt.draw()
            plt.clf()



        except KeyboardInterrupt:
            pass


def test_bipartite_matching():

    tf.InteractiveSession()
    A=tf.constant([[-1.,-10,0],[0,0,-2],[-3,0,0]])
    r2c,c2r=image_ops.bipartite_match(A, 3)

    A_perm=tf.gather(A,c2r)
    min_elem=tf.diag_part(A_perm)

    print("A\n",A.eval())
    print("c2r",c2r.eval())
    print("r2c",r2c.eval())

    print("A_perm\n",A_perm.eval())
    print("min_elem\n",min_elem.eval())



def probaToImg(Y_proba,withBackground:bool,threshold=0.8):

    assert len(Y_proba.shape)==3
    img = np.argmax(Y_proba, axis=2)

    if not withBackground:
        img+=1
        max_proba=np.max(Y_proba,axis=2)
        img[max_proba<threshold]=0
    return img


def test_probaToImg():
    Y_proba=np.random.random(size=[4,4,2])
    img=probaToImg(Y_proba,False)
    print(img)


#
#
# def test_optimization_without_background_batch():
#     img_size = 50
#     max_nb_instance = 10
#     batch_size=5
#     nbCat=max_nb_instance
#
#     imgs,Ys = make_instances_GT_batch(img_size, nbCat,batch_size ,False)
#
#     imgs=np.reshape(imgs,[batch_size,img_size,img_size])
#
#
#     plt.figure()
#     plt.imshow(imgs[0],cmap="jet")
#
#     _Ys = tf.constant(Ys)
#
#     init=np.random.normal(0,0.1,size=[batch_size,img_size,img_size,nbCat]).astype(np.float32)
#
#     pre_Y_hat = tf.Variable(initial_value=init)
#
#     _Ys_hat=tf.nn.softmax(pre_Y_hat,dim=3)
#     _loss_simple = -invariant_IoU_batch(_Ys, _Ys_hat)
#
#     opt=tf.train.AdamOptimizer(1e-2).minimize(_loss_simple)
#
#     plt.figure()
#     plt.ion()
#
#     Y_hat_cat=None
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#
#
#         try:
#             itr=0
#             contin=True
#             Y_hat_cat_prev=np.zeros(imgs[0].shape)
#             nbConsecutive=0
#
#             while contin :  #contin ou True
#                 itr+=1
#                 sess.run(opt)
#                 if itr%10==0:
#
#                     Ys_hat,loss_simple,=sess.run([_Ys_hat,_loss_simple])
#                     print("itr:%d loss_simple:%.4f" %(itr,loss_simple))
#
#                     Y_hat_cat=probaToImg(Ys_hat[0])
#
#
#                     plt.imshow(Y_hat_cat,cmap="jet")
#                     plt.colorbar()
#                     plt.pause(0.001)
#                     plt.draw()
#                     plt.clf()
#
#                     """ au début rien ne bouge. Dès que cela bouge, on attend un palier pour s'arréter """
#                     if norm1_np(np.abs(Y_hat_cat-Y_hat_cat_prev))==0 and norm1_np(Y_hat_cat)>1: nbConsecutive+=1
#                     else : nbConsecutive=0
#
#                     contin= nbConsecutive<10
#
#                     Y_hat_cat_prev=Y_hat_cat
#
#
#             """ photo finish"""
#             plt.imshow(Y_hat_cat, cmap="jet")
#             plt.colorbar()
#             plt.pause(100)
#             plt.draw()
#             plt.clf()
#
#
#         except KeyboardInterrupt:
#             pass
#


def test_zero_dist():

    img_size = 28
    nbInstances=12
    withBackground=False



    Xs,Ys=oneBatch_of_rect_instances(1, img_size, nbInstances, withBackground)
    X,Y=Xs[0],Ys[0]
    print("X.shape,Y.shape",X.shape,Y.shape)
    nb_cat = Y.shape[2]


    plt.figure()
    for i in range(nb_cat):
        plt.subplot(4,4,i+1)
        plt.imshow(Y[:,:,i],cmap="gray")
        plt.title("cat:"+str(i))


    _Y = tf.constant(Y)
    _Y_hat = tf.constant(Y)

    _ioU1 = invariant_IoU(_Y, _Y_hat)
    _ioU2 = matching_IoU(_Y, _Y_hat)



    #
    # Xs, Ys = make_instances_GT_batch(img_size, nb_instances_max, 3, True)
    # _Ys = tf.constant(Ys)
    #
    # _Ys_hat = tf.constant(Ys)
    # _correl_batch = invariant_IoU_batch(_Ys, _Ys_hat)
    #
    #

    begin=time.time()
    with tf.Session() as sess:
        ioUs=sess.run([_ioU1,_ioU2])
        print(ioUs)

    print(time.time()-begin)

    plt.show()

    #     correl,correl_batch=sess.run([_correl,_correl_batch])
    #     print("correl",correl)
    #     print("correl_batch",correl_batch)



if __name__=="__main__":
    #test_zero_dist()

    test_optimization_batch(False)

    #test_bipartite_matching()
    #test_zero_dist()





    """
    img_size = 50
    max_nb_instance = 40
    crossEntropyFactor 
    oo  :  480  (17 s ! )   = naive crossEntropy
    10  :  560  (70 s)
    5   :  350
    1   :  300 ( 37 s)   
    0   :  280 ( 34 s )



    Version naive IoU:
    350, (11 s)  proche de la naive crossEntropy


    ________________

    10 instances

    loss_IoU=invariant_IoU

    itr:270 loss -0.268555,loss_IoU -0.268555,loss_secondary 0.000000  
    DURATION: 7.908565044403076

    loss_secondary = invariant_IoU(1-Y,1-Y_hat)
    itr:270 loss -0.761673,loss_IoU -0.264662,loss_secondary -0.497012  
    DURATION: 10.375704050064087


    20 instance. 3 cat en plus
    itr:300 loss -0.204267,loss_IoU -0.204267,loss_secondary 0.000000  
    DURATION: 21.792906999588013
    Contre sans les 3 cat 
    itr:280 loss -0.266388,loss_IoU -0.266388,loss_secondary 0.000000  
    DURATION: 13.765146970748901


    """



    """ 
    VARIANTES
    
    
    J'ai essayer avec la formule + compliquée (en prenant le max par colonne, avec/sans renormalisation par
    le nombre d'instance estimée). C'est plus long. 
    
    
    J'ai essayer en prenant une puissance de f_UoI, cela détériore
    
    power=1 : 280 itr
        
        power, itr
        
        0.5: 350
        0.8 : 290
        1.2: 280
        1.5: 450
            
    en pondérant par la norme de Y : c'est affreux
    en pondérant par la norme de Y_hat : c'est idem et c'est logique 
    le produit des deux norme : il regroupe des classes. 
       
        
      """







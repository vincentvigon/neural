import tensorflow as tf
#import numpy as np
import matplotlib
import numpy as np
np.random.seed(1234132)




from ne_ne.TasteExample.E_instances.dataDealer import make_instances_GT

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt


def norm1_tf(Y):
    return tf.reduce_sum(Y,axis=[0,1])


def norm1_np(a):
    return np.sum(np.sum(a,axis=0),axis=0)

"""
   Avec facteur 1/n : la correl atteint son max en 1.
   Cela ne change rien au niveau de l'optimisation
   1/n  sum_t=0^n-1  max_s (  f_IoU (Y[t],Y_hat[s]) )  
"""



def two_correl_normalized(Y, Y_hat, nb_pix_to_be_instance:int, epsilon=1e-6):

    n=Y.get_shape().as_list()[2]

    norm_a = norm1_tf(Y)
    norm_b = norm1_tf(Y_hat)

    #todo : essayer smooth version
    nb_instance_hat = tf.reduce_sum(tf.cast(tf.greater( norm_b , nb_pix_to_be_instance),dtype=tf.float32 ))

    Ya_ext=tf.expand_dims(Y, 3)
    Yb_ext=tf.expand_dims(Y_hat, 2)

    """ sca[k,l] = sca( Y[:,:,k] Y_hat[:,:,l])      """
    sca = norm1_tf(Ya_ext * Yb_ext)

    norm_a=tf.expand_dims(norm_a,1)
    norm_b=tf.expand_dims(norm_b,0)

    F_IoU =  sca / ( epsilon + norm_a+norm_b-sca)

    """ 1/n *   sum_k  max_l F_IoU[k,l] = 1/n *  sum_k  max_l F_IoU(Ya[k],Yb[l])   """
    correl_simple=  tf.reduce_sum(tf.reduce_max(F_IoU, axis=1)) /n
    """ 1/n_hat *  sum_l  max_k F_IoU[k,l] =  sum_l  max_k F_IoU(Y[k],Y_hat[l])   """
    #todo essayer de tronquer la somme (double peine)
    correl_complicated=  tf.reduce_sum(tf.reduce_max(F_IoU, axis=0))  /nb_instance_hat

    return -correl_simple,-correl_complicated






def TEST_correl_simple_normalized(Y, Y_hat, epsilon=1e-6):

    n=Y.get_shape().as_list()[2]
    print("tf.shape(Y)",n)

    norm_a = norm1_tf(Y)
    norm_b = norm1_tf(Y_hat)
    print("norm_a",norm_a.eval())
    print("norm_b",norm_b.eval())

    Ya_ext=tf.expand_dims(Y, 3)
    Yb_ext=tf.expand_dims(Y_hat, 2)

    """ sca[k,l] = sca( Y[:,:,k] Y_hat[:,:,l])      """
    sca = norm1_tf(Ya_ext * Yb_ext)
    print("sca",sca.eval())

    norm_a=tf.expand_dims(norm_a,1)
    norm_b=tf.expand_dims(norm_b,0)

    F_IoU =  sca / ( epsilon + norm_a+norm_b-sca)

    print("F_IoU",F_IoU.eval())


    """ 1/n *   sum_k  max_l F_IoU[k,l] = 1/n *  sum_k  max_l F_IoU(Ya[k],Yb[l])   """
    return  tf.reduce_sum(tf.reduce_max(F_IoU, axis=1)) /n





def test():

    tf.InteractiveSession()

    img_size = 10
    nb_cat_max = 8
    Y,img=make_instances_GT(img_size, nb_cat_max, True)
    nb_cat=Y.shape[2]

    perm=np.random.permutation(nb_cat)
    print("perm",perm)
    Y_b=Y[:,:,perm]

    Y=tf.constant(Y)
    Y_b=tf.constant(Y_b)


    print("correl_simple_normalized", TEST_correl_simple_normalized(Y, Y_b, epsilon=0).eval())
    print("-" * 20)

    #print("correl_complicated_normalized", correl_complicated_normalized(Y, Y_b, nb_pix_to_be_instance=5, epsilon=0))


"""
IMPORTANT : il faut laisser le background comme categorie, sinon ils seront affecté à une carégorie au hasard quand 
on prend l'argmax
"""
def test_optimization():
    img_size = 50
    nb_cat_max = 20
    Y, img = make_instances_GT(img_size, nb_cat_max)
    plt.figure()
    plt.imshow(img,cmap="jet")

    Y_shape=Y.shape

    Y = tf.constant(Y)
    pre_Y_hat = tf.Variable(initial_value=np.random.normal(0,0.1,size=Y_shape).astype(np.float32))
    _Y_hat=tf.nn.softmax(pre_Y_hat,dim=2)
    _loss_simple, _loss_complicated = two_correl_normalized(Y, _Y_hat, 5)

    opt=tf.train.AdamOptimizer(1e-2).minimize(_loss_simple)
    _Y_hat_cat=tf.argmax(_Y_hat,dimension=2)

    plt.figure()
    plt.ion()

    Y_hat_cat=None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            itr=0
            contin=True
            Y_hat_cat_prev=np.zeros(img.shape)

            while contin:
                itr+=1
                sess.run(opt)
                if itr%10==0:

                    Y_hat_cat,loss_simple,loss_complicated,=sess.run([_Y_hat_cat,_loss_simple,_loss_complicated])
                    print("itr:%d loss_simple:%.4f loss_complicated:%.4f" %(itr,loss_simple,loss_complicated))


                    plt.imshow(Y_hat_cat,cmap="jet")
                    plt.pause(0.001)
                    plt.draw()
                    plt.clf()

                    """ ce n'est pas un bon critère d'arrêt: avec la loss_complicated on peut avoir 2 images fausses et identiques d'affilée """
                    contin=norm1_np(np.abs(Y_hat_cat-Y_hat_cat_prev))>0
                    Y_hat_cat_prev=Y_hat_cat


            """ photo finish"""
            plt.imshow(Y_hat_cat, cmap="jet")
            plt.pause(100)
            plt.draw()
            plt.clf()


        except KeyboardInterrupt:
            pass





test_optimization()




import numpy as np
import matplotlib

from ne_ne.TasteExample.E_instances.dataDealer import make_instances_GT

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv


def norm1(a):
    return np.sum(np.sum(a,axis=0),axis=0)


def correl_both_unnormalized(Ya, Yb, threshold, epsilon=1e-6):


    norm_a = norm1(Ya)
    norm_b = norm1(Yb)
    print("norm_a",norm_a)
    print("norm_b",norm_b)

    nb_instance_a=  np.sum(norm_a>threshold)
    nb_instance_b=  np.sum(norm_b>threshold)

    print("nb_instance_a",nb_instance_a)
    print("nb_instance_b",nb_instance_b)

    Ya_ext=np.expand_dims(Ya,3)
    Yb_ext=np.expand_dims(Yb,2)

    """ sca[k,l] = sca( Ya[:,:,k] Yb[:,:,l])      """
    sca = norm1(Ya_ext * Yb_ext)
    print("sca",sca)

    norm_a=np.expand_dims(norm_a,1)
    norm_b=np.expand_dims(norm_b,0)

    F_IoU =  sca / ( epsilon + norm_a+norm_b-sca)

    print("F_IoU",F_IoU)

    """dist_1= sum_l  max_k F_IoU[k,l] =  sum_l  max_k F_IoU(Ya[k],Yb[l])   """
    dist_1= np.sum(np.max(F_IoU,axis=0))
    """dist_2= sum_k  max_l F_IoU[k,l] =  sum_k  max_l F_IoU(Ya[k],Yb[l])   """
    dist_2= np.sum(np.max(F_IoU,axis=1))

    return dist_1,dist_2


"""
   Avec facteur 1/n : la correl atteint son max en 1.
   Cela ne change rien au niveau de l'optimisation
   1/n  sum_t=0^n-1  max_s (  f_IoU (Y[t],Y_hat[s]) )  
"""
def correl_simple_normalized(Y, Y_hat, epsilon=1e-6):

    n=Y.shape[2]

    norm_a = norm1(Y)
    norm_b = norm1(Y_hat)
    print("norm_a",norm_a)
    print("norm_b",norm_b)

    Ya_ext=np.expand_dims(Y, 3)
    Yb_ext=np.expand_dims(Y_hat, 2)

    """ sca[k,l] = sca( Y[:,:,k] Y_hat[:,:,l])      """
    sca = norm1(Ya_ext * Yb_ext)
    print("sca",sca)

    norm_a=np.expand_dims(norm_a,1)
    norm_b=np.expand_dims(norm_b,0)

    F_IoU =  sca / ( epsilon + norm_a+norm_b-sca)

    print("F_IoU",F_IoU)


    """dist_2= 1/n *   sum_k  max_l F_IoU[k,l] = 1/n *  sum_k  max_l F_IoU(Ya[k],Yb[l])   """
    dist_2 = np.sum(np.max(F_IoU, axis=1)) /n

    return dist_2



"""
    1/n_hat  sum_t=0^n-1    max_s (  f_IoU (Y[t],Y_hat[s]) )  



   Le tenseur Y_hat contient des probabilités de présence de pixel. 
   On considère qu'il y a une instance quand la somme des probas à une profondeur donnée est >=nb_pix_to_be_instance
   
   
   On pénalise ensuite par ce nombre d'instance estimé (mais on somme quand même sur toutes les instances, pour éviter la double peine)
   Ainsi cette distance préférera 1 grande instance à deux petites (dont l'union fait la grande)
"""
def correl_complicated_normalized(Y, Y_hat, nb_pix_to_be_instance, epsilon=1e-6):

    norm_a = norm1(Y)
    norm_b = norm1(Y_hat)
    print("norm_a",norm_a)
    print("norm_b",norm_b)

    #TODO : mettre un équivalent lisse. Ex x->sigma(x-threshold)
    nb_instance_hat=  np.sum(norm_b > nb_pix_to_be_instance)

    print("nb_instance_b",nb_instance_hat)

    Ya_ext=np.expand_dims(Y, 3)
    Yb_ext=np.expand_dims(Y_hat, 2)

    """ sca[k,l] = sca( Ya[:,:,k] Yb[:,:,l])      """
    sca = norm1(Ya_ext * Yb_ext)
    print("sca",sca)

    norm_a=np.expand_dims(norm_a,1)
    norm_b=np.expand_dims(norm_b,0)

    F_IoU =  sca / ( epsilon + norm_a+norm_b-sca)

    print("F_IoU",F_IoU)

    """dist_1= 1/n_hat *  sum_l  max_k F_IoU[k,l] =  sum_l  max_k F_IoU(Y[k],Y_hat[l])   """
    dist_1=   np.sum(np.max(F_IoU,axis=0)) /nb_instance_hat

    return dist_1




def test():
    img_size = 10
    nb_cat_max = 8

    Y_mod,img=make_instances_GT(img_size, nb_cat_max)
    nb_cat=Y_mod.shape[2]

    perm=np.random.permutation(nb_cat)
    print(perm)
    Y_mod_b=Y_mod[:,:,perm]



    print("correl_both_unnormalized=", correl_both_unnormalized(Y_mod, Y_mod_b, 5, epsilon=0))
    print("-"*20)

    print("correl_simple_normalized", correl_simple_normalized(Y_mod, Y_mod_b, epsilon=0))
    print("-" * 20)

    print("correl_complicated_normalized", correl_complicated_normalized(Y_mod, Y_mod_b, nb_pix_to_be_instance=5, epsilon=0))



test()



































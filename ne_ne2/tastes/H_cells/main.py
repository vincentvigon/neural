import matplotlib

from ne_ne.TasteExample.G_rest_strates.dataDealer import oneBatch_of_rect_stratify
from ne_ne.TasteExample.G_rest_strates.model import Model_fullyConv_regCat

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import numpy as np
np.set_printoptions(linewidth=3000,precision=6,suppress=True)
import matplotlib.pyplot as plt
import tensorflow as tf
import time




def tasteClassif():

    logDir="../log"

    reset=True
    if reset and tf.gfile.Exists(logDir):
        tf.gfile.DeleteRecursively(logDir)


    img_size=28
    nbStrates=6

    """attention Y est d'ordre 4 : chaque strate représentant une catégorie = une instance ou le background"""
    dataDealer=lambda batchSize: oneBatch_of_rect_stratify(either__any_disjoint_separated=0, img_size=img_size, batchSize=16, nbRectPerImg=2, deltaRange=(8, 12),nbStrates=nbStrates)

    #Xs, Ys_cat,Ys_background=dataDealer(1)
    #print("Ys_cat.shape:",Ys_cat.shape)



    """ img """
    nbChannel=1
    model = Model_fullyConv_regCat(img_size, img_size, nbChannel, nbStrates,favoritism=None,depth0=128,depth1=64)
    model.verbose = True
    model.nbConsecutiveOptForOneFit=1
    batchSize=50


    model.learning_rate=1e-3

    now = time.time()
    summary_writer_train = tf.summary.FileWriter(logDir+"/"+str(now), model.sess.graph)


    nbStep=2000

    lossTrain=np.zeros(nbStep)
    lossValid = np.zeros(nbStep)
    lossValid[:]=np.nan



    try:
        for itr in range(nbStep):
            print("itr:", itr)

            X,  Y_cat,Ys_background = dataDealer(batchSize)
            model.fit(X, Y_cat,Ys_background,itr)

            lossTrain[itr] = model.loss
            summary_writer_train.add_summary(model.summary,global_step=itr)

            # if itr==10:
            #     model.learning_rate=5e-4

            if itr%20==0:
                print("\nVALIDATION")
                X, Y_cat,Ys_background = dataDealer(batchSize)
                model.validate(X, Y_cat,Ys_background,itr)
                lossValid[itr] = model.loss
                print("\n")


    except  KeyboardInterrupt:
        print("on a stoppé")


    plt.subplot(1, 2, 2)
    plt.plot(lossTrain, label="train")
    plt.plot(lossValid, '.',label="valid class")

    plt.title("loss")
    plt.legend()


    """ TEST"""
    X, Y_cat,Ys_background = dataDealer(16)
    Y_cat_hat,_=model.predict(X)
    print("Y_cat_hat.shape", Y_cat_hat.shape)

    X=X[:,:,:,0]


    vmax=nbStrates
    vmin=0
    drawOne(X,"X",vmin,vmax,cmap="gray")


    drawOne(np.sum(Y_cat_hat,axis=3), "hat cat", vmin, vmax, "jet")

    drawOne(np.sum(Y_cat,axis=3), "Y_cat", vmin, vmax, "jet")










    # draw_proba_Vs_cat=1
    # if draw_proba_Vs_cat==0:
    #     for i in range(0,nbCat):
    #         drawOne(hat_Y_test_proba[:, :, :,i],"proba cat:"+str(i-1),1)
    # else:
    #     drawOne(hat_Y_test_cat[:, :, :], "hat_cat", nbCat)


    plt.show()
    model.close()





tasteClassif()
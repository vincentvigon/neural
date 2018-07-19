import matplotlib

from ne_ne.TasteExample.F_instance_by_regression.model import Model_fullyConv_regCat
from ne_ne.TasteExample.F_instance_by_regression.dataDealer import oneBatch_of_rect_stratify

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import numpy as np
np.set_printoptions(linewidth=3000,precision=6,suppress=True)
import matplotlib.pyplot as plt
import tensorflow as tf





def drawOne(imgs,title,vmin,vmax,cmap):

    plt.figure().suptitle(title)
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(imgs[i],cmap=cmap,vmin=vmin,vmax=vmax)

    plt.subplots_adjust(bottom=0.1,top=0.9, left=0.1, right=0.8)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)







import time
def tasteClassif():

    logDir="/Users/vigon/GoogleDrive/permanent/python/neurones/ne_ne/log"

    reset=True
    if reset and tf.gfile.Exists(logDir):
        tf.gfile.DeleteRecursively(logDir)


    img_size=28

    """attention Y est d'ordre 4 : chaque strate représentant une catégorie = une instance ou le background"""
    dataDealer=lambda batchSize: oneBatch_of_rect_stratify(either__any_disjoint_separated=0, img_size=img_size, batchSize=16, nbRectPerImg=2, deltaRange=(8, 12))

    Xs, Ys_cat, Ys_reg=dataDealer(1)

    print("Ys_cat.shape:",Ys_cat.shape)
    print("Ys_reg.shape",Ys_reg.shape)

    nbCat=3# background, inside,border
    nbRegressor=Ys_reg.shape[3]


    """ img """
    nbChannel=1
    model = Model_fullyConv_regCat(img_size, img_size, nbChannel, nbCat,nbRegressor,favoritism=None,depth0=128,depth1=64)
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

            X,  Y_cat,Y_reg = dataDealer(batchSize)
            model.fit(X, Y_cat,Y_reg,itr)

            lossTrain[itr] = model.loss
            summary_writer_train.add_summary(model.summary,global_step=itr)

            # if itr==10:
            #     model.learning_rate=5e-4

            if itr%20==0:
                print("\nVALIDATION")
                X, Y_cat, Y_reg = dataDealer(batchSize)
                model.validate(X, Y_cat,Y_reg,itr)
                lossValid[itr] = model.loss
                print("\n")


    except  KeyboardInterrupt:
        print("on a stoppé")


    plt.subplot(1, 2, 2)
    plt.plot(np.log(lossTrain), label="train")
    plt.plot(np.log(lossValid), '.',label="valid class")

    plt.title("loss")
    plt.legend()




    """ TEST"""
    X, Y_cat, Y_reg = dataDealer(16)
    Y_cat_hat, Y_reg_hat=model.predict(X)
    print("Y_cat_hat.shape", Y_cat_hat.shape)
    print("Y_reg_hat.shape", Y_reg_hat.shape)

    X=X[:,:,:,0]


    vmax=np.max(X)
    vmin=0
    drawOne(X,"X",vmin,vmax,cmap="gray")


    vmax=None
    vmin=None
    for i in range(nbRegressor):
        drawOne(Y_reg_hat[:,:,:,i], "Y_reg_hat"+str(i),vmin,vmax, "jet")


    drawOne(Y_cat_hat[:, :, :], "hat cat", vmin, vmax, "jet")









    # draw_proba_Vs_cat=1
    # if draw_proba_Vs_cat==0:
    #     for i in range(0,nbCat):
    #         drawOne(hat_Y_test_proba[:, :, :,i],"proba cat:"+str(i-1),1)
    # else:
    #     drawOne(hat_Y_test_cat[:, :, :], "hat_cat", nbCat)


    plt.show()
    model.close()





tasteClassif()
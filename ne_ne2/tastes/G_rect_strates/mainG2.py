import matplotlib

from ne_ne.TasteExample.G_rest_strates.model import Model_fullyConv_regCat
from ne_ne2.VISU import draw16imgs
from ne_ne2.tastes.G_rect_strates.dataDealerG2 import oneBatch_of_rect_stratify
from ne_ne2.tastes.G_rect_strates.modelG2 import Model_G

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import numpy as np
np.set_printoptions(linewidth=3000,precision=6,suppress=True)
import matplotlib.pyplot as plt
import tensorflow as tf





import time

def taste():

    logDir="../log"

    reset=True
    if reset and tf.gfile.Exists(logDir):
        tf.gfile.DeleteRecursively(logDir)


    img_size=28
    nbStrates=6

    """attention Y est d'ordre 4 : chaque strate représentant une catégorie = une instance ou le background"""
    dataDealer=lambda batchSize: oneBatch_of_rect_stratify(either__any_disjoint_separated=0, img_size=img_size, batchSize=16, nbRectPerImg=2, deltaRange=(8, 12),nbStrates=nbStrates)



    """ img """
    nbChannel=1
    model = Model_G(img_size, img_size, nbChannel, nbStrates,2,32)
    #model=Model_fullyConv_regCat(img_size,img_size,nbChannel,nbStrates,None,30,60)
    model.verbose = True
    model.nbConsecutiveOptForOneFit=1
    batchSize=50


    model.learning_rate=1e-2

    now = time.time()
    summary_writer_train = tf.summary.FileWriter(logDir+"/"+str(now), model.sess.graph)


    nbStep=2000

    lossTrain=[]
    lossValid = []



    try:
        for itr in range(nbStep):
            print("itr:", itr)

            X,  Y_cat,Y_background = dataDealer(batchSize)
            model.fit(X, Y_cat,Y_background,itr)

            lossTrain.append(model.loss)
            summary_writer_train.add_summary(model.summary,global_step=itr)


            if itr%20==0:
                print("\nVALIDATION")
                X, Y_cat,Y_background = dataDealer(batchSize)
                model.validate(X, Y_cat,Y_background,itr)
                lossValid.append(model.loss)
                print("\n")
            else:
                lossValid.append(model.loss)


    except  KeyboardInterrupt:
        print("on a stoppé")


    plt.subplot(1, 2, 2)
    plt.plot(lossTrain, label="train")
    plt.plot(lossValid, '.',label="valid class")

    plt.title("loss")
    plt.legend()


    """ TEST"""
    X, Y_cat,Y_background = dataDealer(16)
    hat_Y_cat,hat_Y_cat_sum=model.predict(X)

    print("hat_Y_cat.shape", hat_Y_cat.shape)



    draw16imgs(X,"X",vmin=0,vmax=1,cmap="gray")

    #draw16imgs(hat_Y_background)

    draw16imgs(hat_Y_cat_sum, "hat cat", 0, nbStrates, "jet")
    draw16imgs(np.sum(Y_cat,axis=3), "Y_cat", 0, nbStrates, "jet")






    # draw_proba_Vs_cat=1
    # if draw_proba_Vs_cat==0:
    #     for i in range(0,nbCat):
    #         drawOne(hat_Y_test_proba[:, :, :,i],"proba cat:"+str(i-1),1)
    # else:
    #     drawOne(hat_Y_test_cat[:, :, :], "hat_cat", nbCat)


    plt.show()
    model.close()





taste()
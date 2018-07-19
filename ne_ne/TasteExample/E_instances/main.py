import matplotlib

from ne_ne.TasteExample.E_instances.dataDealer import oneBatch_of_rect_instances
from ne_ne.TasteExample.E_instances.model import Model_fullyConv_instances

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import numpy as np
np.set_printoptions(linewidth=3000,precision=6,suppress=True)
import matplotlib.pyplot as plt
import tensorflow as tf






def drawOne(imgs,title,vmax,cmap="jet"):

    plt.figure().suptitle(title)
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(imgs[i],cmap=cmap,vmin=0,vmax=vmax)

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
    nb_instance_max=4

    """ +1 pout le background """
    nbCat = nb_instance_max+1

    """attention Y est d'ordre 4 : chaque strate représentant une catégorie = une instance ou le background"""
    def dataDealer(batchSize):
        oneBatch_of_rect_instances(batchSize=batchSize, img_size=img_size, nbLinesPerImage=nb_instance_max)



    model = Model_fullyConv_instances(28, 28, 1, nbCat,favoritism=None,depth0=128,depth1=64)
    model.verbose = True
    model.nbConsecutiveOptForOneFit=10
    batchSize=10


    model.learning_rate=1e-4

    now = time.time()
    summary_writer_train = tf.summary.FileWriter(logDir+"/"+str(now), model.sess.graph)


    nbStep=200

    lossTrain=np.zeros(nbStep)
    lossValid = np.zeros(nbStep)
    lossValid[:]=np.nan



    try:
        for itr in range(nbStep):
            print("itr:", itr)

            X_batch, Y_batch = dataDealer(batchSize)
            model.fit(X_batch, Y_batch,itr)

            lossTrain[itr] = model.loss
            summary_writer_train.add_summary(model.summary,global_step=time.time()-now)

            # if itr==10:
            #     model.learning_rate=5e-4

            if itr%5==0:
                print("\nVALIDATION")
                X_val, Y_val = dataDealer(batchSize)
                model.validate(X_val, Y_val,itr)
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
    X_test,Y_test = dataDealer(16)
    hat_Y_test_cat,hat_Y_test_proba=model.predict(X_test)
    print("X_test.shape", X_test.shape)
    print("Y_test.shape", Y_test.shape)

    X_test=X_test.reshape([16,28,28])


    drawOne(X_test,"X",1,cmap="gray")
    drawOne(np.argmax(Y_test,axis=3),"Y",nbCat)


    draw_proba_Vs_cat=1
    if draw_proba_Vs_cat==0:
        for i in range(0,nbCat):
            drawOne(hat_Y_test_proba[:, :, :,i],"proba cat:"+str(i-1),1)
    else:
        drawOne(hat_Y_test_cat[:, :, :], "hat_cat", nbCat)


    plt.show()
    model.close()





tasteClassif()
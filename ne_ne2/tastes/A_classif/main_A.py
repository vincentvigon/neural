import matplotlib

from ne_ne2.VISU import draw16imgs
from ne_ne2.tastes.A_classif.data_dealer_A import batchOneVignette, batchSeveralVignette
from ne_ne2.tastes.A_classif.model_A import Model_A

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
    multi_label=False


    if multi_label:
        dataDealer=lambda batchSize: batchSeveralVignette(batchSize)
        nbCat=3

    else:
        dataDealer = lambda batchSize: batchOneVignette(batchSize)
        nbCat=3

    #Xs, Ys_cat,Ys_background=dataDealer(1)
    #print("Ys_cat.shape:",Ys_cat.shape)


    model = Model_A(multi_label,img_size, img_size,1,nbCat)
    model.verbose = True

    batchSize=50
    model.learning_rate=1e-3

    now = time.time()
    summary_writer = tf.summary.FileWriter(logDir+"/"+str(now), model.sess.graph)


    nbStep=2000


    lossTrain=[]
    lossValid = []
    accuracyTrain=[]
    accuracyValide=[]



    try:
        for itr in range(nbStep):
            print("itr:", itr)

            X,  Y = dataDealer(batchSize)
            model.fit(X, Y)

            lossTrain.append(model.loss)
            accuracyTrain.append(model.accuracy)

            summary_writer.add_summary(model.summary, global_step=itr)

            if itr%20==0:
                print("\nVALIDATION")
                X, Y = dataDealer(batchSize)
                model.validate(X, Y)
                lossValid.append(model.loss)
                accuracyValide.append(model.accuracy)
                print("\n")
            else:
                lossValid.append(np.nan)
                accuracyValide.append(np.nan)




    except  KeyboardInterrupt:
        print("on a stoppé")


    plt.subplot(1, 2, 1)
    plt.plot(accuracyTrain, label="train")
    plt.plot(accuracyValide, '.',label="valid")
    plt.title("accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(lossTrain, label="train")
    plt.plot(lossValid, '.',label="valid")
    plt.title("loss")
    plt.legend()


    """ TEST"""
    X, Y = dataDealer(16)
    hat_Y=model.predict(X)


    if multi_label:
        subTitles= (hat_Y>0.5).astype(np.int32)
    else:
        subTitles=list(zip(np.argmax(Y,axis=1),np.argmax(hat_Y,axis=1)))
        print(subTitles)

    draw16imgs(X,"X",0,1,cmap="gray",subTitles=subTitles)



    plt.show()
    model.close()





tasteClassif()
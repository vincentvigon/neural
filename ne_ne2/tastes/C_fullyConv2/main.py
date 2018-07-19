import matplotlib
from ne_ne2.VISU import draw16imgs
from ne_ne2.tastes.C_fullyConv2.data_dealer_A import batchOneVignette
from ne_ne2.tastes.C_fullyConv2.model_C2 import Model_C

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import numpy as np
np.set_printoptions(linewidth=3000,precision=2,suppress=True)
import matplotlib.pyplot as plt
import tensorflow as tf



import time
def taste():

    logDir="../log"
    resetLog=False
    if resetLog:
        if tf.gfile.Exists(logDir):
            tf.gfile.DeleteRecursively(logDir)


    model = Model_C(28, 28, 1, nbCategories=2,favoritism=(1,5))
    model.verbose = True


    now = time.time()
    summary_writer = tf.summary.FileWriter(logDir+"/"+str(now), model.sess.graph)


    lossTrain=[]
    lossValid = []
    accuracyTrain=[]
    accuracyValide=[]



    try:
        nbStep = 2000
        for itr in range(nbStep):
            print("itr:", itr)

            X,  Y = batchOneVignette(50)
            model.fit(X, Y)

            lossTrain.append(model.loss)
            accuracyTrain.append(model.accuracy)

            summary_writer.add_summary(model.summary, global_step=itr)

            if itr%20==0:
                print("\nVALIDATION")
                X, Y = batchOneVignette(50)
                model.validate(X, Y)
                lossValid.append(model.loss)
                accuracyValide.append(model.accuracy)
                print("\n")
            else:
                lossValid.append(np.nan)
                accuracyValide.append(np.nan)




    except  KeyboardInterrupt:
        print("on a stoppé")


    X, Y = batchOneVignette(16)
    hat_Y,hat_Y_proba =model.predict(X)

    draw16imgs(Y,"Y")
    draw16imgs(hat_Y,"hat Y")


    plt.show()

    model.close()



if __name__=="__main__":

    taste()

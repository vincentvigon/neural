import matplotlib

from ne_ne2.VISU import draw16imgs

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2,linewidth=3000)

np.random.seed(1234132)
IMG_SIZE=28




def addSquareVignette(size,position,img,gt):
    """ inverser 0 et 1 pour avoir x et y dans l'ordre """
    img[position[1]:position[1]+size[1],position[0]:position[0]+size[0]]*=0.5
    gt[position[1]:position[1]+size[1],position[0]:position[0]+size[0]]=1




def createImgOneVignette():

    img = np.ones([IMG_SIZE, IMG_SIZE],dtype=np.float32)
    gt = np.zeros([IMG_SIZE, IMG_SIZE],dtype=np.int32)

    KO=True
    while KO:
        center=(np.random.randint(0,IMG_SIZE),np.random.randint(0,IMG_SIZE))
        radius=np.random.randint(4,10)

        pos=(center[0]-radius,center[1]-radius)
        sizeVignette=2*radius

        if 0<=pos[0]<img.shape[0]-sizeVignette and 0<=pos[1]<img.shape[1]-sizeVignette :

            addSquareVignette((sizeVignette,sizeVignette),pos,img,gt)

            KO=False

    return img,gt




def batchOneVignette(batch_size):


    imgs=np.zeros(shape=[batch_size,IMG_SIZE,IMG_SIZE,1],dtype=np.float32)
    gts=np.zeros(shape=[batch_size,IMG_SIZE,IMG_SIZE],dtype=np.int32)


    for i in range(batch_size):

        img,gt=createImgOneVignette()
        imgs[i,:,:,0]=img
        gts[i,:,:]=gt


    return imgs,gts








if __name__=="__main__":

    X,Y=batchOneVignette(16)




    draw16imgs(X,"X",cmap="gray")
    draw16imgs(Y,"Y")



    plt.show()


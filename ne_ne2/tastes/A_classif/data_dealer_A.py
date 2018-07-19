import matplotlib

from ne_ne2.VISU import draw16imgs

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
import imageio
import numpy as np
np.set_printoptions(precision=2,linewidth=3000)

np.random.seed(1234132)
path_perso='/Users/vigon/GoogleDrive/permanent/python/dataSignal/TP_synthese/'
IMG_SIZE=28




def addSquareVignette(size,position,img):
    """ inverser 0 et 1 pour avoir x et y dans l'ordre """
    img[position[1]:position[1]+size[1],position[0]:position[0]+size[0]]*=0.5



def addBlurCircleVignette(size, position, img):

    assert size+position[0]<img.shape[0] and size+position[1]<img.shape[1], "pas les bonnes tailles"

    rangex=2
    x=np.linspace(-rangex,rangex,size)
    y=np.linspace(-rangex,rangex,size)

    xx,yy=np.meshgrid(x,y)

    circle=(1-0.7*np.exp(-xx**2-yy**2))

    """ inverser 0 et 1 pour avoir x et y dans l'ordre """
    img[position[1]:position[1]+size,position[0]:position[0]+size]*=circle



def addCircleVignette(size:int, position:tuple, img):

    assert size+position[0]<img.shape[0] and size+position[1]<img.shape[1], "pas les bonnes tailles"


    rangex=1.1
    x=np.linspace(-rangex,rangex,size)
    y=np.linspace(-rangex,rangex,size)

    xx,yy=np.meshgrid(x,y)

    circle=(xx**2+yy**2)<1
    circle=circle.astype(np.float)
    circle=1-0.5*circle

    """ inverser 0 et 1 pour avoir x et y dans l'ordre """
    img[position[1]:position[1]+size,position[0]:position[0]+size]*=circle



def createImgOneVignette(typeImg:str):

    img = np.ones([IMG_SIZE, IMG_SIZE],dtype=np.float32)

    KO=True
    while KO:
        center=(np.random.randint(0,IMG_SIZE),np.random.randint(0,IMG_SIZE))
        radius=np.random.randint(4,10)

        pos=(center[0]-radius,center[1]-radius)
        sizeVignette=2*radius

        if 0<=pos[0]<img.shape[0]-sizeVignette and 0<=pos[1]<img.shape[1]-sizeVignette :
            if typeImg=="blurCircles":
                addBlurCircleVignette(sizeVignette, pos, img)
            elif typeImg=="squares" :
                addSquareVignette((sizeVignette,sizeVignette),pos,img)
            elif typeImg=="circles" :
                addCircleVignette(sizeVignette, pos, img)

            KO=False

    return img,typeImg





def createImgSeveralVignette():



        img = np.ones([IMG_SIZE, IMG_SIZE],dtype=np.float32)

        nbVignette=0
        wantedNbVignette=np.random.randint(1,4)

        haveBlurCircles=False
        haveSquares=False
        haveCircles=False

        while nbVignette<wantedNbVignette:

            center=(np.random.randint(0,IMG_SIZE),np.random.randint(0,IMG_SIZE))
            radius=np.random.randint(3,5)

            pos=(center[0]-radius,center[1]-radius)
            sizeVignette=2*radius

            coin=np.random.randint(3)

            if 0<=pos[0]<img.shape[0]-sizeVignette and 0<=pos[1]<img.shape[1]-sizeVignette :
                nbVignette += 1

                if coin==0:
                    addBlurCircleVignette(sizeVignette, pos, img)
                    haveBlurCircles=True
                elif coin==1  :
                    addSquareVignette((sizeVignette,sizeVignette),pos,img)
                    haveSquares=True
                else:
                    addCircleVignette(sizeVignette, pos, img)
                    haveCircles = True



        label=np.array([haveBlurCircles,haveSquares,haveCircles]).astype(np.int32)

        return img,label



def batchOneVignette(batch_size):

    labels=np.random.randint(0,3,batch_size).astype(np.int32)

    imgs=np.zeros(shape=[batch_size,IMG_SIZE,IMG_SIZE,1],dtype=np.float32)

    for i in range(batch_size):
        label=labels[i]
        if label==0: type="blurCircles"
        elif label==1: type="squares"
        else : type="circles"
        img,_=createImgOneVignette(type)
        imgs[i,:,:,0]=img

    labels_proba=np.eye(3)[labels]

    return imgs,labels_proba




def batchSeveralVignette(batch_size):

    imgs=np.zeros(shape=[batch_size,IMG_SIZE,IMG_SIZE,1],dtype=np.float32)
    labels=np.zeros(shape=[batch_size,3])

    for i in range(batch_size):

        img,label=createImgSeveralVignette()
        imgs[i,:,:,0]=img
        labels[i,:]=label


    return imgs,labels





def tasteImgs():


    img,label=createImgOneVignette("blurCircles")
    plt.subplot(4,4,1)
    plt.imshow(img,vmin=0,vmax=1)
    plt.title(label)


    img,_=createImgOneVignette("squares")
    plt.subplot(4, 4, 2)
    plt.imshow(img,vmin=0,vmax=1)
    plt.title(label)

    img,_=createImgOneVignette("circles")
    plt.subplot(4, 4, 3)
    plt.imshow(img,vmin=0,vmax=1)
    plt.title(label)



    for i in range(3,16):
        img,label=createImgSeveralVignette()
        plt.subplot(4,4,i+1)
        plt.imshow(img,vmin=0,vmax=1)
        plt.title(str(label))


    plt.show()






if __name__=="__main__":

    #X,Y=batchOneVignette(4)
    X,Y=batchSeveralVignette(16)


    print("X.shape",X.shape)
    print("Y.shape",Y.shape)
    print(Y[0,:])
    print(Y[1, :])
    print(Y[2, :])
    print(Y[3, :])


    draw16imgs(X,subTitles=Y)



    plt.show()


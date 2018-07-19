import numpy as np
import matplotlib

from ne_ne.dataGen.dataLines.weighted_line import weighted_line
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt


def addOneRect_disjoint(point0:tuple, point1:tuple, img:np.ndarray, Y:np.ndarray, cat:int):

    """ les images sont initialisées à 0."""

    partEnlarged=img[point0[0]-1:point1[0]+2, point0[1]-1:point1[1]+2]
    if np.sum(partEnlarged)==0:
        img[point0[0]:point1[0] + 1, point0[1]:point1[1]]=1
        Y[point0[0]:point1[0] + 1, point0[1]:point1[1]] = cat
        return True
    else:
        return False



def __oneBatch_of_rect_instances(img_size:int, batchSize:int, nbLinesPerImg, deltaRange=(2, 7)):

    imgs = np.zeros([batchSize, img_size, img_size], dtype=np.float32)
    Ys = np.zeros([batchSize, img_size, img_size], dtype=np.uint8)

    for b in range(batchSize):

        for j in range(nbLinesPerImg):
            OK=False
            count=0

            while not OK:
                count+=1
                point0 = (np.random.randint(1, img_size - 2), np.random.randint(1, img_size - 2))
                delta=(np.random.randint(deltaRange[0],deltaRange[1]), np.random.randint(deltaRange[0],deltaRange[1]))
                point1 = (point0[0]+delta[0],point0[1]+delta[1])
                OK=   ( (point0!=point1) and 0<=point1[0]<img_size and 0<=point1[1]<img_size  )

                if OK :
                    """ dans j+1, le +1, c'est à cause de la catégorie 0.
                     Renvoit false s'il n'arrive pas à mettre le rectangle de manière disjointe"""
                    OK=addOneRect_disjoint(point0=point0, point1=point1, img=imgs[b], Y=Ys[b], cat=j + 1)
                if count>10000:
                    raise Exception("pas possible de mettre autant de carré de manière disjointe")



    return np.expand_dims(imgs,3),Ys



def oneBatch_of_rect_instances(batchSize, img_size, nbLinesPerImage):

    Xs,Ys=__oneBatch_of_rect_instances(batchSize=batchSize, img_size=img_size, nbLinesPerImg=nbLinesPerImage)

    eye = np.eye(nbLinesPerImage + 1, dtype=np.float32)
    Ys = eye[Ys]


    return Xs,Ys



def test():
    img_size = 28
    nbInstances=5

    Xs,Ys=oneBatch_of_rect_instances(1, img_size, nbInstances)
    X,Y=Xs[0],Ys[0]
    nb_cat = Y.shape[2]


    plt.figure()
    plt.imshow(np.reshape(X,[img_size,img_size]),cmap="gray")
    plt.colorbar()

    plt.figure()
    plt.imshow(np.argmax(Y,axis=2), cmap="jet")
    plt.colorbar()


    plt.figure()

    for i in range(nb_cat):
        plt.subplot(4,4,i+1)
        plt.imshow(Y[:,:,i],cmap="gray")
        plt.title("cat:"+str(i))

    plt.show()


if __name__=="__main__":
    test()

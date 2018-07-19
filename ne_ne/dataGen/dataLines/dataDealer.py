import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
from ne_ne.dataGen.dataLines.weighted_line import weighted_line

import numpy as np
np.set_printoptions(precision=2,linewidth=3000,suppress=True)

np.random.seed(1234132)



MAX_NB_IMG=500
IMG_SIZE=28




""" pour faire de la classification"""
def angle_0_pi_ToCat(angle, nbAngleCat):
    """ la cat 0  est réservée au background"""

    assert 0<=angle<=np.pi, "the angle " + str(angle) +" does not check 0<=angle<np.pi"

    limits =np.linspace(0, np.pi, nbAngleCat + 1)# +1 car il y a 1 baton de plus que d'intervalle
    for i in  range(len(limits)):
        if angle  <= limits[i] :
            return i

    raise ValueError("angle bizarre:" ,angle)

""" pour faire de la régression """



def addOneLine(point0:tuple, point1:tuple, img:np.ndarray, Y_class:np.ndarray,Y_reg:np.ndarray, nbAngleCat:int):
    yy, xx, vals = weighted_line(point0[0], point0[1], point1[0], point1[1], 0)

    dx = point1[0] - point0[0]
    dy = point1[1] - point0[1]

    angle = np.arctan2(dx, dy)

    """ 1 seule catégorie d'angle => on indique seulement si le point appartient à la droite """
    if nbAngleCat!=1:
        cat = angle_0_pi_ToCat(angle%np.pi, nbAngleCat)
        Y_class[xx, yy] = cat
    else:
        Y_class[xx, yy] = 1

    Y_reg[xx, yy] =angle % np.pi

    """ les images sont initialisées à 1."""
    img[xx, yy] *= 1 - vals



def norm(v):
    return np.sqrt(np.sum(v**2))


def angleBetweenTwoVector(v0,v1):
    v0nor=v0/norm(v0)
    v1nor=v1/norm(v1)
    sca=np.dot(v0nor,v1nor)
    if sca>1 : return 0
    if sca<-1: return np.pi
    return np.arccos(sca)





def addOneCurve_Y_is_angle(points:list, img:np.ndarray, Y_class:np.ndarray, Y_reg:np.ndarray, nbAngleCat:int):

    assert 0<nbAngleCat<10

    points=np.array(points)

    vectors=np.zeros([len(points),2])
    vectors[0,:]=np.nan



    for i in range(1,len(points)):
        pointA = points[i - 1]
        pointB=points[i]
        yy, xx, vals = weighted_line(pointA[0], pointA[1], pointB[0], pointB[1], 0)
        vectors[i,:]=pointB-pointA

        """ les images sont initialisées à 1."""
        img[xx, yy] *= 1 - vals


        """
        ICI Deux choix
        - ou bien : Les points 'intérieur' sont mis dans la catégorie 0 (comme le background).
        - ou bien on les mets à cat(pi). Mais cette catégorie devient ultra majoritaire (en dehors du background). 
        """
        Y_class[xx, yy]= nbAngleCat
        Y_reg[xx, yy]=np.pi


    """catégorie minimale (=1)  bout de ligne ou angle de zéro (la courbe fait demi tour)"""
    Y_class[points[0,1],points[0,0]]=1
    Y_reg[points[0, 1], points[0, 0]] = 0.

    last=len(points)-1
    Y_class[points[last, 1], points[last, 0]] = 1
    Y_reg[points[last, 1], points[last, 0]] = 0.


    for i in range(1,len(points)-1):
        """ l'angle entre deux segments : cette formule donne un angle dans [0,pi[ """
        d_angle=angleBetweenTwoVector(-vectors[i],vectors[i+1])
        Y_class[points[i, 1], points[i, 0]] = angle_0_pi_ToCat(d_angle,nbAngleCat)
        Y_reg[points[i, 1], points[i, 0]] = d_angle





def addOneCurve_Y_is_width(points: list, widths:list, img: np.ndarray, Y_class: np.ndarray, Y_reg: np.ndarray):

    for i in range(1,len(points)):
        pointA = points[i - 1]
        pointB = points[i]
        yy, xx, vals = weighted_line(pointA[0], pointA[1], pointB[0], pointB[1], widths[i])

        img[xx, yy] *= 1 - vals

        yy, xx, vals = weighted_line(pointA[0], pointA[1], pointB[0], pointB[1], 0)

        if widths[i]==0: cat=1
        else: cat=2
        Y_class[xx, yy] = cat
        Y_reg=Y_class




def batch_of_lines_Y_is_orientation(nbAngleCat:int, img_size:int, batchSize:int,nbLinesPerImg = 4):

    imgs = np.ones([batchSize, img_size, img_size], dtype=np.float32)
    Ys_class = np.zeros([batchSize, img_size, img_size], dtype=np.uint8)
    Ys_reg = np.zeros([batchSize, img_size, img_size], dtype=np.float32)
    Ys_reg[:]=np.nan

    for b in range(batchSize):


        for j in range(nbLinesPerImg):
            KO=True
            while KO:
                point0 = (np.random.randint(1, img_size - 1), np.random.randint(1, img_size - 1))
                point1 = (np.random.randint(1, img_size - 1), np.random.randint(1, img_size - 1))
                KO= (point0==point1)
                if not KO : addOneLine(point0=point0, point1=point1, img=imgs[b], Y_class=Ys_class[b], Y_reg=Ys_reg[b] , nbAngleCat=nbAngleCat)


    return np.expand_dims(imgs,3),Ys_class,Ys_reg






def batch_of_curve_Y_is_angle(nbAngleCat:int, img_size:int, batchSize:int):

    imgs = np.ones([batchSize, img_size, img_size], dtype=np.float32)
    Ys_class = np.zeros([batchSize, img_size, img_size], dtype=np.uint8)
    Ys_reg = np.zeros([batchSize, img_size, img_size], dtype=np.float32)
    Ys_reg[:]=np.nan

    for b in range(batchSize):


        while True:

            step=3
            points=[]

            point=np.array([5,5])

            while 1<=point[0]<img_size-1 and 1<=point[1]<img_size-1:

                points.append(point)

                delta0=np.random.choice([-step,0,2*step])
                if delta0<0 : delta1=np.random.choice([step,2*step])
                elif delta0==0: delta1 = np.random.choice([ step])
                else : delta1 = np.random.choice([-step,0, step])

                point=point+np.array([delta0,delta1])


            if len(points)>=4 :
                addOneCurve_Y_is_angle(points=points, img=imgs[b], Y_class=Ys_class[b], Y_reg=Ys_reg[b], nbAngleCat=nbAngleCat)
                break


    return np.expand_dims(imgs,3),Ys_class,Ys_reg






def batch_of_curve_Y_is_width(img_size:int, batchSize:int):

    imgs = np.ones([batchSize, img_size, img_size], dtype=np.float32)
    Ys_class = np.zeros([batchSize, img_size, img_size], dtype=np.uint8)
    Ys_reg = np.zeros([batchSize, img_size, img_size], dtype=np.float32)
    Ys_reg[:]=np.nan


    for b in range(batchSize):

        while True:

            step=4
            points=[]
            widths=[]

            point=np.array([5,5])
            width=0

            while 1<=point[0]<img_size-1 and 1<=point[1]<img_size-1:

                points.append(point)
                widths.append(width)

                delta0=np.random.choice([-step,0,2*step])
                if delta0<0 : delta1=np.random.choice([step,2*step])
                elif delta0==0: delta1 = np.random.choice([ step])
                else : delta1 = np.random.choice([-step,0, step])

                point=point+np.array([delta0,delta1])

                width=np.random.choice([0,3])

                # if width==0: width=1
                # elif width<maxWidth : width+=np.random.choice([-1,1])
                # else :width-=1


            if len(points)>=4 :
                addOneCurve_Y_is_width(points=points, widths=widths,img=imgs[b], Y_class=Ys_class[b], Y_reg=Ys_reg[b])
                break


    return np.expand_dims(imgs,3),Ys_class,Ys_reg






def computeFreqOfClasses():
    nbAngleCat = 4
    nbCat=nbAngleCat+1
    cat_freq=np.zeros(nbCat)

    for i in range(50):
        imgs, Y_class, _ = batch_of_lines_Y_is_orientation(nbAngleCat=4, img_size=28, batchSize=2)
        for j in range(nbCat):
            cat_freq[i]=np.sum(Y_class[0,:]==j)

    print(cat_freq)
    print(1/cat_freq)





def test():

    imgsToPlot=[]
    gtToPlot=[]
    img_size=20


    def oneMethod(imgs, Y_class,Y_reg):

        print("imgs.shape",imgs.shape)
        print("Y_class.shape", Y_class.shape)

        imgsToPlot.append(imgs[0,:,:,0])
        imgsToPlot.append(imgs[1, :, :, 0])

        print(Y_class)
        print(Y_reg)

        gtToPlot.append(Y_class[0])
        gtToPlot.append(Y_class[1])


    for i in range(2):
        imgs, Y_class, Y_reg = batch_of_lines_Y_is_orientation(nbAngleCat=4, img_size=img_size, batchSize=2)
        oneMethod(imgs, Y_class,Y_reg)



    for i in range(2):
        imgs, Y_class, Y_reg = batch_of_curve_Y_is_angle(nbAngleCat=4, img_size=img_size, batchSize=2)
        oneMethod(imgs, Y_class,Y_reg)



    for i in range(4):
        imgs, Y_class, Y_reg = batch_of_curve_Y_is_width(img_size=img_size, batchSize=2)
        oneMethod(imgs, Y_class,Y_reg)





    plt.figure()
    for i in range(len(imgsToPlot)):
        plt.subplot(4,4,i+1)
        plt.imshow(imgsToPlot[i],cmap="gray")

    plt.figure()
    for i in range(len(gtToPlot)):
        plt.subplot(4,4,i+1)
        plt.imshow(gtToPlot[i],cmap="jet")

    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)


    plt.show()







if __name__=="__main__":

    test()
















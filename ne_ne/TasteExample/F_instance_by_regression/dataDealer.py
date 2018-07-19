import numpy as np
import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=50000)




def addOneRect(either__any_disjoint_separated:int, pointA, pointB, img:np.ndarray, Y_reg:np.ndarray, Y_cat:np.ndarray):


    if either__any_disjoint_separated==0:
        go=True
    elif either__any_disjoint_separated==0:
        part = img[pointA[0]:pointB[0] + 1, pointA[1]:pointB[1] + 1]
        go = np.sum(part) == 0

    else :
        partEnlarged= img[pointA[0] - 1:pointB[0] + 2, pointA[1] - 1:pointB[1] + 2]
        go=np.sum(partEnlarged)==0

    if go:
        """les nouvelles cellules passent 'au dessus' des anciennes """
        img[pointA[0]:pointB[0] + 1, pointA[1]:pointB[1]]=1

        contour=[]
        for i in range(pointA[0],pointB[0]+1):
            contour.append((i,pointA[1]))
            contour.append((i, pointB[1]))
            img[i,pointA[1]]=1.2
            img[i, pointB[1]]=1.2


        for j in range(pointA[1]+1,pointB[1]):
            contour.append((pointA[0],j))
            contour.append((pointB[0], j))
            img[pointA[0],j]=1.2
            img[pointB[0], j]=1.2


        Y_cat[pointA[0]:pointB[0] + 1, pointA[1]:pointB[1]]=1
        for (i,j) in contour:
            Y_cat[i,j]=2




        """
        Tentative d'ajout du center pour voir si la distance au centre s'apprenait mieux
        center = (pointA + pointB) / 2
        img[int(center[0]),int(center[1])]=2
        img[int(center[0])+1,int(center[1])]=2
        img[int(center[0]),int(center[1])+1]=2
        img[int(center[0])+1,int(center[1])+1]=2
        """




        contour=np.array(contour)
        corner=np.array([pointA,[pointA[0],pointB[1]],[pointB[0],pointA[1]],pointB])


        for i in range(pointA[0],pointB[0]+1):
            for j in range(pointA[1],pointB[1]+1):
                point=np.array([i,j])

                # la distance au centre c'est pas terrible  Y_reg[i,j,0]=np.sqrt(np.sum((point-center)**2))
                """ min_i   sum_j (point[j]-contour[i,j])**2 """
                Y_reg[i, j, 1] = np.min( np.sqrt( np.sum((point-contour)**2,axis=1) ))
                Y_reg[i, j, 2] = np.min( np.sqrt( np.sum((point - corner) ** 2, axis=1)))

        # part=Y_reg[pointA[0]:pointB[0] + 1, pointA[1]:pointB[1],0]
        # part-=np.max(part)



        return True
    else:
        return False




def oneBatch_of_rect_stratify(either__any_disjoint_separated:int,img_size:int, batchSize:int, nbRectPerImg:int, deltaRange:tuple):

    Xs = np.zeros([batchSize, img_size, img_size,1], dtype=np.float32)
    Ys_reg = np.zeros([batchSize, img_size, img_size,2], dtype=np.float32)
    Ys_cat = np.zeros([batchSize, img_size, img_size], dtype=np.int32)




    for b in range(batchSize):

        for j in range(nbRectPerImg):
            OK=False
            count=0

            while not OK:
                count+=1
                point0 = np.array((np.random.randint(1, img_size - 2), np.random.randint(1, img_size - 2)))
                delta=np.array((np.random.randint(deltaRange[0],deltaRange[1]), np.random.randint(deltaRange[0],deltaRange[1])))
                point1 = point0+delta
                OK=   0<=point1[0]<img_size and 0<=point1[1]<img_size

                if OK :
                    """ dans j+1, le +1, c'est à cause de la catégorie 0.
                     Renvoie false s'il n'arrive pas à mettre le rectangle de manière disjointe"""
                    OK=addOneRect(either__any_disjoint_separated,pointA=point0, pointB=point1, img=Xs[b,:,:,0], Y_reg=Ys_reg[b], Y_cat=Ys_cat[b])
                if count>10000:
                    raise Exception("pas possible de mettre autant de carré de manière disjointe")



    #
    # for i in range(img_size):
    #     for j in range(img_size):
    #         Xs[:, i, j, 1] =i/28
    #         Xs[:, i, j, 2] =j/28



    noise_gauss=np.random.normal(loc=0,scale=0.2,size=[batchSize, img_size, img_size,1])
    #noise_salt= (np.random.random(size=[batchSize, img_size, img_size,1])>0.2)
    Xs+=noise_gauss
    #Xs[noise_salt]=0


    return Xs,Ys_cat,Ys_reg




if __name__=="__main__":

    Xs, Ys_cat, Ys_reg=oneBatch_of_rect_stratify(either__any_disjoint_separated=0 ,img_size=28, batchSize=16, nbRectPerImg=1, deltaRange=(10, 15))


    plt.figure()

    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(Xs[i,:,:,0],cmap="gray")

    plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(Ys_reg[i,:,:,0])

    plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(Ys_reg[i, :,:,1])



    plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(Ys_reg[i, :,:,2])



    plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(Ys_cat[i])

    plt.show()










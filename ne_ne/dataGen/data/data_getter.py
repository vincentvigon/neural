import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
import imageio
import numpy as np
np.set_printoptions(precision=2,linewidth=3000)


path_perso='/Users/vigon/GoogleDrive/permanent/python/neurones/ne_ne/dataGen/'
MAX_NB_IMG=500
IMG_SIZE=28


def get_data_oneType(nbImg:int,imgType:str):

    assert nbImg<=MAX_NB_IMG, "max "+str(MAX_NB_IMG)+" are allowed"

    imgs=np.zeros([nbImg,28,28])
    for i in range(nbImg):
        img=imageio.imread(path_perso+"data/"+imgType+"/img_"+str(i)+".png")
        """normalization (sinon on a Nan comme loss)"""
        imgs[i,:,:]=img/255

    boundingBox=np.loadtxt(path_perso+"data/"+imgType+"/aaa_labels.csv")[:nbImg]

    return imgs,boundingBox


def get_data_circlesAndSquares(nbImgPerType:int):
    imgs1, bounding1 = get_data_oneType(nbImgPerType, "squares")
    imgs2, bounding2 = get_data_oneType(nbImgPerType, "circles")

    labels=np.zeros(nbImgPerType*2)
    labels[nbImgPerType:]=1

    """one hot encoding"""
    labels=np.eye(2)[labels.astype(np.int32)]


    imgs=np.concatenate([imgs1,imgs2])
    boundings=np.concatenate([bounding1,bounding2])


    return imgs,labels,boundings


def get_data_superposed(nbImg:int):
    dirName="blurCirclesAndSquares"
    imgs1,_=get_data_oneType(nbImg, dirName)

    labels=np.loadtxt(path_perso+"data/"+dirName+"/aaa_labels.csv")

    labels= labels[:nbImg]

    labels[:,0]=(labels[:,0]>0).astype(np.int32)
    labels[:,1]=(labels[:,1]>0).astype(np.int32)


    return imgs1,labels



def showCircle(img,center_x,center_y,radius):
    plt.imshow(img)
    circle1 = plt.Circle((center_x,center_y), radius, color='r',fill=False)
    plt.gcf().gca().add_artist(circle1)

def showSquare(img,pos_x,pos_y,size):
    plt.imshow(img)
    plt.plot([pos_x, pos_x + size, pos_x + size, pos_x, pos_x],
             [pos_y, pos_y, pos_y + size, pos_y + size, pos_y])



"""test graphique"""
if __name__=="__main__":

    imgs1,boundings1=get_data_oneType(3,"squares")
    imgs2,boundings2=get_data_oneType(3,"circles")
    imgs3, boundings3 = get_data_oneType(3, "blurCircles")

    for i in range(9):
        plt.subplot(3,3,i+1)
        if i <3:
            showSquare(imgs1[i,:,:],*boundings1[i,:])
        elif i<6:
            showCircle(imgs2[i-3],*boundings2[i-3,:])
        else:
            showCircle(imgs3[i-6],*boundings3[i-6,:])

    plt.show()







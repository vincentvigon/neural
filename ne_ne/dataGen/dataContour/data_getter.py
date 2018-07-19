import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
import imageio
import numpy as np
np.set_printoptions(precision=2,linewidth=3000)


MAX_NB_IMG=500
IMG_SIZE=28


location='/Users/vigon/GoogleDrive/permanent/python/neurones/ne_ne/dataGen/dataContour/'

def get_one_data_withGT(nbImg:int, imgType:str):

    assert nbImg<=MAX_NB_IMG, "max "+str(MAX_NB_IMG)+" are allowed"

    imgs=np.zeros([nbImg,28,28])
    for i in range(nbImg):
        img=imageio.imread(location+imgType+"/img_"+str(i)+".png")
        """normalization (sinon on a vite Nan comme loss)"""
        imgs[i,:,:]=img/255


    gts = np.zeros([nbImg, 28, 28])
    for i in range(nbImg):
        gt = imageio.imread(location+imgType + "/gt_" + str(i) + ".png")
        """normalization (sinon on a vite Nan comme loss)"""
        gts[i, :, :] = gt / 255

    """ img=0 => gts=1 
        img=1 => gts=0
    """
    return np.expand_dims(imgs,3),1-np.round(gts).astype(np.int32)



def get_both_data_withGT(nbImgPerType, indicateCat2:bool):
    imgs0, gts0 = get_one_data_withGT(nbImgPerType, "blurCircles")
    imgs1, gts1 = get_one_data_withGT(nbImgPerType, "rectangles")
    if indicateCat2: gts1*=2




    imgs=np.concatenate([imgs0,imgs1])
    gts=np.concatenate([gts0,gts1])

    shuffle=np.random.permutation(len(imgs))
    imgs=imgs[shuffle]
    gts=gts[shuffle]

    return imgs,gts



def proportionOf1():
    _, Y = get_one_data_withGT(100, "blurCircles")

    print(np.sum((Y == 1)))
    print(np.sum((Y == 0)))



def showImgs():
    #imgs, gts = get_one_data_withGT(10, "blurCircles")

    X, Y = get_both_data_withGT(100,False)

    print(X[10, :, :, 0])
    print(Y[10, :, :])

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        if i % 2 == 0:
            plt.imshow(X[i // 2, :, :, 0],vmax=2)
        else:
            plt.imshow(Y[i // 2, :, :],vmax=2)

    plt.colorbar()
    plt.show()


if __name__=="__main__":
    showImgs()
















import os
import imageio
import numpy as np
import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=50000)
import scipy.ndimage as ndi

from ne_ne2.VISU import draw16imgs

np.set_printoptions(linewidth=50000,precision=2)
from sklearn.feature_extraction import image

np.random.seed(10000)



def oneImgModif(img, imgs):

    imgs.append(img)
    angles = np.linspace(5, 360, 10)

    for angle in angles:
        mod_img = ndi.interpolation.rotate(img, angle=angle, reshape=False,order=1)
        imgs.append(mod_img)



def createInitialTensor(deb:int, end:int):

    Xs=[]
    Ys=[]

    for i,file in enumerate(os.listdir("data/NormalizedPNG")):
        if deb<=i<end:
            print(file)
            img = imageio.imread("data/NormalizedPNG/"+file)
            img=ndi.zoom(img,zoom=(0.5,0.5,1),order=1)
            Xs.append(img)


    for i, file in enumerate(os.listdir("data/GroundTruthInside")):
        if deb <= i < end:
            print(file)
            label = imageio.imread("data/GroundTruthInside/" + file)
            label=ndi.zoom(label,zoom=(0.5,0.5),order=1)
            Ys.append(label)


    Xs=np.array(Xs)
    Ys=np.array(Ys)
    Ys=(Ys>200).astype(np.uint8)
    Ys+=1


    padding=4
    s=Xs.shape
    Xs_pad=np.zeros(shape=(s[0],s[1]+2*padding,s[2]+2*padding,s[3]),dtype=np.uint8)
    Xs_pad[:,padding:s[1]+padding,padding:s[2]+padding,:]=Xs


    s=Ys.shape
    Ys_pad = np.zeros(shape=(s[0], s[1] + 2 * padding, s[2] + 2 * padding),dtype=np.uint8)
    Ys_pad[:, padding:s[1] + padding, padding:s[2] + padding] = Ys



    if True:
        print(Ys[0,:30,:30])
        print(Xs[0,:30,:30,0])

        print(Xs_pad[0,:30,:30,0])
        print(Ys_pad[0,:30,:30])

    print("Xs_pad.shape",Xs_pad.shape)
    print("Ys_pad.shape",Ys_pad.shape)

    return Xs_pad,Ys_pad



def createAugmentedTensor(deb:int, end:int):
    Xs,Ys=createInitialTensor(deb,end)

    Xss=[]
    Yss=[]
    for X in Xs:
        oneImgModif(X,Xss)
    for Y in Ys:
        oneImgModif(Y,Yss)

    Xss=np.array(Xss)
    Yss=np.array(Yss)

    print("Xss.shape",Xss.shape)
    print("Yss.shape",Yss.shape)

    np.save("data/X_aug.npy",Xss)
    np.save("data/Y_aug.npy",Yss)

    #for test
    return Xss,Yss




def multiply(imgs):
    batch=[]
    for img in imgs:
        patches_imgs = image.extract_patches_2d(img, (100, 100),max_patches=16,random_state=1234)
        batch.append(patches_imgs)

    return batch






def preprocessingX(X):
    return X.astype(np.float32) / 255


def preprocessingY(Y, nbCat):
    return np.eye(nbCat, dtype=np.float32)[Y]


def oneEpoch(vignetteSize:int,preprocessing:bool):

    Xs=np.load("data/X_aug.npy")
    Ys=np.load("data/Y_aug.npy")
    perm=np.random.permutation(len(Xs))
    Xs=Xs[perm]
    Ys=Ys[perm]


    """on les prend 16 par 16 dans chaque grande image. Elles ne sont donc pas complètement mélangées"""
    for X,Y in zip(Xs,Ys):
        seed=np.random.randint(10000)
        xs = image.extract_patches_2d(X, (vignetteSize, vignetteSize), max_patches=16,random_state=seed)
        ys = image.extract_patches_2d(Y, (vignetteSize, vignetteSize), max_patches=16,random_state=seed)
        for x,y in zip(xs,ys):
            if preprocessing:
                x=preprocessingX(x)
                y=preprocessingY(y,3)
            yield x,y



def tastetRot():
    X = np.arange(100).reshape((10, 10))

    X_rot = ndi.interpolation.rotate(X, angle=45, reshape=False)

    X_zoom = ndi.zoom(X, zoom=2)
    print(X_rot.shape)
    print(X_zoom.shape)

    plt.imshow(X_rot)




if __name__=="__main__":



    xs=[]
    ys=[]
    for x,y in oneEpoch(50,False):
        xs.append(x)
        ys.append(y)

    xs=np.array(xs)
    ys=np.array(ys)

    print(xs.shape)
    print(ys.shape)
    decal=60
    draw16imgs(xs[decal:],cmap="gray")
    draw16imgs(np.expand_dims(ys[decal:],3),vmin=0,vmax=3)
    plt.show()









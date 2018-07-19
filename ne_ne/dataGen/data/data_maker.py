import matplotlib
matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import matplotlib.pyplot as plt
import imageio
import numpy as np
np.set_printoptions(precision=2,linewidth=3000)

np.random.seed(1234132)
path_perso='/Users/vigon/GoogleDrive/permanent/python/dataSignal/TP_synthese/'
MAX_NB_IMG=500
IMG_SIZE=28


"""une fonction utilitaire pour sauver une image couleur"""
def saveRGB(r:np.ndarray,g:np.ndarray,b:np.ndarray,fileName)->None:
    s = r.shape
    couleur = np.zeros((s[0], s[1], 3), dtype=np.uint8)
    couleur[:, :, 0] = r
    couleur[:, :, 1] = g
    couleur[:, :, 2] = b
    imageio.imwrite("out/"+fileName, couleur)


def addSquareVignette(size,position,img):
    """ inverser 0 et 1 pour avoir x et y dans l'ordre """

    #img[position[1]:min(position[1]+size[1],img.shape[1]),position[0]:min(position[0]+size[0],img.shape[0])]*=0.5
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



def addCircleVignette(size, position, img):

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

    totalSize = IMG_SIZE

    label=[]
    for i in range(MAX_NB_IMG):
        img = np.ones([totalSize, totalSize])

        nbVignette=0
        while nbVignette<1:
            center=(np.random.randint(0,totalSize),np.random.randint(0,totalSize))
            radius=np.random.randint(4,10)

            pos=(center[0]-radius,center[1]-radius)
            sizeVignette=2*radius

            if 0<=pos[0]<img.shape[0]-sizeVignette and 0<=pos[1]<img.shape[1]-sizeVignette :
                if typeImg=="blurCircles":
                    addBlurCircleVignette(sizeVignette, pos, img)
                    label.append([center[0], center[1], radius])
                elif typeImg=="squares" :
                    addSquareVignette((sizeVignette,sizeVignette),pos,img)
                    label.append([pos[0], pos[1], sizeVignette])
                elif typeImg=="circles" :
                    addCircleVignette(sizeVignette, pos, img)
                    label.append([center[0], center[1], radius])


                nbVignette+=1

        imageio.imwrite("data/"+typeImg+"/img_"+str(i)+".png",img)
        np.savetxt("data/"+typeImg+"/aaa_labels.csv",label,fmt="%0.2f",header="pos_x, pos_y, size")




def createImgSeveralVignette():

    totalSize = IMG_SIZE

    label=[]
    for i in range(MAX_NB_IMG):
        img = np.ones([totalSize, totalSize])

        nbVignette=0
        wantedNbVignette=np.random.randint(1,4)

        labelLine=[]
        nbSquares=0
        nbBlurCircles=0
        while nbVignette<wantedNbVignette:
            center=(np.random.randint(0,totalSize),np.random.randint(0,totalSize))
            radius=np.random.randint(4,10)

            pos=(center[0]-radius,center[1]-radius)
            sizeVignette=2*radius

            coin=np.random.randint(2)

            if 0<=pos[0]<img.shape[0]-sizeVignette and 0<=pos[1]<img.shape[1]-sizeVignette :
                nbVignette += 1
                if coin==0:
                    addBlurCircleVignette(sizeVignette, pos, img)
                    typeVig = "blurCircle"
                    nbBlurCircles+=1
                else  :
                    addSquareVignette((sizeVignette,sizeVignette),pos,img)
                    typeVig = "square"
                    nbSquares+=1

                #labelLine.append([typeVig,pos[0], pos[1], sizeVignette])


        label.append([nbSquares,nbBlurCircles])

        dirName="blurCirclesAndSquares"
        imageio.imwrite("data/"+dirName+"/img_"+str(i)+".png",img)
        np.savetxt("data/"+dirName+"/aaa_labels.csv",label,fmt="%d",header="nb squares, nb blur circles")


def createImgNumerousSmallVignette():

    totalSize = IMG_SIZE

    label=[]
    for i in range(MAX_NB_IMG):
        img = np.ones([totalSize, totalSize])

        nbVignette=0
        wantedNbVignette=np.random.randint(4,15)

        prop = np.random.rand()*0.8+0.2
        #coin = np.random.randint(0, 2)

        while nbVignette<wantedNbVignette:

            center=(np.random.randint(0,totalSize),np.random.randint(0,totalSize))
            radius=np.random.randint(1,5)

            pos=(center[0]-radius,center[1]-radius)
            sizeVignette=2*radius


            s_x = sizeVignette
            s_y = int(sizeVignette * prop)
            """ici un bug, les rect verticaux ne fonctionnent pas"""
            # if coin==1:
            #     ss_x,ss_y=s_y,s_x
            #     s_x,s_y=ss_x,ss_y
            #     prop=1/prop

            if 0<=pos[0]<img.shape[0]-s_x and 0<=pos[1]<img.shape[1]-s_y :
                nbVignette += 1
                addSquareVignette([s_x,s_y], pos, img)


        label.append([nbVignette,prop])

        dirName="numerous"
        imageio.imwrite("data/"+dirName+"/img_"+str(i)+".png",img)
        np.savetxt("data/"+dirName+"/aaa_labels.csv",label,fmt="%d %.2f",header="nb rectangle, aspect")



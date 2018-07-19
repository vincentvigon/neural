import matplotlib

matplotlib.use('TkAgg') #truc bizarre à rajouter spécifique à mac+virtualenv
import imageio
import numpy as np
np.set_printoptions(precision=2,linewidth=3000)

np.random.seed(1234132)
#path_perso='/Users/vigon/GoogleDrive/permanent/python/dataSignal/TP_synthese/'
MAX_NB_IMG=500
IMG_SIZE=28




def addSquareVignette(sizes, cornerPosition, img, groundTrueImg):
    """ inverser 0 et 1 pour avoir x et y dans l'ordre """
    deb_x=cornerPosition[1]
    end_x=min(cornerPosition[1] + sizes[1], img.shape[1])
    deb_y=cornerPosition[0]
    end_y=min(cornerPosition[0] + sizes[0], img.shape[0])

    img[deb_x:end_x,deb_y:end_y]*=0
    groundTrueImg[deb_x,deb_y:end_y]*=0
    groundTrueImg[end_x-1,deb_y:end_y]*=0
    groundTrueImg[deb_x:end_x,deb_y]*=0
    groundTrueImg[deb_x:end_x,end_y-1]*=0






def addBlurCircleVignette(diameter, center, img):

    cornerPosition=(center[0]-diameter//2,center[1]-diameter//2)

    assert diameter + cornerPosition[0] < img.shape[0] and diameter + cornerPosition[1] < img.shape[1], "pas les bonnes tailles"

    rangex=2
    x=np.linspace(-rangex, rangex, diameter)
    y=np.linspace(-rangex, rangex, diameter)

    xx,yy=np.meshgrid(x,y)

    circle=(1-np.exp(-xx**2-yy**2))

    """ inverser 0 et 1 pour avoir x et y dans l'ordre """
    img[cornerPosition[1]:cornerPosition[1] + diameter, cornerPosition[0]:cornerPosition[0] + diameter]*=circle




def addCircleVignette(diameter, center, img):

    cornerPosition=(center[0]-diameter//2,center[1]-diameter//2)
    assert diameter + cornerPosition[0] < img.shape[0] and diameter + cornerPosition[1] < img.shape[1], "pas les bonnes tailles"

    rangex=1.1
    x=np.linspace(-rangex, rangex, diameter)
    y=np.linspace(-rangex, rangex, diameter)

    xx,yy=np.meshgrid(x,y)

    circle=(xx**2+yy**2)<1
    circle=circle.astype(np.float)
    circle=1-circle

    """ inverser 0 et 1 pour avoir x et y dans l'ordre """
    img[cornerPosition[1]:cornerPosition[1] + diameter, cornerPosition[0]:cornerPosition[0] + diameter]*=circle





def createImgOneVignette(typeImg:str):

    totalSize = IMG_SIZE
    for i in range(MAX_NB_IMG):
        img = np.ones([totalSize, totalSize])
        groundTruthImg = np.ones([totalSize, totalSize])

        nbVignette=0
        while nbVignette<1:

            center=(np.random.randint(0,totalSize),np.random.randint(0,totalSize))
            sizes=(2*np.random.randint(4,10),2*np.random.randint(4,10))


            if typeImg=="rectangles":
                cornerPos = (center[0] - sizes[0], center[1] - sizes[1])
                if 0 <= cornerPos[0] < img.shape[0] - sizes[0] and 0 <= cornerPos[1] < img.shape[1] - sizes[1]:
                    addSquareVignette(sizes,cornerPos,img,groundTruthImg)
                    nbVignette += 1

            elif typeImg=="blurCircles":
                diameter = sizes[0]
                cornerPos = (center[0] - diameter//2, center[1] - diameter//2)

                if 0 <= cornerPos[0] < img.shape[0] - diameter and 0 <= cornerPos[1] < img.shape[1] - diameter:
                    addBlurCircleVignette(diameter,center,img)
                    addCircleVignette(diameter//2,center,groundTruthImg)
                    nbVignette += 1


        imageio.imwrite(typeImg+"/img_"+str(i)+".png",img)
        imageio.imwrite(typeImg+"/gt_"+str(i)+".png",groundTruthImg)






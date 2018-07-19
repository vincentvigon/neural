import numpy as np
import matplotlib.pyplot as plt

from ne_ne.dataGen.dataLines.weighted_line import weighted_line
from ne_ne2.VISU import draw16imgs


def dataGenerator_croix_sur_ile(batch_size, patch_size):

    while True:

        a=np.linspace(0,1,patch_size,dtype=np.float32)
        aa,bb=np.meshgrid(a,a)

        nu_a=np.random.uniform(0,10,size=batch_size)
        nu_b=np.random.uniform(0,10,size=batch_size)


        y=np.sin(nu_a[:,np.newaxis,np.newaxis]*aa[np.newaxis,:,:])*np.sin(nu_b[:,np.newaxis,np.newaxis]*bb[np.newaxis,:,:])
        x=np.copy(y)

        for i in range(batch_size):
            trou_a=np.random.randint(0,patch_size)
            trou_b=np.random.randint(0,patch_size)
            x[i,trou_a,:]=0
            x[i, :, trou_b] = 0

        yield x[:,:,:,np.newaxis],y[:,:,:,np.newaxis]


def dataGenerator_carre_sur_ile(batch_size, patch_size):

    while True:

        a=np.linspace(0,1,patch_size,dtype=np.float32)
        aa,bb=np.meshgrid(a,a)

        nu_a=np.random.uniform(0,10,size=batch_size)
        nu_b=np.random.uniform(0,10,size=batch_size)


        y=np.sin(nu_a[:,np.newaxis,np.newaxis]*aa[np.newaxis,:,:])*np.sin(nu_b[:,np.newaxis,np.newaxis]*bb[np.newaxis,:,:])
        x=np.copy(y)


        for i in range(batch_size):
            trou_a=np.random.randint(0,patch_size//2)
            trou_b=np.random.randint(0,patch_size//2)

            end_a=np.random.randint(trou_a+1,patch_size)
            end_b=np.random.randint(trou_b+1,patch_size)


            part=x[i,trou_a:end_a,trou_b:end_b]


            part[:,:]=np.mean(part)

        yield x[:,:,:,np.newaxis],y[:,:,:,np.newaxis]





if __name__=="__main__":

    gen=dataGenerator_carre_sur_ile(batch_size=30, patch_size=32)

    for x,y in gen:
        print(y.shape)

        draw16imgs(x[:,:,:,0],addColorbar=False)

        #plt.imshow(y[:,:,0])
        break

    plt.show()
























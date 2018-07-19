import numpy as np




def dataGenerator(batch_size,patch_size):

    while True:
        X=np.zeros(shape=[batch_size,patch_size,patch_size,patch_size,1],dtype=np.float32)

        for i in range(batch_size):
            (a,b,c)=np.random.randint(0,batch_size,size=3)
            X[i,:a,:b,:c]=1.

        yield X,X













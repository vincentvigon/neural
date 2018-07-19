import tensorflow as tf
import numpy as np
np.set_printoptions(linewidth=20000)


DEFAULT_INIT='glorot_uniform'



class Dense:

    def __init__(self,filters:int,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=DEFAULT_INIT,
                 bias_initializer='zeros',
                 name="dense"):

        self.filters=filters
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.name=name



    def __call__(self,X:tf.Tensor):

        X_shape=X.get_shape().as_list()
        assert len(X_shape)==2 or len(X_shape)==4, "input of a Dense layer must be a 2D or 4D tensor"

        if len(X_shape)==4:
            size=X_shape[1]*X_shape[2]*X_shape[3]
            Y=tf.reshape(X,shape=[-1,size])
        else :
            size=X_shape[1]
            Y=X

        kernel_shape=(size,self.filters)
        W = matrix_var(shape=kernel_shape,type=self.kernel_initializer, dtype=X.dtype, name="W_" + self.name)

        self.res=tf.matmul(Y,W)


        if self.use_bias:
            self.res += bias_var(shape=(self.filters,), type=self.bias_initializer, dtype=X.dtype,
                                 name="B_" + self.name)

        if self.activation is not None:
            self.res=activation(self.res,self.activation,self.name)



        return self.res


class Conv2D:

    def __init__(self, filters:int,
                 kernel_size:int,
                 strides=(1, 1),
                 padding='valid' , #because in keras default is 'valid'
                 dilation_rate=(1, 1), #TODO
                 activation=None,
                 use_bias=True,
                 kernel_initializer=DEFAULT_INIT,
                 bias_initializer='zeros',
                 name="conv2d"):

        self.filters=filters
        self.kernel_size=kernel_size
        self.strides = strides
        self.padding = padding.upper() #because upper in tf
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.name=name



    def __call__(self,X:tf.Tensor):

        """ (batch,h,w,channels) """

        input_shape=X.get_shape().as_list()

        W=kernel_var(shape=(self.kernel_size,self.kernel_size,input_shape[3],self.filters),type=self.kernel_initializer,dtype=X.dtype,name="W_"+self.name)

        self.res = tf.nn.conv2d(X, W, strides=[1, self.strides[0], self.strides[1], 1], padding=self.padding)


        if self.use_bias:
            self.res+=bias_var(shape=(self.filters,),type=self.bias_initializer,dtype=X.dtype,name="B_"+self.name)


        if self.activation is not None:
            self.res=activation(self.res,self.activation,name=self.name)

        return self.res


    def shape(self):
        return self.res.shape




class MaxPooling2D:


    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',name="pool"):
        self.name=name

        if strides is None:
            strides = pool_size

        self.pool_size = pool_size

        if strides is None: self.strides=pool_size
        else : self.strides = strides

        self.padding = padding.upper()



    def __call__(self,X:tf.Tensor):
        return tf.nn.max_pool(X, ksize=[1, self.pool_size[0], self.pool_size[1], 1], strides=[1, self.strides[0], self.strides[1], 1], padding=self.padding,name=self.name)



#TODO
class UpSampling2D:

    """x = tf.image.resize_nearest_neighbor(x, (2*H,2*W))"""

    def __init__(self,size=(2, 2)):
        self.size=size

    def __call__(self,X:tf.Tensor):
        X_shape=X.get_shape().as_list()
        return tf.image.resize_nearest_neighbor(X, (self.size[0] * X_shape[1], self.size[1] * X_shape[2]))





class UpConv2D:

    def __init__(self,filters:int,
                 up_factor=2,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=DEFAULT_INIT,
                 bias_initializer='zeros',
                 name:str=""):


        self.up_factor = up_factor
        self.filters=filters
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.name=name



    def __call__(self,X:tf.Tensor):

        X_shape=X.get_shape().as_list()
        """ attention, la shape de up_convolution est inversé: [ker_w,ker_h,ker_ch_out,ker_ch_in] """
        kernel_shape=(2*self.up_factor, 2*self.up_factor,self.filters,X_shape[3])

        if self.kernel_initializer=='bilinear':
            W = tf.Variable(initial_value=self.get_bilinear_initial_tensor(kernel_shape, self.up_factor), name='W_'+self.name)
        else:
            W = kernel_var(shape=kernel_shape,
                           type=self.kernel_initializer, dtype=X.dtype, name="W_" + self.name)


        self.res=self.up_convolution(X,W,self.up_factor,self.up_factor)


        if self.use_bias:
            self.res += bias_var(shape=(self.filters,), type=self.bias_initializer, dtype=X.dtype,
                                 name="B_" + self.name)

        if self.activation is not None:
            self.res=activation(self.res,self.activation,self.name)



        return self.res




    def up_convolution(self,X, W, stride_h, stride_w):

        X_shape=X.get_shape().as_list()
        W_shape=W.get_shape().as_list()


        nb_input_chanel = W_shape[3]
        nb_output_chanel = W_shape[2]


        assert X_shape[3] == nb_input_chanel, "X and W are not compatible"

        input_size_h = X_shape[1]
        input_size_w = X_shape[2]

        output_size_h = input_size_h * stride_h
        output_size_w = input_size_w * stride_w

        output_shape = tf.stack([X_shape[0], output_size_h, output_size_w, nb_output_chanel])
        upconv = tf.nn.conv2d_transpose(X, W, output_shape, [1, stride_h, stride_w, 1])


        # """tf.shape(input)[0] c'est le batch-size, qui n'est déterminée qu'au moment du sess.run.
        #   Du coup c'est un tf.shape et pas tf.get_shape """
        #
        # output_shape = tf.stack([tf.shape(X)[0], output_size_h, output_size_w, nb_output_chanel])
        #
        # upconv = tf.nn.conv2d_transpose(X, W, output_shape, [1, stride_h, stride_w, 1])
        #
        # # Now output.get_shape() is equal (?,?,?,?) which can become a problem in the
        # # next layers. This can be repaired by reshaping the tensor to its shape:
        # output = tf.reshape(upconv, output_shape)
        # # now the shape is back to (?, H, W, C)

        return upconv

    """
    Une bonne  manière d'initialiser le filtre de convolution-transposée : on utilise les noyaux bilinéaires
    C'est les noyaux classiques que l'on utilise pour dilater une image
    d'après http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/"""

    def get_bilinear_initial_tensor(self,filter_shape, upscale_factor):

        assert filter_shape[0] == filter_shape[1], "only square filters are produced here"

        kernel_size = filter_shape[1]
        """point culminant """
        if kernel_size % 2 == 1:
            centre_location = upscale_factor - 1
        else:
            centre_location = upscale_factor - 0.5

        bilinear = np.zeros([filter_shape[0], filter_shape[1]], dtype=np.float32)
        for x in range(filter_shape[0]):
            for y in range(filter_shape[1]):
                """calcul de l'interpolation"""
                value = (1 - abs((x - centre_location) / upscale_factor)) * (
                            1 - abs((y - centre_location) / upscale_factor))
                bilinear[x, y] = value

        """on recopie pour tous les chanel"""
        weights = np.zeros(filter_shape, dtype=np.float32)
        for i in range(filter_shape[2]):
            for j in range(filter_shape[3]):
                """j'ai divisé par nb_in-chanel (dans la référence cité ci-dessus, ils n'initialisent que la diagonale, bizarre."""
                weights[:, :, i, j] = bilinear / filter_shape[2]

        return weights



#TODO
class Dropout:

    def __init__(self,rate:float,name=""):
        self.rate=rate
        self.name=name


    def __call__(self,X):
        return X




class Concatenate:

    def __init__(self,axis:int,name="concat"):
        self.axis=axis
        self.name=name

    def __call__(self,Xs:list):
        return tf.concat(Xs,axis=self.axis,name=self.name)



def matrix_var(shape:tuple,type:str,dtype,name:str):

    if type=="glorot_uniform":
        fan_in=shape[0]
        fan_out=shape[1]
        initial = (2.*tf.random_uniform(shape=shape,dtype=dtype,name=name)-1)*tf.sqrt(6 / (fan_in + fan_out))

    elif type=="glorot_normal":
        fan_in=shape[0]
        fan_out=shape[1]
        initial = tf.truncated_normal(stddev=tf.sqrt(2 / (fan_in + fan_out)), shape=shape, dtype=dtype, name=name)


    elif type == "simple":
        initial = tf.truncated_normal(stddev=0.2, shape=shape, dtype=dtype, name=name)

    else : raise ValueError("'"+type+"' not yet implemented")


    return tf.Variable(initial_value=initial, name=name)




def kernel_var(shape:tuple,type:str,dtype,name:str):


    if type=="glorot_uniform":
        fan_in_out=(shape[2]+shape[3])*shape[0]*shape[1]
        initial = (2.*tf.random_uniform(shape=shape,dtype=dtype,name=name)-1)*np.sqrt(6 / fan_in_out)


    elif type=="glorot_normal":
        fan_in_out=(shape[2]+shape[3])*shape[0]*shape[1]
        initial = tf.truncated_normal(stddev=tf.sqrt(2 / fan_in_out ), shape=shape, dtype=dtype, name=name)


    elif type=="simple":
        initial=tf.truncated_normal(stddev=0.02,shape=shape,dtype=dtype,name=name)

    else : raise ValueError("'"+type+"' not yet implemented")

    return tf.Variable(initial_value=initial, name=name)






def bias_var(shape:tuple,type:str,dtype,name:str):

    if type == 'zeros':
        initial = tf.constant(0.,shape=shape, dtype=dtype)
        return tf.Variable(initial_value=initial, name=name)


    raise ValueError("'" + type + "' not yet implemented")




def activation(X:tf.Tensor,type:str,name:str):
    if type == 'relu':
        return tf.nn.relu(X,name=name)
    else:
        raise ValueError("activation: '" + type + "' not implemented")






def taste():


    _X=np.zeros(shape=[1,20,20,1])
    for i in range(10):
        _X[0,2*i,:,0]=1

    _X[0,:,5,0]=2

    print(_X[0, :30, :30, 0])

    _X=tf.constant(_X)
    _Y=UpSampling2D()(_X)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        Y=sess.run(_Y)
        print(Y.shape)
        print(Y[0,:30,:30,0])


if __name__=="__main__":
    #taste()

    print(list(range(5-1,-1,-1)))







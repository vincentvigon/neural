import tensorflow as tf
import numpy as np




""" Autre problème : souvent la dim-0 du tenseur d'entrée n'est pas connue car cela correspond à la taille du batch.
Cette petite fonction règle le problème.
"""
def up_convolution(X, W, stride_h, stride_w):

        nb_input_chanel=W.get_shape().as_list()[3]
        nb_output_chanel=W.get_shape().as_list()[2]

        assert X.get_shape().as_list()[3]==nb_input_chanel, "X and W are not compatible"

        input_size_h = X.get_shape().as_list()[1]
        input_size_w = X.get_shape().as_list()[2]

        output_size_h=input_size_h*stride_h
        output_size_w=input_size_w*stride_w


        """tf.shape(input)[0] c'est le batch-size, qui n'est déterminée qu'au moment du sess.run. 
          Du coup c'est un tf.shape et pas tf.get_shape """
        output_shape = tf.stack([tf.shape(X)[0],output_size_h, output_size_w,nb_output_chanel])

        upconv = tf.nn.conv2d_transpose(X, W, output_shape, [1, stride_h, stride_w, 1])

        # Now output.get_shape() is equal (?,?,?,?) which can become a problem in the
        # next layers. This can be repaired by reshaping the tensor to its shape:
        output = tf.reshape(upconv, output_shape)
        # now the shape is back to (?, H, W, C)

        return output



"""
Une bonne  manière d'initialiser le filtre de convolution-transposée : on utilise les noyaux bilinéaires
C'est les noyaux classiques que l'on utilise pour dilater une image
d'après http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/"""
def get_bilinear_initial_tensor(filter_shape, upscale_factor):

    assert filter_shape[0]==filter_shape[1], "only square filters are produced here"

    kernel_size = filter_shape[1]
    """point culminant """
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]],dtype=np.float32)
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            """calcul de l'interpolation"""
            value = (1 - abs((x - centre_location) / upscale_factor)) * (1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value

    """on recopie pour tous les chanel"""
    weights = np.zeros(filter_shape,dtype=np.float32)
    for i in range(filter_shape[2]):
        for j in range(filter_shape[3]):
            """j'ai divisé par nb_in-chanel (dans la référence cité ci-dessus, ils n'initialisent que la diagonale, bizarre."""
            weights[:, :, i, j] = bilinear/filter_shape[2]


    return weights



""" ici :https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
ils indique que faire un clip sur Y_hay n'est pas une bonne idée car cela arrète la propagation du gradient... 
J'ai tester  :  un clip  "grossier"  ex : tf.clip_by_value(Y_hat, 0.0001, 0.9999) : cela tue vraiment l'apprentissage.
Un clip plus fin ne change pas grand chose, mais il faudrait faire des tests plus poussé.
   """
def crossEntropy(Y, Y_hat):
    return - tf.reduce_mean(Y*tf.log(Y_hat+1e-10))


""" d'après https://stackoverflow.com/questions/33712178/tensorflow-nan-bug.
 Mais c'est vraiment trop long (4 fois plus long d'après le test ci-dessous)
 """
def crossEntropy_tooSlow(x, y, axis=None):
  safe_y = tf.where(tf.equal(x, 0.), tf.ones_like(y), y)
  return -tf.reduce_mean(x * tf.log(safe_y), axis=axis)



""" Parfois mon modèle peut prédire des proba>1 : c'est mal. 
  Il faut pénaliser """
def crossEntropy_forProbaGreaterThatOne(Y, Y_hat):
    return - tf.reduce_mean(Y*tf.log(Y_hat+1e-10))


def quadraticLoss(Y, Y_hat):
    return tf.reduce_mean((Y-Y_hat)**2)



"""
Attention ici au fameux  1+1e-10 -Y avec Y=1  ---> 0  et quand on passe au log ---> nan
"""
def crossEntropy_multiLabel(Y,Y_hat):
    return - tf.reduce_mean(  Y * tf.log(Y_hat+1e-10) + (1 - Y) * tf.log( (1 - Y_hat)+1e-10))



def accuracy(true_Y_proba, hat_Y_proba):
    return tf.reduce_mean(tf.cast(tf.equal(hat_Y_proba, true_Y_proba), tf.float32))

#
# def make_binary_with_arg_max(Y_hat,nb_category):
#     return tf.one_hot(tf.arg_max(Y_hat, dimension=1), nb_category )
#
# def make_binary_with_threshold(Y_hat,threshold):
#     return tf.cast(tf.greater(Y_hat, threshold), tf.float32)



def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_strided(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)



def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)




""" attention, stddev=0.1 est très important. 
Par exemple, sur le modèle bounding box avec les données squareCircle, cela ne converge pas avec stddev=0.02.
Une initialisation standard est stddev=sqrt(2/nbInput). Mais avec nbInput =1024 comme dans leNet, cela fait 0.05: très petit. 
"""
def weight_variable(shape, stddev=0.1, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial_value=initial,name=name)



"""Dans le cours de Stanfort, Karpathy indique que cela serait mieux en prenant des biais nul. """
def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial_value=initial,name=name)




def leaky_relu(x, alpha=0.0, name=""):
    return tf.maximum(alpha * x, x, name)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)


def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5):
    """
    Code taken from http://stackoverflow.com/a/34634291/2267819
    """
    with tf.variable_scope(scope):
        beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                               , trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, 0.02),
                                trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed





def summarizeW_asImage(W):
    mat=stick_imgs(W)
    mat_shape=mat.get_shape().as_list()
    mat=tf.reshape(mat,shape=[1,mat_shape[0],mat_shape[1],1])
    tf.summary.image(W.name,mat, max_outputs=1)



#TODO ça ne marche pas
def summarize_batch_images_NB(X, title:str, min_batch_size:int):
    """
    :param X: images, shape=[batch_size,w,h]
    :param title:
    :param min_batch_size:
    """
    if min_batch_size >= 25:
        nn = 5
    elif min_batch_size >= 16:
        nn = 4
    elif min_batch_size >= 4:
        nn = 2
    else:
        nn = 1

    trans = tf.transpose(X[:nn*nn], perm=[1, 2, 0])
    trans = tf.reshape(trans, shape=[28, 28, nn, nn])
    mat = stick_imgs(trans)
    tf.summary.image(title, mat, max_outputs=1)


import matplotlib
import matplotlib.cm

def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```

    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = tf.constant(cm.colors, dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value










def stick_imgs(W, nbChannel2=10, nbChannel3=10):
    """

    :param W: de shape [w,h,nb in channel, nb out channel]
    :param nbChannel2:
    :param nbChannel3:
    :return:
    """

    W_shape = W.get_shape().as_list()


    nb2=min(nbChannel2, W_shape[2])
    nb3=min(nbChannel3, W_shape[3])

    sep=tf.constant(1.,shape=(1,W_shape[1]))
    noImg=tf.constant(1.,shape=(W_shape[0],W_shape[1]))
    columns=[]
    for j in range(nbChannel3):
        Ws=[]
        for i in range(nbChannel2):
            if i<nb2 and j<nb3: Ws.append(W[:,:,i,j])
            else: Ws.append(noImg)
            Ws.append(sep)
        column=tf.concat(Ws,0)
        shape_Ligne = column.get_shape().as_list()
        sep_column=tf.constant(1.,shape=[shape_Ligne[0],1])
        columns.append(column)
        columns.append(sep_column)
    mat=tf.concat(columns,1)

    return mat









import time


def test_compare_crossEntropy():
    tf.InteractiveSession()
    y = np.random.normal([100, 100])
    y_hat = np.random.normal([100, 100])

    _y = tf.constant(y)
    _y_hat = tf.constant(y_hat)

    nb = 300

    begin = time.time()
    for i in range(nb):
        crossEntropy(_y, _y_hat).eval()

    print("adding bias", time.time() - begin)
    # 20 seconds

    begin = time.time()
    for i in range(nb):
        crossEntropy_tooSlow(_y, _y_hat).eval()

    print("with where", time.time() - begin)
    # 68 seconds




if __name__=="__main__":

    pass















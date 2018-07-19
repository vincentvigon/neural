import tensorflow as tf
import ne_ne.INGREDIENTS as ing
import numpy as np


"""Le bas du réseau de Yann Le Cun. 
Les images d'entrées sont de 28*28 pixels. La structure :

        [Conv5*5 > relu > max_pool2*2]  *2    

TODO : faire la même chose * 3, * 4 en fonction de la taille des images et de leur complexité. 

TODO : la mode c'est plutôt de faire des : 
        [Conv3*3 > relu > Conv3*3 > relu> max_pool2*2] * n

"""
class LeNet_bottom:

    def __init__(self,X:tf.Tensor,nbChannels:int):

        self.nbChannels=nbChannels
        nbSummaryOutput=4

        """"""
        ''' couche de convolution 1'''
        with tf.variable_scope("conv1"):
            W_conv1 = ing.weight_variable([5, 5, self.nbChannels, 32],name="W")
            b_conv1 = ing.bias_variable([32],name="b")

            self.filtred1=tf.nn.relu(ing.conv2d_basic(X,W_conv1,b_conv1))
            """ shape=(?,14*14,nbChannels)  """
            self.pool1 =ing.max_pool_2x2(self.filtred1)

            ing.summarizeW_asImage(W_conv1)
            tf.summary.image("filtred", self.filtred1[:, :, :, 0:1], max_outputs=nbSummaryOutput)




        ''' couche de convolution 2'''
        with tf.variable_scope("conv2"):

            W_conv2 = ing.weight_variable([5, 5, 32, 64],name="W")
            b_conv2 = ing.bias_variable([64],name="b")

            self.filtred2=tf.nn.relu(ing.conv2d_basic(self.pool1, W_conv2, b_conv2))
            """ shape=(?,7*7,nbChannels)  """
            self.pool2 =ing.max_pool_2x2(self.filtred2)

            ing.summarizeW_asImage(W_conv2)
            tf.summary.image("filtred",self.filtred2[:,:,:,0:1],max_outputs=12)




        """un alias pour la sortie"""
        self.Y=self.pool2







"""le bas de VGG, à charger à partir d'un fichier de donnée.
Je ne l'ai pas encore tester. 
"""
def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image

    for i, name in enumerate(layers):
        kind = name[:4]

        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = get_variable(bias.reshape(-1), name=name + "_b")
            current = ing.conv2d_basic(current, kernels, bias)

        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)

        elif kind == 'pool':
            current = ing.avg_pool_2x2(current)
        net[name] = current

    return net



"""pas très utile"""
def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var

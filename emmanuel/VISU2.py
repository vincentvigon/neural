import tensorflow as tf
import matplotlib
import matplotlib.cm
import matplotlib
import matplotlib.pyplot as plt




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

    assert len(value.shape)==3, "in colorize: tensor must be 4D"


    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    print(vmin,vmax)
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # squeeze last dim if it exists
    #value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = tf.constant(cm.colors, dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value




import numpy as np
def draw16imgs(imgs, mainTitle=None, vmin=None, vmax=None, cmap="jet",subTitles=None,addColorbar=False,file_name:str=None):

    assert len(imgs.shape)==4 or len(imgs.shape)==3, "imgs must be a 3D or 4D tensor"

    if len(imgs.shape)==3: imgs=np.expand_dims(imgs,3)


    for k in range(imgs.shape[3]):

        fig=plt.figure()
        if mainTitle is not None:  fig.suptitle(mainTitle)

        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(imgs[i,:,:,k],cmap=cmap,vmin=vmin,vmax=vmax)
            if subTitles is not None:
                plt.title(subTitles[i])
                """les x_ticks cachent les subTitles"""
                plt.tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom='off',  # ticks along the bottom edge are off
                    top='off',  # ticks along the top edge are off
                    labelbottom='off')  # labels along the bottom edge are off


        if addColorbar:
            plt.subplots_adjust(bottom=0.1,top=0.9, left=0.1, right=0.8)
            cax = plt.axes([0.85, 0.1, 0.075, 0.8])
            plt.colorbar(cax=cax)

        if file_name is not None:
            fig.savefig(file_name)

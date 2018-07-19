import keras
from keras.optimizers import *
from keras.callbacks import *
import matplotlib.pyplot as plt
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from emmanuel.A_generateData import dataGenerator_croix_sur_ile, dataGenerator_carre_sur_ile
from emmanuel.VISU2 import draw16imgs
from emmanuel.models_Keras import get_keras_model
import glob

dir_du_moment = "out/test_callback"
patch_size = 32


if not os.path.exists(dir_du_moment):
    os.makedirs(dir_du_moment)
    os.makedirs(dir_du_moment + "/weights")
    os.makedirs(dir_du_moment + "/fig")
    os.makedirs(dir_du_moment + "/log")


def quadratic_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=[-1])
    y_pred = tf.reshape(y_pred, shape=[-1])
    return tf.reduce_mean((y_true - y_pred) ** 2)


def absolute_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=[-1])
    y_pred = tf.reshape(y_pred, shape=[-1])
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def make_model():
    model = get_keras_model(patch_size=patch_size, dropout=1)
    model.compile(optimizer=Adam(lr=0.001), loss=quadratic_loss, metrics=[absolute_loss])
    return model


def load_model(starting_weights):
    model = keras.models.load_model(starting_weights,
                                    custom_objects={"quadratic_loss": quadratic_loss, "absolute_loss": absolute_loss})
    return model


def load_model_ensemble():
    print('loaded weights from:'+dir_du_moment + "/weights/")

    names = dir_du_moment + "/weights/*"
    models_path = glob.glob(names)
    dice = []
    for paths in models_path:
        dice.append(paths[-5:-3])
    models_path_dict = dict(zip(models_path, dice))
    models_path_sorted = sorted(models_path_dict, key=models_path_dict.__getitem__)

    weights = []
    model = None
    for i in range(1, 7):
        model = load_model(models_path_sorted[-i])
        weights.append(model.get_weights())

    weights_mean = np.mean(weights, axis=0)
    print('weights shape: ', weights_mean.shape)

    model.set_weights(weights_mean)
    return model


def train(model, startingTime):
    batch_size = 50
    generator_train = dataGenerator_carre_sur_ile(patch_size=patch_size, batch_size=batch_size)
    generator_test = dataGenerator_carre_sur_ile(patch_size=patch_size, batch_size=batch_size)

    print('Fitting model...')

    try:
        model.fit_generator(
            generator=generator_train,
            steps_per_epoch=4,
            epochs=20,
            callbacks=callbacks(startingTime, batch_size, dir_du_moment),
            validation_data=generator_test,
            validation_steps=1
        )

    except  KeyboardInterrupt:
        print("on a stoppé")

    """test final"""
    test(model, startingTime)


def test(model, startingTime):
    batch_size = 16

    x_test, y_test = None, None

    for x_test, y_test in dataGenerator_carre_sur_ile(patch_size=patch_size, batch_size=batch_size):
        break
    hat_y_test = model.predict(x_test, batch_size=batch_size)

    print("x_test.shape", x_test.shape)

    draw16imgs(hat_y_test[:, :, :, 0], vmin=-1, vmax=1, addColorbar=False,
               file_name=dir_du_moment + "/fig/%d-hat_y_test.png" % startingTime)
    draw16imgs(x_test[:, :, :, 0], vmin=-1, vmax=1, addColorbar=False,
               file_name=dir_du_moment + "/fig/%d-x_test.png" % startingTime)
    draw16imgs(y_test[:, :, :, 0], vmin=-1, vmax=1, addColorbar=False,
               file_name=dir_du_moment + "/fig/%d-y_test.png" % startingTime)

    plt.show()


#
# def test_fromSaved(path):
#     startingTime=time.time()
#     model = keras.models.load_model(path, custom_objects={"quadratic_loss": quadratic_loss})
#     model.summary()
#     test(model,startingTime)


""" CALLBACKS """


class Histories(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses=[]

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        # y_pred = self.model.predict(self.model.validation_data[0])
        plt.plot(self.losses)
        plt.plot(self.val_losses,'.')

        test(self.model, time.time())
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        return


""" pour trouver un bon learning rate de démarrage """


def lr_finder(epoch):
    lr1 = 1e-6 * (0.5 ** (-epoch))  # 30 epochs
    print(lr1)
    return lr1


""" pour trouver des minimums locaux assez distincts(pour ensuite utiliser la moyenne des poids)"""


def lr_cycle(epoch, lr_max=0.001, lr_min=0.0001, step_size=15):
    step = epoch % (2 * step_size)
    if step < step_size:
        lr = lr_max - step * (lr_max - lr_min) / step_size
    else:
        lr = lr_min + (step - step_size) * (lr_max - lr_min) / step_size
    return lr


def callbacks(startingTime, batch_size, dir_du_moment):
    """ tensorboard ne fonction plus """
    path = dir_du_moment + "/log/%d" % startingTime
    print("tensorboard at:\n" + path)
    tensorboard = keras.callbacks.TensorBoard(log_dir=path,
                                              batch_size=batch_size, write_images=True)

    return [
        Histories(),

        ModelCheckpoint(
            filepath=dir_du_moment + '/weights/%d-Vnet3couches-{epoch:02d}-{val_loss:.2f}.h5' % startingTime,
            monitor='loss',  # ou monitor='val_loss'
            verbose=0,
            save_best_only=True),
        CSVLogger(path + '.csv')
        # LearningRateScheduler(lr)
    ]


if __name__ == '__main__':

    starting_time = time.time()
    # model=make_model()
    # model=load_model("out/test_callback/weights/1532002385-Vnet3couches-01-3.99.h5")
    model = load_model_ensemble()
    train(model, starting_time)


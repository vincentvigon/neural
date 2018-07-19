import tensorflow as tf
import os
import sys
import numpy as np
import keras
from keras import metrics
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as keras
from keras import backend as K

from perdro.A_generateData import dataGenerator
from perdro.metrics_losses import sensitivity, specificity, dice_coef_loss


#
# sys.path.append('../data_manipulation')
# sys.path.append('../models')
# sys.path.append('../utils')
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from perdro.models_Keras import Model_Vnet3D_keras


def train_generator_patch():


    patch_size=8

    model = Model_Vnet3D_keras(patch_size,1)

    metric = [sensitivity, specificity]

    model.compile(optimizer=Adam(lr=0.001), loss=dice_coef_loss, metrics=metric)

    generator_train = dataGenerator(patch_size=patch_size, batch_size=20)
    generator_test = dataGenerator(patch_size=patch_size, batch_size=10)

    print('Fitting model...')
    model.fit_generator(
        generator=generator_train,
        steps_per_epoch=10,
        epochs=1,
        #callbacks=callbacks(output_path, batch_size, 0),
        validation_data=generator_test,
        validation_steps=5
    )


#
# def fine_tunning(num_epoch, steps, model_path, output_path, patch_size=64, batch_size=1):
#     # load_model
#     model = load_model('../../my_weights/' + model_path + '/64deepVal.h5',
#                        custom_objects={'dice_coef_loss': dice_coef_loss,
#                                        'sensitivity': sensitivity, 'specificity': specificity})
#
#     model.summary()
#     '''
#     #freeze layers until 'conv3d_19'
#     set_trainable = False
#     for layer in model.layers:
#         if layer.name == 'conv3d_19':
#             set_trainable = True
#         if set_trainable:
#             layer.trainable = True
#         else:
#             layer.trainable = False
#     model.summary()
#     print('Fine tunning...')
#     for layer in model.layers:
#         print(layer, layer.trainable)
#     '''
#     mydata = dataProcess(patch_len=patch_size, batch_size=batch_size)
#
#     history = model.fit_generator(generator=mydata.loadImages('train'), steps_per_epoch=steps, epochs=num_epoch,
#                                   callbacks=callbacks(output_path, batch_size, 1),
#                                   validation_data=mydata.loadImages('val'),
#                                   validation_steps=2,
#                                   use_multiprocessing=False)


## Learning Rate Functions
def lr_fine_tunning(epoch):
    return lr_exp(epoch, 0.99, 0.0005, 0)


def lr_scheduler(epoch):
    lr_max = 0.001
    lr_min = 0.0001
    step_size = 15
    gamma = 0.98
    change = 41 * step_size
    if epoch < change:
        lr = lr_cycle(epoch, lr_max, lr_min, step_size)
    else:
        lr = lr_exp(epoch, gamma, lr_min, change)
    print('Learning rate: ', lr)
    return lr


def lr_exp(epoch, gamma, init, change):
    return init * (gamma ** (epoch - change))


def lr_cycle(epoch, lr_max, lr_min, step_size):
    step = epoch % (2 * step_size)
    if step < step_size:
        lr = lr_max - step * (lr_max - lr_min) / step_size
    else:
        lr = lr_min + (step - step_size) * (lr_max - lr_min) / step_size
    return lr


def lr_finder(epoch):
    lr1 = 1e-6 * (0.5 ** (-epoch))  # 30 epochs
    print(lr1)
    return lr1



## callbacks
def callbacks(folder_name, batch_size, iD):
    if iD == 0:
        lr = lr_scheduler
    else:
        lr = lr_fine_tunning
    if not os.path.exists('../../my_weights/' + folder_name):
        os.makedirs('../../my_weights/' + folder_name)

    return [
        TensorBoard(
            log_dir='../../my_logs/' + folder_name + '/',
            batch_size=batch_size,
            # histogram_freq = 1
        ),

        ModelCheckpoint(filepath='../../my_weights/' + folder_name + '/64deepVal-{epoch:02d}-{val_loss:.2f}.h5',
                        monitor='val_loss',
                        verbose=0,
                        save_best_only=True),
        ModelCheckpoint(filepath='../../my_weights/' + folder_name + '/64deep-{epoch:02d}-{loss:.2f}.h5',
                        monitor='loss',
                        verbose=0,
                        save_best_only=True),
        CSVLogger('../../my_csv/' + folder_name + '.csv'),
        LearningRateScheduler(lr)
    ]


if __name__ == '__main__':
    train_generator_patch()

# train_generator_image(patch_size = 64,batch_size = 1,num_epoch=500,steps=100,output_path='KerasFullFcnDeep',net='fcn')

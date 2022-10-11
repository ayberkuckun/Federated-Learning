import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

import numpy as np


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def fit_model(federated_train_data, federated_test_data, model, model_name, model_type, model_no, dataset_name):
    # Prepare model saving directory.
    save_dir = os.path.join(os.getcwd(), f'saved_models/{dataset_name}/{model_type}/{model_no}')
    model_file_name = '%s_model.h5' % model_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_file_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   monitor='val_loss',
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=15)

    callbacks = [checkpoint, lr_reducer, early_stopping]
    # callbacks = [checkpoint, lr_reducer, lr_scheduler]

    model.fit(federated_train_data,
              validation_data=federated_test_data,
              callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate(federated_test_data, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    return model

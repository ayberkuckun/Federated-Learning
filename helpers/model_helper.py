import os

from keras import Model
from keras.layers import Dropout, Dense, BatchNormalization, Flatten
from tensorflow import keras
import tensorflow as tf
from helpers import training_helper
import resnet_for_cifar


# def prepare_models(federated_train_data, federated_test_data, dataset, model, surrogate_no, reference_no):
#     assert surrogate_no != 0 and reference_no != 0
#
#     # Input image dimensions.
#     input_shape = federated_train_data[0]["image"].shape[1:]
#
#     if dataset == 'cifar100':
#         num_classes = 100
#     else:
#         raise NotImplementedError
#
#     path_source = os.path.join(os.getcwd(), f'saved_models/{dataset}/source/0/{model}_model.h5')
#     try:
#         source_model = keras.models.load_model(path_source)
#     except:
#         print("No source model has found.")
#         # Create source model
#         source_model = create_model(model, num_classes, input_shape)
#
#         # Fit source model
#         print(f"Training source model")
#         training_helper.fit_model(federated_train_data, federated_test_data,
#                                   model=source_model,
#                                   model_name=model,
#                                   model_type='source',
#                                   model_no=0,
#                                   dataset_name=dataset)
#
#     reference_models = []
#     for i in range(reference_no):
#         path_reference = os.path.join(os.getcwd(), f'saved_models/{dataset}/reference/{i + 1}/{model}_model.h5')
#         try:
#             reference_model = keras.models.load_model(path_reference)
#         except:
#             print(f"Reference model no: {i} is missing.")
#             # Create reference model
#             reference_model = create_model(model, num_classes, input_shape)
#
#             # Fit reference model
#             print(f"Training reference No: {i}")
#             training_helper.fit_model(federated_train_data, federated_test_data,
#                                       model=reference_model,
#                                       model_name=model,
#                                       model_type='reference',
#                                       model_no=i,
#                                       dataset_name=dataset)
#
#         reference_models.append(reference_model)
#
#     # Get labels for surrogate models
#     y_predict_train_source = source_model.predict(x_train)
#     y_predict_test_source = source_model.predict(x_test)
#
#     surrogate_models = []
#     for i in range(surrogate_no):
#         path_surrogate = os.path.join(os.getcwd(), f'saved_models/{dataset}/surrogate/{i + 1}/{model}_model.h5')
#         try:
#             surrogate_model = keras.models.load_model(path_surrogate)
#         except:
#             print(f"Surrogate model no: {i} is missing.")
#             # Create surrogate models
#             surrogate_model = create_model(model, num_classes, input_shape)
#
#             # Fit reference model
#             print(f"Training surrogate No: {i}")
#             training_helper.fit_model(x_train, y_predict_train_source, x_test, y_predict_test_source,
#                                       model=surrogate_model,
#                                       model_name=model,
#                                       model_type='surrogate',
#                                       model_no=i,
#                                       dataset_name=dataset)
#
#         surrogate_models.append(surrogate_model)
#
#     return source_model, reference_models, surrogate_models


def create_model(model_name, num_classes, input_shape):
    if model_name == "ResNet50v2":
        model = tf.keras.applications.ResNet50(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
            classes=num_classes
        )
    elif model_name == "MobileNetV2":
        model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None,
            classes=num_classes
        )
    elif model_name == "DenseNet121":
        model = tf.keras.applications.DenseNet121(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=input_shape,
            pooling="max",
            classes=num_classes
        )
        # add top layers
        out = model.output
        out = Flatten()(out)
        out = BatchNormalization()(out)
        out = Dense(256, activation='relu')(out)
        out = Dropout(0.3)(out)
        out = BatchNormalization()(out)
        out = Dense(128, activation='relu')(out)
        out = Dropout(0.3)(out)
        out = BatchNormalization()(out)
        out = Dense(64, activation='relu')(out)
        out = Dropout(0.3)(out)
        out = Dense(num_classes, activation='softmax')(out)

        model = Model(inputs=model.inputs, outputs=out)

    elif model_name == "ResNet20v1":
        model = create_resnet_model(input_shape=input_shape, num_classes=num_classes)
    else:
        raise Exception("Model is not defined.")

    return model


def create_resnet_model(input_shape, num_classes=10):
    # Computed depth from supplied model parameter n
    depth = 20
    model = resnet_for_cifar.resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)

    return model


def save_model(model, dataset_name, model_name, model_type, model_no):
    # Prepare model saving directory.
    save_dir = os.path.join(os.getcwd(), f'saved_models/{dataset_name}/{model_type}/{model_no}')
    model_file_name = '%s_model2.h5' % model_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_file_name)

    model.save(filepath)  # , include_optimizer=False

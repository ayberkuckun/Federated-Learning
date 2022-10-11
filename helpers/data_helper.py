import collections
import os

import numpy as np
import tensorflow_federated as tff
from helpers import constant_helper
import tensorflow as tf
from tensorflow import keras

# Training Variables
epochs_train = 5
batch_size_train = 8


def prepare_for_inference(federated_dataset, training=True, seed=None):
    if training:
        return [dataset.shuffle(len(list(dataset)), seed=seed, reshuffle_each_iteration=True).repeat(
            epochs_train).batch(batch_size_train).prefetch(tf.data.AUTOTUNE) for dataset in federated_dataset]
    else:
        return [dataset.batch(batch_size_train).prefetch(tf.data.AUTOTUNE) for dataset in federated_dataset]


def _preprocessing_fn(dataset, num_classes, train_mean):
    def _batch_format_fn(element):
        element["image"] = tf.cast(element["image"], tf.float32) / 255
        element["image"] -= train_mean

        element["label"] = tf.one_hot(element["label"], num_classes)

        # return collections.OrderedDict(
        #     x=element['image'],
        #     y=element['label']
        # )

        return element['image'], element['label']

    return dataset.map(_batch_format_fn, num_parallel_calls=tf.data.AUTOTUNE)


def _add_images(old_state, input_element):
    new_state = tf.add(old_state, tf.cast(input_element["image"], tf.float32) / 255)

    return new_state


def preprocess_data(dataset='cifar100', num_train_clients=None, num_test_clients=None):
    try:
        print("Trying to load processed data.")

        if dataset == 'cifar100':
            if not num_train_clients:
                num_train_clients = 500
            if not num_test_clients:
                num_test_clients = 100

            federated_train_data = []
            for i in range(num_train_clients):
                path = os.path.join(os.getcwd(), f'processed_data/{dataset}/train/')
                file_name = f'client-{i}'
                federated_train_data.append(tf.data.experimental.load(path + file_name))

            federated_test_data = []
            for i in range(num_test_clients):
                path = os.path.join(os.getcwd(), f'processed_data/{dataset}/test/')
                file_name = f'client-{i}'
                federated_test_data.append(tf.data.experimental.load(path + file_name))

        else:
            raise NotImplementedError
    except:
        print("Couldn't find processed data. Processing...")

        if dataset == 'cifar100':
            train_data, test_data = tff.simulation.datasets.cifar100.load_data()
            num_classes = 100

            image_mean_of_each_client = []
            federated_train_data = []
            for x in train_data.client_ids[:num_train_clients]:
                client_dataset = train_data.create_tf_dataset_for_client(x)
                image_mean_for_client = client_dataset.reduce(0.0, _add_images) / len(list(client_dataset))
                image_mean_of_each_client.append(image_mean_for_client)
                federated_train_data.append(_preprocessing_fn(client_dataset, num_classes, train_mean=image_mean_for_client))

            federated_test_data = []
            for x in test_data.client_ids[:num_test_clients]:
                client_dataset = test_data.create_tf_dataset_for_client(x)
                federated_test_data.append(
                    _preprocessing_fn(client_dataset, num_classes, train_mean=np.mean(image_mean_of_each_client))
                )

        else:
            raise NotImplementedError

        for no, client_train_data in enumerate(federated_train_data):
            path = os.path.join(os.getcwd(), f'processed_data/{dataset}/train/')
            file_name = f'client-{no}'
            if not os.path.exists(path):
                os.makedirs(path)
            tf.data.experimental.save(client_train_data, path + file_name)

        for no, client_test_data in enumerate(federated_test_data):
            path = os.path.join(os.getcwd(), f'processed_data/{dataset}/test/')
            file_name = f'client-{no}'
            if not os.path.exists(path):
                os.makedirs(path)
            tf.data.experimental.save(client_test_data, path + file_name)

    return federated_train_data, federated_test_data

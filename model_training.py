import os

import keras.metrics
import numpy as np
import tensorflow_federated as tff
import tensorflow as tf
from tqdm import tqdm

from helpers import data_helper
from helpers import model_helper
from helpers import federated_helper

# todo batch norm non-trainable / check pretrained models which doesn't have batch norm.
# todo callbacks etc.
# todo checkpoint every some iteration.
# todo check other datasets for test loss comparison.

# cpu_device = tf.config.list_logical_devices('CPU')[0]
# gpu_devices = tf.config.list_logical_devices('GPU')[0]
# tff.backends.native.set_local_python_execution_context(
    # server_tf_device=cpu_device,
    # client_tf_devices=cpu_device,
    # clients_per_thread=2,
    # max_fanout=50
# )

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)


def model_train():
    dataset_name = "cifar100"
    model_name = "ResNet20v1"
    num_classes = 100
    NUM_TRAIN_CLIENTS = 500
    NUM_TEST_CLIENTS = 10

    sur_start_point = 0
    ref_start_point = 0

    # Random Selection Clients.
    CLIENTS_PER_ROUND = 10

    federated_train_data, federated_test_data = data_helper.preprocess_data(dataset=dataset_name,
                                                                            num_train_clients=NUM_TRAIN_CLIENTS,
                                                                            num_test_clients=NUM_TEST_CLIENTS)

    input_shape = next(iter(federated_train_data[0]))[0].shape
    federated_train_data = np.array(federated_train_data)

    federated_test_data_all = data_helper.prepare_for_inference(federated_test_data[:NUM_TEST_CLIENTS], training=False)

    train_source = True
    reference_no = 2
    surrogate_no = 2
    NUM_ROUNDS = 5000

    def _model_fn():
        keras_model = model_helper.create_model(model_name, num_classes, input_shape)
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=federated_test_data_all[0].element_spec,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy()])

    # Evaluation function
    evaluation = tff.learning.build_federated_evaluation(_model_fn)

    if train_source:
        # Iterative process for the source model
        iterative_process_source = tff.learning.algorithms.build_weighted_fed_avg(
            _model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.05)
        )
        state_source = iterative_process_source.initialize()

        # Create a environment to get communication cost.
        environment = federated_helper.set_sizing_environment()

        print("Start Federated Source Training...")
        for round_num in tqdm(range(NUM_ROUNDS)):
            print(f"Round: {round_num}")

            selected_clients = np.random.choice(list(range(0, NUM_TRAIN_CLIENTS)), size=CLIENTS_PER_ROUND, replace=False)
            federated_train_data_source = data_helper.prepare_for_inference(federated_train_data[selected_clients],
                                                                            training=True)

            result_source = iterative_process_source.next(state_source, federated_train_data_source)
            state_source = result_source.state
            loss_source_train = result_source.metrics['client_work']['train']['loss']
            acc_source_train = result_source.metrics['client_work']['train']['categorical_accuracy']

            weights_source = iterative_process_source.get_model_weights(state_source)
            test_metrics_source = evaluation(weights_source, federated_test_data_all)
            print(test_metrics_source)
            loss_source_test = test_metrics_source['eval']['loss']
            acc_source_test = test_metrics_source['eval']['categorical_accuracy']

            print(f"\t Source Model - Train Loss: {loss_source_train:.4f} - Train Accuracy: {acc_source_train:.2f}"
                  f" - Test Loss: {loss_source_test:.4f} - Test Accuracy: {acc_source_test:.2f}")

            # For more about size_info, please see https://www.tensorflow.org/federated/api_docs/python/tff/framework/SizeInfo
            size_info = environment.get_size_info()
            broadcasted_bits = size_info.broadcast_bits[-1]
            aggregated_bits = size_info.aggregate_bits[-1]

            print('broadcasted_bits={}, aggregated_bits={}'.format(
                federated_helper.format_size(broadcasted_bits), federated_helper.format_size(aggregated_bits)))

            # Finish when accuracy is above 85%.
            # if acc_source_test > 0.85:
            if acc_source_train > 0.85:
                break

        source_model = model_helper.create_model(model_name, num_classes, input_shape)
        weights_source = iterative_process_source.get_model_weights(state_source)
        weights_source.assign_weights_to(source_model)

        model_helper.save_model(source_model, dataset_name, model_name, "source", 0)

    else:
        path_source = os.path.join(os.getcwd(), f'saved_models/{dataset_name}/source/0/{model_name}_model.h5')
        source_model = keras.models.load_model(path_source, compile=False)

    if reference_no > 0:
        # Iterative processes for the reference models
        iterative_process_refs = []
        state_refs = []
        for i in range(reference_no):
            iterative_process_refs.append(
                tff.learning.algorithms.build_weighted_fed_avg(
                    _model_fn,
                    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=1e-3),
                    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1)
                )
            )
            state_refs.append(iterative_process_refs[i].initialize())

        print("Start Federated Reference Training...")
        for round_num in range(NUM_ROUNDS):
            print(f"Round: {round_num}")

            result_refs = []
            loss_refs_train = []
            acc_refs_train = []
            loss_refs_test = []
            acc_refs_test = []
            for i in range(reference_no):
                selected_clients = np.random.choice(list(range(0, NUM_TRAIN_CLIENTS)), size=CLIENTS_PER_ROUND, replace=False)
                federated_train_data_ref = data_helper.prepare_for_inference(federated_train_data[selected_clients], training=True)

                result_refs.append(iterative_process_refs[i].next(state_refs[i], federated_train_data_ref))
                state_refs[i] = result_refs[i].state
                loss_refs_train.append(result_refs[i].metrics['client_work']['train']['loss'])
                acc_refs_train.append(result_refs[i].metrics['client_work']['train']['categorical_accuracy'])

                weights_ref = iterative_process_refs[i].get_model_weights(state_refs[i])
                test_metrics_ref = evaluation(weights_ref, federated_test_data_all)
                loss_refs_test.append(test_metrics_ref['eval']['loss'])
                acc_refs_test.append(test_metrics_ref['eval']['categorical_accuracy'])

            for i in range(reference_no):
                print(f"\t Reference Model No: {i} - Train Loss: {loss_refs_train[i]:.4f} - Train Accuracy: {acc_refs_train[i]:.2f}"
                      f" - Test Loss: {loss_refs_test[i]:.4f} - Test Accuracy: {acc_refs_test[i]:.2f}")

            # Finish when accuracy is above 85%.
            # if all(np.array(acc_refs_test) > 0.85):
            if all(np.array(acc_refs_train) > 0.85):
                break

        reference_models = []
        for i in range(reference_no):
            reference_models.append(model_helper.create_model(model_name, num_classes, input_shape))
            weights_ref = iterative_process_refs[i].get_model_weights(state_refs[i])
            weights_ref.assign_weights_to(reference_models[i])
            model_helper.save_model(reference_models[i], dataset_name, model_name, "reference", i + ref_start_point)

    if surrogate_no > 0:
        print("Start Federated Surrogate Training.")
        # Labels for surrogate model
        surrogate_train_dataset = []
        for train_data in federated_train_data:
            x_train_source = np.array([data[0] for data in list(train_data)])
            y_predict_train_source = source_model.predict(train_data.batch(8))
            surrogate_train_dataset.append(tf.data.Dataset.from_tensor_slices((x_train_source, y_predict_train_source)))

        surrogate_train_dataset = np.array(surrogate_train_dataset)

        surrogate_test_dataset = []
        for test_data in federated_test_data[:NUM_TEST_CLIENTS]:
            x_test_source = np.array([data[0] for data in list(test_data)])
            y_predict_test_source = source_model.predict(test_data.batch(8))
            surrogate_test_dataset.append(tf.data.Dataset.from_tensor_slices((x_test_source, y_predict_test_source)))

        federated_test_data_surs = data_helper.prepare_for_inference(surrogate_test_dataset, training=False)

        # Iterative processes for the surrogate models
        iterative_process_surs = []
        state_surs = []
        for i in range(surrogate_no):
            iterative_process_surs.append(
                tff.learning.algorithms.build_weighted_fed_avg(
                    _model_fn,
                    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=1e-3),
                    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1)
                )
            )
            state_surs.append(iterative_process_surs[i].initialize())

        for round_num in range(NUM_ROUNDS):
            print(f"Round: {round_num}")

            result_surs = []
            loss_surs_train = []
            acc_surs_train = []
            loss_surs_test = []
            acc_surs_test = []
            for i in range(surrogate_no):
                selected_clients = np.random.choice(list(range(0, NUM_TRAIN_CLIENTS)), size=CLIENTS_PER_ROUND, replace=False)
                federated_train_data_sur = data_helper.prepare_for_inference(surrogate_train_dataset[selected_clients], training=True)

                result_surs.append(iterative_process_surs[i].next(state_surs[i], federated_train_data_sur))
                state_surs[i] = result_surs[i].state
                loss_surs_train.append(result_surs[i].metrics['client_work']['train']['loss'])
                acc_surs_train.append(result_surs[i].metrics['client_work']['train']['categorical_accuracy'])

                weights_sur = iterative_process_surs[i].get_model_weights(state_surs[i])
                test_metrics_sur = evaluation(weights_sur, federated_test_data_surs)
                loss_surs_test.append(test_metrics_sur['eval']['loss'])
                acc_surs_test.append(test_metrics_sur['eval']['categorical_accuracy'])

            for i in range(surrogate_no):
                print(f"\t Surrogate Model No: {i} - Train Loss: {loss_surs_train[i]:.4f} - Train Accuracy: {acc_surs_train[i]:.2f}"
                      f" - Test Loss: {loss_surs_test[i]:.4f} - Test Accuracy: {acc_surs_test[i]:.2f}")

            # Finish when accuracy is above 85%.
            # if all(np.array(acc_surs_test) > 0.85):
            if all(np.array(acc_surs_train) > 0.85):
                break

        surrogate_models = []
        for i in range(surrogate_no):
            surrogate_models.append(model_helper.create_model(model_name, num_classes, input_shape))
            weights_sur = iterative_process_surs[i].get_model_weights(state_surs[i])
            weights_sur.assign_weights_to(surrogate_models[i])
            model_helper.save_model(surrogate_models[i], dataset_name, model_name, "surrogate", i + sur_start_point)


if __name__ == "__main__":
    model_train()

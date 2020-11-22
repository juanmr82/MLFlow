"""
Simple run of a Fully Connected NN on the MNIST Database using Keras together with MLFLow
"""
__author__ = "Juan Reina"
__email__ = "pybsoft@gmail.com"
__license__ = "MIT License"
__version__ = "0.1.0"
__credits__ = ["dmatrix (https://github.com/dmatrix)"]

import sys

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from urllib.parse import urlparse
import mlflow.keras
import matplotlib.pyplot as plt
import pathlib
import os
import logging as local_logging

IMAGES_DIRECTORY_NAME = "images"
MLFLOW_EXPERIMENT_ID = "keras_learning_1"
MLFLOW_MODEL_NAME = "MNIST_NN"


def setup_logging():
    formatter = local_logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                        datefmt='%Y-%m-%d %H:%M:%S')
    screen_handler = local_logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    my_logger = local_logging.getLogger("SimpleMNISTNN")
    my_logger.setLevel(local_logging.INFO)
    my_logger.addHandler(screen_handler)
    return my_logger


def get_images_directory():
    actual_dir = pathlib.Path(__file__).parent.absolute()
    directory = os.path.join(actual_dir, IMAGES_DIRECTORY_NAME)
    if not os.path.exists(directory):
        os.mkdir(directory, mode=0o755)
    return directory


def plot_training_params(history):
    """
    This function will take a model history and create a plot with the history loss and accuracy parameters
    The plot will be save on a subdirectory called 'images' (created if it does not exist); name of the
    file is 'traning_parameters.png'
        :param history: A model training history object
        :return matplotlib Figure object
    """
    training_parameters_image = "training_parameters.png"

    acc = history.history['accuracy']
    loss = history.history['loss']
    ep = range(1, len(acc) + 1)
    figu, ax = plt.subplots()

    left_color = 'tab:blue'
    right_color = 'tab:red'

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Training Loss")
    ax.plot(ep, loss, color=left_color, marker='o')
    ax.tick_params(axis='y', labelcolor=left_color)

    ax_right = ax.twinx()

    ax_right.set_ylabel("Training Accuracy")
    ax_right.plot(ep, acc, color=right_color, marker='x')
    ax_right.tick_params(axis='y', labelcolor=right_color)

    plt.title("Training Parameters")
    plt.xlabel('Epochs')
    plt.legend()

    figu.tight_layout()

    image_dir = get_images_directory()
    plot_png = os.path.join(image_dir, training_parameters_image)

    figu.savefig(plot_png)

    return figu


def get_loss(hist):
    """
    Get the last value of the loss function of the training history
    :param hist: History object
    :return: the last loss value
    """
    loss = hist.history["loss"]
    loss_val = loss[len(loss) - 1]
    return loss_val


def get_accuracy(hist):
    """
    Get the last accuracy value of the training session of a model
    :param hist: History object
    :return: The last accuracy value
    """
    accuracy = hist.history["accuracy"]
    accuracy_val = accuracy[len(accuracy) - 1]
    return accuracy_val


if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Loading MNIST Data")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    epochs = 15
    batch_size = 128
    nr_hidden_units = 512
    learning_rate = 0.001

    logger.info(f"Building NN with {nr_hidden_units} hidden units")

    network = models.Sequential()

    network.add(layers.Dense(nr_hidden_units, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    logger.info(f"Reshaping images")

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_ID)

    logger.info(f"Starting MLFlow experiment {MLFLOW_EXPERIMENT_ID}")
    with mlflow.start_run():

        logger.info("Enabling Experimental Autolog")
        mlflow.keras.autolog()

        # Log the Loss Function, not logge by the automatic logging feature
        mlflow.log_param("loss_function", network.loss)

        logger.info("Training Model")
        fit_history = network.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)

        logger.info(f"Saving Training parameters plot under {get_images_directory()}")
        fig = plot_training_params(fit_history)

        logger.info("Evaluating Network")
        test_loss, test_acc = network.evaluate(test_images, test_labels)

        """
        If no 'autolog' would be enabled, the following will be used to log the metrics
             mlflow.log_param("epochs", epochs)
             mlflow.log_param("batch_size", batch_size)
             mlflow.log_param("hidden_units", nr_hidden_units)


             mlflow.log_metric("training_loss", get_loss(fit_history))
             mlflow.log_metric("training_accuracy", get_accuracy(fit_history))

             mlflow.log_metric("test_accuracy", test_acc)
             mlflow.log_metric("test_loss", test_loss)
        """
        track = mlflow.get_tracking_uri()

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            logger.info("Logging Model and Artifacts into MLFlow Remote Tracking server")
            mlflow.keras.log_model(network, MLFLOW_MODEL_NAME, registered_model_name="SimpleMNISTNN")
            mlflow.log_artifacts(get_images_directory())

        else:
            logger.info("Logging Model and Artifacts into MLFlow Local mlrun directory")
            mlflow.keras.log_model(network, MLFLOW_MODEL_NAME)

        logger.info("Script finished")

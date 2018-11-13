from __future__ import print_function

import argparse
import gzip
import json
import os
import pickle

import numpy as np
import tensorflow as tf

def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir="./data"):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, "mnist.pkl.gz")
    if not os.path.exists(data_file):
        print("... downloading MNIST from the web")
        try:
            import urllib

            urllib.urlretrieve("http://google.com")
        except AttributeError:
            import urllib.request as urllib
        url = "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"
        urllib.urlretrieve(url, data_file)

    print("... loading data")
    # Load the dataset
    f = gzip.open(data_file, "rb")
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype("float32")
    test_x = test_x.astype("float32").reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype("int32")
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype("float32")
    valid_x = valid_x.astype("float32").reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype("int32")
    train_x, train_y = train_set
    train_x = train_x.astype("float32").reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype("int32")
    print("... done loading data")
    return train_x, train_y, valid_x, one_hot(valid_y), test_x, one_hot(test_y)


def cnn_mnist(features, labels, mode, params):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    print("input_layer: ", input_layer.shape)
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=params["num_filters"],
        kernel_size=params["filter_size"],
        padding="same",
        activation=tf.nn.relu,
    )
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
    print("pool1: ", pool1.shape)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=params["num_filters"],
        kernel_size=params["filter_size"],
        padding="same",
        activation=tf.nn.relu,
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)
    print("pool2: ", pool2.shape)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * params["num_filters"]])
    print(pool2_flat.shape)
    dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=params["learning_rate"]
        )
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


def train_and_validate(
    x_train,
    y_train,
    x_valid,
    y_valid,
    num_epochs,
    lr,
    num_filters,
    batch_size,
    filter_size,
):
    # TODO: train and validate your convolutional neural networks with the provided data and hyperparameters
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_mnist,
        params={
            "learning_rate": lr,
            "num_filters": num_filters,
            "filter_size": filter_size,
        },
        model_dir="./model",
    )

    # Set up logging for predictions
    logging_hook = tf.train.LoggingTensorHook(
        tensors={"probabilities": "softmax_tensor"}, every_n_iter=50
    )
    learning_curve = []

    print(x_train.shape)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train}, y=y_train, batch_size=batch_size, num_epochs=None, shuffle=True
    )

    res = mnist_classifier.train(
        input_fn=train_input_fn, steps=20000, hooks=[logging_hook]
    )
    print("Training: ", res)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_valid}, y=y_valid, num_epochs=1, shuffle=False
    )

    res = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print("Validation: ", res)
    learning_curve.append(res)

    return (
        learning_curve,
        model,
    )  # TODO: Return the validation error after each epoch (i.e learning curve) and your model


def test(x_test, y_test, model):
    # TODO: test your network here by evaluating it on the test data
    return model.evaluate(x_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        default="./",
        type=str,
        nargs="?",
        help="Path where the results will be stored",
    )
    parser.add_argument(
        "--input_path",
        default="./",
        type=str,
        nargs="?",
        help="Path where the data is located. If the data is not available it will be downloaded first",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        nargs="?",
        help="Learning rate for SGD",
    )
    parser.add_argument(
        "--num_filters",
        default=32,
        type=int,
        nargs="?",
        help="The number of filters for each convolution layer",
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, nargs="?", help="Batch size for SGD"
    )
    parser.add_argument(
        "--epochs",
        default=12,
        type=int,
        nargs="?",
        help="Determines how many epochs the network will be trained",
    )
    parser.add_argument(
        "--run_id",
        default="0",
        type=str,
        nargs="?",
        help="Helps to identify different runs of an experiments",
    )
    parser.add_argument(
        "--filter_size", default=3, type=int, nargs="?", help="Filter width and height"
    )
    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs
    filter_size = args.filter_size

    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    learning_curve, model = train_and_validate(
        train_data,
        train_labels,
        eval_data,
        eval_labels,
        epochs,
        lr,
        num_filters,
        batch_size,
        filter_size,
    )

    test_error = test(x_test, y_test, model)

    print("loss: ", test_error[0])
    print("accuracy: ", test_error[1])

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["filter_size"] = filter_size
    results["learning_curve"] = learning_curve
    results["test_error"] = test_error

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%s.json" % args.run_id)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()

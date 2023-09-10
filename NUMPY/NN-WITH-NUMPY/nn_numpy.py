import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# random seed
np.random.seed(1)


def initialize_weights_and_biases(X, out_layer=1, layers=[4, 4], p=False):
    weights = []
    biases = []

    # order will follow layer 0 (input layer), weight 0, layer 1, weight 0, ...
    # wights list will have weight 0, weight 1, ...

    # initialize weights & biases randomly with mean 0
    layer_0_weights = np.random.normal(0, 1 / np.sqrt(X.shape[-1]), (X.shape[-1], layers[0]))
    layer_n_weights = np.random.normal(0, 1 / np.sqrt(layers[-1]), (layers[-1], out_layer))

    layer_0_biases = np.zeros((1, layers[0]))
    layer_n_biases = np.zeros((1, out_layer))

    # add layer 0 weights
    weights.append(layer_0_weights)
    biases.append(layer_0_biases)

    if len(layers) >= 2:
        for num_layer in range(1, len(layers)):
            # initialize weights randomly with mean 0
            layer_weights = np.random.normal(0, 1 / np.sqrt(layers[num_layer - 1]),
                                             (layers[num_layer - 1], layers[num_layer]))
            layer_biases = np.zeros((1, layers[num_layer]))
            weights.append(layer_weights)
            biases.append(layer_biases)

    weights.append(layer_n_weights)
    biases.append(layer_n_biases)

    if p:
        print("Weights Shape")

        for weight in weights:
            print(weight.shape)

        print("-------")
        print("Biases Shape")

        for biase in biases:
            print(biase.shape)

    return weights, biases


def forward_prop(X, weights, biases):
    # iterate over layers
    activations = []

    # first layer
    z1 = np.dot(X, weights[0]) + biases[0]
    activations.append(relu(z1))

    for i in range(1, len(weights) - 1):
        z = np.dot(activations[i - 1], weights[i]) + biases[i]
        activations.append(relu(z))

    # last layer
    zn = np.dot(activations[-1], weights[-1]) + biases[-1]
    activations.append(sigmoid(zn))

    return activations


def binary_cross_entropy_loss(y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def categorical_cross_entropy_loss(y, y_hat):
    # both of shape (num_data_points, num_classes).
    m = y.shape[0]
    loss = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat), axis=1)
    loss = np.mean(loss)
    return loss


def back_prop(X, y, activations, weights):
    m = X.shape[0]

    dws = []
    dbs = []

    # last layer
    dz = activations[-1] - y
    dw = np.dot(activations[-2].T, dz) / m
    db = np.sum(dz, axis=0, keepdims=True) / m
    da = np.dot(dz, weights[-1].T)
    dws.append(dw)
    dbs.append(db)

    for i in range(len(activations) - 2, -1, -1):
        dz = da * (activations[i] > 0)
        dw = np.dot(activations[i - 1].T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        da = np.dot(dz, weights[i].T)
        dws.append(dw)
        dbs.append(db)

    dws.reverse()
    dbs.reverse()

    return dws, dbs


def update_params(weights, biases, dws, dbs, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * dws[i]
        biases[i] -= learning_rate * dbs[i]

    return weights, biases


def train(X, y, weights, biases, iterations=20000, learning_rate=0.05, loss="binary", pr_int=1000):
    loss_history = []
    for i in range(iterations):
        # forward propagations
        activations = forward_prop(X, weights, biases)
        yhat = activations[-1]

        # backward propagation
        dws, dbs = back_prop(X, y, activations, weights)

        # update parameters
        weights, biases = update_params(weights, biases, dws, dbs, learning_rate)

        # compute loss
        if loss == "binary":
            iter_loss = binary_cross_entropy_loss(y, yhat)
        elif loss == "category":
            iter_loss = categorical_cross_entropy_loss(y, yhat)
        loss_history.append(iter_loss)

        # print loss every pr_int
        if i % pr_int == 0:
            print(f"Loss after {i} iterations: {iter_loss}")

    return weights, biases, loss_history


def predict(X, weights, biases):
    activations = forward_prop(X, weights, biases)
    return (activations[-1] > 0.5).astype(int)


def get_dataset():
    # Load the MNIST dataset from OpenML
    mnist = fetch_openml(name='mnist_784', version=1, data_home=None)
    o_X, o_y = mnist['data'], mnist['target']

    # Convert the target labels to integers
    o_y = o_y.astype(int)

    # Select only two classes, for example class 0 and class 1
    class_0 = np.where(o_y == 0)[0]
    class_1 = np.where(o_y == 1)[0]
    selected_classes = np.concatenate([class_0, class_1])

    X, y = np.take(o_X, selected_classes, 0), np.take(o_y, selected_classes, 0)

    # Preprocess data
    X = np.array(X) / 255

    # lets check only 1 number
    y = np.array(y).reshape(-1, 1)

    # Split the dataset into training and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    # Balance the training dataset using random under-sampling
    rus = RandomUnderSampler(random_state=0)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    y_train = y_train.reshape(-1, 1)

    return X_train, X_temp, X_val, X_test, y_val, y_test


def accuracy(y, y_hat):
    return np.mean(y == y_hat)

import numpy as np
import cv2
import NeuralNetwork
import json
import os
import matplotlib.pyplot as plt

#defining the initial parameters and the learning rate
batch_size = 10
nn_hdim = 2048
learning_rate = 0.1
f1 = "relu"
f2 = "sigmoid"
threshold = 0.0001
sd_init = 0.01
sd_init_w2 = sd_init


def make_json(W1, W2, b1, b2, id1, id2, activation1, activation2, nn_h_dim, path_to_save):
    """
    make json file with trained parameters.
    W1: numpy arrays of shape (1024, nn_h_dim)
    W2: numpy arrays of shape (nn_h_dim, 1)
    b1: numpy arrays of shape (1, nn_h_dim)
    b2: numpy arrays of shape (1, 1)
    nn_hdim - 2048
    id1: id1 - str '204214928'
    id2: id2 - str '308407907'
    activation1:  'ReLU'
    activation2:  'sigmoid'
    """
    trained_dict = {'weights': (W1.tolist(), W2.tolist()),
                    'biases': (b1.tolist(), b2.tolist()),
                    'nn_hdim': nn_h_dim,
                    'activation_1': activation1,
                    'activation_2': activation2,
                    'IDs': (id1, id2)}
    file_path = os.path.join(path_to_save, 'trained_dict_{}_{}'.format(
        trained_dict.get('IDs')[0], trained_dict.get('IDs')[1])
                             )
    with open(file_path, 'w') as f:
        json.dump(trained_dict, f, indent=4)

def load_image(prefix, number, data_vec, label_vec, is_training):
    if is_training:
        path = "data\\training\\"
    else:
        path = "data\\validation\\"

    path = path + prefix + number + ".png"
    image = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
    data_vec.append(image.flatten() / 255.0)
    if prefix == "pos_":
        label_vec.append(1)
    else:
        label_vec.append(0)


def load_data(train_data, val_data, train_label, val_label):
    # load train data
    for i in range(256):
        load_image("neg_", str(i), train_data, train_label, True)
        load_image("pos_", str(i), train_data, train_label, True)
    for i in range(256, 334):
        load_image("neg_", str(i), val_data, val_label, False)
        load_image("pos_", str(i), val_data, val_label, False)
    return np.asarray(train_data), np.asarray(val_data), np.asarray(train_label), np.asarray(val_label),


def main():
    convergence_flag = False
    previous_loss = np.inf
    counter = 0
    accuracy_per_training_epoch = 0
    loss_per_training_epoch = 0
    train_data = []
    val_data = []
    train_label = []
    val_label = []
    epoch_training_loss = []
    epoch_validation_loss= []
    epoch_training_accuracy = []
    epoch_validation_accuracy = []
    train_data, val_data, train_label, val_label = load_data(train_data, val_data, train_label, val_label)
    my_net = NeuralNetwork.NeuralNetwork(learning_rate, f1, f2, sd_init, sd_init_w2)
    epoc = 0
    my_net.forward_pass(val_data, val_label)
    my_net.calculate_accuracy(val_label)
    print("Inintial validation loss: ", my_net.loss, "Inintial accuracy: ", my_net.accuracy)
    while not convergence_flag:
        batch_count = 0
        shuffler = np.random.permutation(len(train_label))
        train_label = train_label[shuffler]
        train_data = train_data[shuffler]
        if (not epoc % 10) and (epoc != 0):
            my_net.learning_rate = my_net.learning_rate / 2
        for i in range(0, len(train_label), batch_size):
            batch = train_data[i:batch_size + i, :]
            batch_labels = train_label[i:batch_size + i]
            my_net.forward_pass(batch, batch_labels)
            my_net.calculate_accuracy(batch_labels)
            accuracy_per_training_epoch += my_net.accuracy
            loss_per_training_epoch += my_net.loss
            # print("epoc:", epoc, "batch:", batch_count, "loss:", my_net.loss, "accuracy:",
            #         my_net.accuracy, "prediction:", my_net.a2, np.round(my_net.a2).squeeze(), "real labels:", batch_labels)
            my_net.backward_pass(batch_labels)
            my_net.compute_gradient(batch)
            batch_count += 1
        accuracy_per_training_epoch = accuracy_per_training_epoch/(len(train_label)/batch_size)
        loss_per_training_epoch = loss_per_training_epoch/(len(train_label)/batch_size)
        epoch_training_accuracy.append(accuracy_per_training_epoch)
        epoch_training_loss.append(loss_per_training_epoch)
        accuracy_per_training_epoch = 0
        loss_per_training_epoch = 0
        my_net.forward_pass(val_data, val_label)
        my_net.calculate_accuracy(val_label)
        if (my_net.loss - previous_loss) <= threshold:
            counter += 1
        else:
            counter = 0
        if epoc > 100:
            convergence_flag = (counter >= 3)

        print("Validation loss: ", my_net.loss, "Accuracy:", my_net.accuracy, "learning rate:", my_net.learning_rate)
        previous_loss = my_net.loss
        epoch_validation_accuracy.append(my_net.accuracy)
        epoch_validation_loss.append(my_net.loss)
        epoc += 1

## plotting section-----------------------------------------------------------------------------------------------

    trained_dict = {
        'weights': (my_net.W1, my_net.W2),
        'biases': (my_net.b1, my_net.b2),
        'nn_hdim': 2048,
        'activation_1': 'relu',
        'activation_2': 'sigmoid',
        'IDs': (204214928, 308407907)
        }
    json_path = ''
    make_json(my_net.W1,my_net.W2,my_net.b1,my_net.b2,'204214928','308407907','relu','sigmoid',nn_hdim, json_path)
    plt.subplot(2, 1, 1)
    plt.plot(range(epoc), epoch_training_loss)
    plt.plot(range(epoc), epoch_validation_loss)
    plt.scatter(epoc, epoch_training_loss[epoc-1], marker='o')
    plt.scatter(epoc, epoch_validation_loss[epoc-1], marker='o')
    x = [epoc, epoc]
    n = [round(epoch_training_loss[epoc-1], 2), round(epoch_validation_loss[epoc-1], 2)]
    for i, txt in enumerate(n):
        plt.annotate(txt, (x[i], n[i]))
    plt.legend(["training", "validation"])
    plt.title('loss and accuracy as function of epoc number')
    plt.ylabel('loss [au]')
    plt.subplot(2, 1, 2)
    plt.plot(range(epoc), epoch_training_accuracy)
    plt.plot(range(epoc), epoch_validation_accuracy)
    plt.scatter(epoc, epoch_training_accuracy[epoc-1], marker='o')
    plt.scatter(epoc, epoch_validation_accuracy[epoc-1], marker='o')
    y = [epoc, epoc]
    s = [round(epoch_training_accuracy[epoc-1], 2), round(epoch_validation_accuracy[epoc-1], 2)]
    for i, txt in enumerate(s):
        plt.annotate(txt, (y[i], s[i]))
    plt.legend(["training", "validation"])
    plt.xlabel('epoc number')
    plt.ylabel('accuracy [%]')
    plt.show()


if __name__ == "__main__":
    main()
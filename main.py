import numpy as np
import cv2
import NeuralNetwork
import math

batch_size = 10
nn_hdim = 2048
learning_rate = 1
f1 = "sigmoid"
f2 = "sigmoid"
threshold = 0
sd_init = 0.4
sd_init_w2 = 0.4


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
    train_data = []
    val_data = []
    train_label = []
    val_label = []
    train_data, val_data, train_label, val_label = load_data(train_data, val_data, train_label, val_label)
    my_net = NeuralNetwork.NeuralNetwork(learning_rate, f1, f2, sd_init, sd_init_w2)
    epoc = 0
    while not convergence_flag:
        batch_count = 0
        for i in range(0, len(train_label), batch_size):
            if not epoc % 5:
                my_net.learning_rate = my_net.learning_rate / 2
            shuffler = np.random.permutation(len(train_label))
            train_label = train_label[shuffler]
            train_data = train_data[shuffler]
            batch = train_data[i:batch_size + i, :]
            batch_labels = train_label[i:batch_size + i]
            my_net.forward_pass(batch, batch_labels)
            my_net.calculate_accuracy(batch_labels)
            print("epoc:", epoc, "batch:", batch_count, "loss:", my_net.loss, "accuracy:",
                    my_net.accuracy, "prediction:", my_net.a2, np.round(my_net.a2).squeeze(), "real labels:", batch_labels)
            my_net.backward_pass(batch_labels)
            my_net.compute_gradient(batch)
            batch_count+=1
        my_net.forward_pass(val_data, val_label)
        if (my_net.loss - previous_loss) >= threshold:
            counter += 1
        else:
            counter = 0
        if(epoc > 100):
            convergence_flag = (counter >= 3)
        print("Validation loss: ", my_net.loss)
        previous_loss = my_net.loss
        epoc+=1




if __name__ == "__main__":
    main()

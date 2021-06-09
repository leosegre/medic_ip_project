import numpy as np
import cv2

batch_size = 1
nn_hdim = 1024



def load_image(prefix, number, data_vec, label_vec, is_training):
    if is_training:
        path = "data\\training\\"
    else:
        path = "data\\validation\\"

    path = path + prefix + number + ".png"
    image = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
    data_vec.append(image.flatten()/255.0)
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
    train_data = []
    val_data = []
    train_label = []
    val_label = []
    train_data, val_data, train_label, val_label = load_data(train_data, val_data, train_label, val_label)


if __name__ == "__main__":
    main()

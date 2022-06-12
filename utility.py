import os
import cv2
import numpy as np
import tensorflow as tf


def process_image(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (64, 64))

    return image, label


def split_data(data, labels):
    n = int(0.16 * len(labels))
    testing_data = data[:n]
    testing_labels = labels[:n]
    training_data = data[n:]
    training_labels = labels[n:]
    assert len(testing_labels) == len(testing_data)
    assert len(training_labels) == len(training_data)
    return training_data, training_labels, testing_data, testing_labels


def load(image_paths, verbose=-1):
    """
    Expects images for each class in separate directory
    (E.g - all digits in 0 class in the directory named 0).
    :param image_paths: Path to the image
    :param verbose: The number after which to inform the user.
    :return: Tuple of data and labels
    """

    data = list()  # Stores the image data
    labels = list()  # Stores the corresponding labels for the images

    # Iterate over each image path
    for (i, image_path) in enumerate(image_paths):

        # Load the image and extract the class labels
        # im_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # image = np.array(im_gray).flatten()
        # print(image_path)
        # image_path -> "./Image_Dataset/3-image9 - 2.png".
        # To extract the label, we need to split the path string on the file separator based on os.
        # Here it is / and split gives list ['.', 'Image_Dataset', '3-image9 - 2.png']
        # Access the -1 element of the list and first element of it will give the label of the image.
        label = image_path.split('/')[-1][0]
        # print(label)

        # Scale the Image to [0, 1] and add to list
        # data.append(image / 255)
        data.append(image)
        labels.append(int(label))

        # Show an update after every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))

    # Return the Data and Labels
    return split_data(data, labels)

import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # get the path to each category under data_dir
    for category in range(NUM_CATEGORIES):
        path_to_category = os.path.join(data_dir, f"{category}")

        # get the list of all files under a given category
        files = sorted(os.listdir(path_to_category))

        # read and resize image files
        for file in files:
            path_to_file = os.path.join(data_dir, f"{category}", f"{file}")
            img = cv2.imread(path_to_file)
            resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # append each resized image to "images" list and each integer of category to "labels" list
            images.append(resized)
            labels.append(category)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([

        # convolutional layer
        tf.keras.layers.Conv2D(
            56, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # maxpooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # convolutional layer
        tf.keras.layers.Conv2D(
            28, (3, 3), activation="relu", input_shape=(IMG_WIDTH-2, IMG_HEIGHT-2, 3)
        ),

        # flatten units
        tf.keras.layers.Flatten(),

        # add 1 hidden layer with dropout
        tf.keras.layers.Dense(676, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()

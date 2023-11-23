import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from PIL import Image
import os
import pandas as pd


def load_images(folder, image_rows, image_cols, image_channels):

    image_list = []
    files = os.listdir(folder)

    for filename in files:
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            file_path = os.path.join(folder, filename)
            try:
                # Open the image using PIL
                img = Image.open(file_path)
                img = img.resize((image_cols, image_rows))  # Resizing the image
                img = np.array(img)  # Image -> NumPy array
                if img.shape[-1] != image_channels:
                    raise ValueError(f"Image '{filename}' has {img.shape[-1]} channels, expected {image_channels} channels.")

                image_list.append(img)
            except Exception as e:
                print(f"Error loading image '{filename}': {str(e)}")

    images = np.array(image_list)
    image_count = len(images)
    print(image_count)
    return images



def load_labels_from_csv(folder_path):
    labels = []
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV file found in the specified folder")

    csv_file_path = os.path.join(folder_path, csv_files[0])

    df = pd.read_csv(csv_file_path, header=None)
    labels = df.values.flatten()

    labels_array = np.array(labels)
    labels_array = labels_array.reshape(-1, 1)  # Reshaping to have (image_count, 1) shape
    return labels_array


def normalize_images(images):
    images = images.astype('float32')
    # Normalize the pixel values to the range [-1, 1], I got help from the ChatGpt
    images = (images / 127.5) - 1.0

    return images


def split_test_val(test_data, split_point):
    if split_point >= len(test_data):
        raise ValueError("Split point is greater than or equal to the length of the test data.")

    array2 = test_data[:split_point]  # Array before the split point
    array3 = test_data[split_point:]  # Array after the split point

    return array2, array3


#I searched about one-hot encoding using ChatGPT and BlackBox
def one_hot_encode_labels(train_labels, validation_labels, test_labels, num_classes):

    train_labels_encoded = to_categorical(train_labels, num_classes=num_classes)
    validation_labels_encoded = to_categorical(validation_labels, num_classes=num_classes)
    test_labels_encoded = to_categorical(test_labels, num_classes=num_classes)

    return train_labels_encoded, validation_labels_encoded, test_labels_encoded


def create_model(input_shape, num_classes):
    x = Input(shape=input_shape)

    y = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(x)
    y = MaxPooling2D((2, 2))(y)

    y = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(y)
    y = MaxPooling2D((2, 2))(y)

    y = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(y)

    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)
    y = Dense(64, activation='relu')(y)
    y = Dense(num_classes, activation='softmax')(y)

    model = Model(inputs=x, outputs=y)

    return model


train_location = "C:\\University\\Machine Learning for Computer Vision\\Photos\\training"
test_location = "C:\\University\\Machine Learning for Computer Vision\\Photos\\testing"

image_rows = 32  # Specify your desired image dimensions
image_cols = 32
image_channels = 3

Y_test = load_labels_from_csv(test_location)
X_test = load_images(test_location,image_rows, image_cols, image_channels)

Y_train = load_labels_from_csv(train_location)
X_train = load_images(train_location,image_rows, image_cols, image_channels)

normalized_training_images = normalize_images(X_train)
normalized_test_images = normalize_images(X_test)

split_point = 3000
test_set, validation_set = split_test_val(normalized_test_images, split_point)
test_labels, validation_labels = split_test_val(Y_test, split_point)

num_classes = 10
train_labels_encoded, validation_labels_encoded, test_labels_encoded = one_hot_encode_labels(Y_train, validation_labels, test_labels, num_classes)

input_shape = (32, 32, 3)  # Specify your desired input shape
num_classes = 10  # Number of output classes
model = create_model(input_shape, num_classes)
model.summary()

learning_rate = 3e-4
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(
    x=normalized_training_images,
    y=train_labels_encoded,
    epochs=12,
    validation_data=(validation_set, validation_labels_encoded)
)

score = model.evaluate(test_set, test_labels_encoded, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, GroupNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

import optimizers
from centralized_gradients import  centralized_gradients_for_optimizer
from optimizers import adam
from PIL import Image
import time
import os
import pandas as pd


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
    
    y = Conv2D(32, kernel_size = (3, 3), activation='relu', padding='same',  strides=1,kernel_initializer='he_normal')(x)
    y = MaxPooling2D((2, 2))(y)
    
    y = Conv2D(64,kernel_size = (3, 3), activation='relu', padding='same',  strides=1,kernel_initializer='he_normal')(y)
    y = MaxPooling2D((2, 2))(y)
  
    y = Conv2D(128, kernel_size =(3, 3), activation='relu', padding='same', strides=1,kernel_initializer='he_normal')(y)
    
    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)
    y = Dense(64, activation='relu')(y)
    y = Dense(num_classes, activation='softmax')(y)
    
    model = Model(inputs=x, outputs=y)
    
    return model

#Added Batch Normalization to the Model
def create_model_2(input_shape, num_classes):
    x = Input(shape=input_shape)
    
    y = Conv2D(32, kernel_size = (3, 3), activation='relu', padding='same',  strides=1,kernel_initializer='he_normal')(x)
    y = BatchNormalization()(y)  # Add BatchNormalization after Conv2D
    y = MaxPooling2D((2, 2))(y)
    
    y = Conv2D(64,kernel_size = (3, 3), activation='relu', padding='same',  strides=1,kernel_initializer='he_normal')(y)
    y = BatchNormalization()(y)  # Add BatchNormalization after Conv2D
    y = MaxPooling2D((2, 2))(y)
    
    y = Conv2D(128, kernel_size =(3, 3), activation='relu', padding='same', strides=1,kernel_initializer='he_normal')(y)
    y = BatchNormalization()(y)  # Add BatchNormalization after Conv2D

    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)
    y = BatchNormalization()(y)  # Add BatchNormalization after Dense
    y = Dense(64, activation='relu')(y)
    y = Dense(num_classes, activation='softmax')(y)
    
    model = Model(inputs=x, outputs=y)
    
    return model

#Added Group Normalization to the Model
def create_model_3(input_shape, num_classes):
    x = Input(shape=input_shape)
    
    y = Conv2D(32, kernel_size = (3, 3), activation='relu', padding='same',  strides=1,kernel_initializer='he_normal')(x)
    y = GroupNormalization()(y)  # Add BatchNormalization after Conv2D
    y = MaxPooling2D((2, 2))(y)
    
    y = Conv2D(64,kernel_size = (3, 3), activation='relu', padding='same',  strides=1,kernel_initializer='he_normal')(y)
    y = GroupNormalization()(y)  # Add BatchNormalization after Conv2D
    y = MaxPooling2D((2, 2))(y)

    
    y = Conv2D(128, kernel_size =(3, 3), activation='relu', padding='same', strides=1,kernel_initializer='he_normal')(y)
    y = GroupNormalization()(y)  # Add BatchNormalization after Conv2D

    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)
    y = GroupNormalization()(y)  # Add BatchNormalization after Dense
    y = Dense(64, activation='relu')(y)
    y = Dense(num_classes, activation='softmax')(y)
    
    model = Model(inputs=x, outputs=y)
    
    return model


#Added DropOut to the Model
def create_model_4(input_shape, num_classes):
    x = Input(shape=input_shape)
    
    y = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(x)
    y = MaxPooling2D((2, 2))(y)
    
    y = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(y)
    y = MaxPooling2D((2, 2))(y)

    y = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', strides=1, kernel_initializer='he_normal')(y)
    
    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)
    y = Dropout(0.2)(y)  # Add Dropout after the 1st dense layer with a rate of 0.2
    y = Dense(64, activation='relu')(y)
    y = Dropout(0.2)(y)  # Add Dropout after the 2nd dense layer with a rate of 0.2
    y = Dense(num_classes, activation='softmax')(y)
    
    model = Model(inputs=x, outputs=y)
    
    return model

#Make train dataset function necessary for the custom training loop
def make_train_dataset(x_data, y_data, batch_size, train_size=0.9, shuffle=True):
    size = len(x_data)
    indices = np.arange(size)

    if shuffle:
        np.random.shuffle(indices)

    train_samples = int(size * train_size)

    train_indices = indices[:train_samples]
    x_train, y_train = x_data[train_indices], y_data[train_indices]

    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_train = dataset_train.shuffle(buffer_size=len(x_train)).batch(batch_size)

    return dataset_train

#Make validation dataset function necessary for the custom training loop
def make_val_dataset(x_data, y_data, batch_size, train_size=0.9, shuffle=True):
    size = len(x_data)
    indices = np.arange(size)

    if shuffle:
        np.random.shuffle(indices)

    train_samples = int(size * train_size)

    val_indices = indices[train_samples:]
    x_val, y_val = x_data[val_indices], y_data[val_indices]

    dataset_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    dataset_val = dataset_val.batch(batch_size)

    return dataset_val

#Data augmentation function
def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    #image = tf.image.random_crop(image, size=[32, 32, 3])


    return image

#Function for the CutMix implementation
def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


train_location = "C:\\University\\Machine Learning for Computer Vision\\Photos\\training"
test_location = "C:\\University\\Machine Learning for Computer Vision\\Photos\\testing"
image_rows = 32  # Specify your desired image dimensions
image_cols = 32
image_channels = 3

Y_test = load_labels_from_csv(test_location)
X_test = load_images(test_location,image_rows, image_cols, image_channels)

Y_train = load_labels_from_csv(train_location)
X_train = load_images(train_location,image_rows, image_cols, image_channels)

X_train = normalize_images(X_train)
X_test = normalize_images(X_test)

split_point = 3000
test_set, test_validation_set = split_test_val(X_test, split_point)
test_labels, test_validation_labels = split_test_val(Y_test, split_point)

#One-hot encoding
num_classes = 10
Y_train = to_categorical(Y_train, num_classes=num_classes)
test_validation_labels = to_categorical(test_validation_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

#For custom training loop
train_dataset = make_train_dataset(X_train, Y_train, batch_size=32,train_size=0.9, shuffle=True)
val_dataset = make_val_dataset(test_validation_set, test_validation_labels, batch_size=32, train_size=0.9, shuffle=True)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()


#Creating the model
input_shape = (32, 32, 3)  # Specify your desired input shape
num_classes = 10  # Number of output classes
model = create_model_4(input_shape, num_classes)

learning_rate = 3e-4
#optimizer = Adam(learning_rate=learning_rate)


#Gradient Optimzation
adam_opt = optimizers.adam(learning_rate=learning_rate)
optimizer = adam_opt

#SGD Optimzier.
#optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, verbose=1, min_delta=0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#Summary of the model
model.summary()



"""
# Fit the model without callbacks
model.fit(
    x=X_train,
    y=train_labels_encoded,
    epochs=12,
    validation_data=(test_validation_set, validation_labels_encoded),
)"""

"""
#Fit the model with callbacks and 20 epochs
model.fit(
    x=X_train,
    y=train_labels_encoded,
    epochs=20,
    validation_data=(test_validation_set, validation_labels_encoded),
    callbacks = [reduce_lr]
)
"""
epochs = 15 # Number of epochs as needed
batch_size = 16  # Adjustable
beta = 0.5
cutmix_prob = 1.0
num_classes = 100
r = 0.7 # fixed for cutmix display

loss_fn = tf.keras.losses.CategoricalCrossentropy()
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

"""
#I got the help of ChatGpt when adopting the custom train loop in my code
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # Apply CutMix
            if beta > 0 and r < cutmix_prob:
                lam = np.random.beta(beta, beta)
                rand_index = tf.random.shuffle(tf.range(len(y_batch_train)))
                target_a = y_batch_train
                target_b = tf.gather(y_batch_train, rand_index)
                bbx1, bby1, bbx2, bby2 = rand_bbox(x_batch_train.shape, lam)
                image_a = x_batch_train
                image_b = tf.gather(x_batch_train, rand_index)
                mask = np.ones_like(x_batch_train)
                mask[:, bbx1:bbx2, bby1:bby2, :] = 0
                x_batch_train = tf.math.multiply(image_a, mask) + tf.math.multiply(image_b, (abs(1. - mask)))
                y_batch_train = lam * target_a + (1. - lam) * target_b

            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric
        train_acc_metric.update_state(y_batch_train, logits)

        # Log every 200 batches
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * batch_size))

    # Display metrics
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        # Update validation metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))

    #reduce_lr.on_epoch_end(epoch, logs={'val_accuracy': val_acc, 'lr': optimizer.lr})

    score = model.evaluate(test_set, test_labels, verbose=0)
    print('Test accuracy:', score[1])
"""


#SADT Training Loop
def add_noise(grads):
    cgrads = []
    for grad in grads:
        rank = len(grad.shape)
        if rank > 1:
            rv = (tf.random.normal(grad.shape, mean=0.0, stddev=1e-4, dtype=tf.dtypes.float32))
            grad = grad + rv
        cgrads.append(grad)
    return cgrads

def add_grads(parameters, gradients):
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
        ap = 5e-1
        new_grads.append(grads + params)
    return new_grads

def sadt_fit(train_dataset, val_dataset, model, optimizer, epochs, reduce_lr):
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    kld = tf.keras.losses.KLDivergence()

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape(persistent=True) as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # --- SADT: Perturb parameters and distill with auxiliary teacher ---
            wgt1 = model.get_weights()
            model.set_weights(add_noise(wgt1))

            with tf.GradientTape(persistent=True) as tape:
                logits1 = model(x_batch_train, training=True)
                loss_value = kld(logits, logits1)

            model.set_weights(wgt1)
            grads1 = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(add_grads(grads, grads1), model.trainable_weights))
            # --- SADT end ---

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))

        #reduce_lr.on_epoch_end(epoch, logs={'val_accuracy': val_acc})

        score = model.evaluate(test_set, test_labels, verbose=0)
        print('Test accuracy:', score[1])

    return model


##SADT Loop
#epochs = 10 # or any other number of epochs you want
sadt_model = sadt_fit(train_dataset, val_dataset, model, optimizer, epochs, reduce_lr)



score = model.evaluate(test_set, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# -*- coding: utf-8 -*-
"""Neural_network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sFd-MYWwU0GN4NTsKvFnhzUl8fpxd5zx
"""

import tensorflow as tf
from tensorflow.keras.layers import (Dense,
                                     BatchNormalization,
                                     LeakyReLU,
                                     Reshape,
                                     Conv2DTranspose,
                                     Conv2D,
                                     Dropout,
                                     Flatten)
import matplotlib.pyplot as plt
from numpy import array, zeros,count_nonzero, ndarray

from emnist import extract_training_samples
import os
import time
from IPython import display

def get_concrete_datasets(dataset:ndarray, classes:ndarray):
    datasets_by_classes = []
    for i in range(0, 36):
        # print(i,end="")
        TempDataset = zeros((count_nonzero(classes==i), 28,28))
        datasets_by_classes.append(TempDataset)

    current_dataset_index = [0 for i in range(0,36)]
    for i in range(0, len(classes)):
        if classes[i]<=35:
            # print(classes[i], end = "")
            datasets_by_classes[classes[i]][current_dataset_index[classes[i]]] = dataset[i].copy()
            current_dataset_index[classes[i]]+=1
    print(current_dataset_index)
    return datasets_by_classes

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model

def discriminator_loss(real_output, fake_output, cross_entropy):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output,cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images,cross_entropy):

    # 1 - Create a random noise to feed it into the model
    # for the image generation
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    # 2 - Generate images and calculate loss values
    # GradientTape method records operations for automatic differentiation.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output,cross_entropy)
        disc_loss = discriminator_loss(real_output, fake_output,cross_entropy)

    # 3 - Calculate gradients using loss values and model variables
    # "gradient" method computes the gradient using 
    # operations recorded in context of this tape (gen_tape and disc_tape).

    # It accepts a target (e.g., gen_loss) variable and 
    # a source variable (e.g.,generator.trainable_variables)
    # target --> a list or nested structure of Tensors or Variables to be differentiated.
    # source --> a list or nested structure of Tensors or Variables.
    # target will be differentiated against elements in sources.

    # "gradient" method returns a list or nested structure of Tensors  
    # (or IndexedSlices, or None), one for each element in sources. 
    # Returned structure is the same as the structure of sources.
    gradients_of_generator = gen_tape.gradient(gen_loss,
                                               generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)

    # 4 - Process  Gradients and Run the Optimizer
    # "apply_gradients" method processes aggregated gradients. 
    # ex: optimizer.apply_gradients(zip(grads, vars))
    """
    Example use of apply_gradients:
    grads = tape.gradient(loss, vars)
    grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
    # Processing aggregated gradients.
    optimizer.apply_gradients(zip(grads, vars), experimental_aggregate_gradients=False)
    """
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs, checkpoint,cross_entropy):
    # A. For each epoch, do the following:
    for epoch in range(epochs):
        start = time.time()
        # 1 - For each batch of the epoch,
        for image_batch in dataset:
            # 1.a - run the custom "train_step" function
            # we just declared above
            train_step(image_batch,cross_entropy)

        # 2 - Produce images for the GIF as we go
        display.clear_output(wait=True)
        # generate_and_save_images(generator,
        #                       epoch + 1,
        #                     seed)

        # 3 - Save the model every 5 epochs as
        # a checkpoint, which we will use later
        if (epoch + 1) % 100 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        # 4 - Print out the completed epoch no. and the time spent
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # B. Generate a final image after the training is completed
    display.clear_output(wait=True)
    #generate_and_save_images(generator,
    #                     epochs,
    #                 seed)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    # 1 - Generate images
    predictions = model(test_input, training=False)
    # 2 - Plot the generated images
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    # 3 - Save the generated images
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

"""Зададим глобальные переменные:"""

# EPOCHS = 60
# num_examples_to_generate = 4
# noise_dim = 100
#
# images, labels = extract_training_samples('bymerge')
# seed = tf.random.normal([num_examples_to_generate, noise_dim])
# datasets_list = get_concrete_datasets(images,labels)
# del images
# class_names_list = ["0", "1", "2","3", "4", "5","6", "7", "8","9",
#                     "a", "b","c", "d", "e","f", "g", "h","i", "j", "k",
#                     "l", "m", "n","o", "p", "q","r", "s", "t","u", "v",
#                     "w","x", "y", "z"]
#
#
#
# """Главный цикл:"""
#
# for i in range(0,len(datasets_list)):
#
#     i = len(datasets_list)-1
#     #Приведем датасет к нужному типу данных
#     ds = datasets_list[i].reshape(datasets_list[i].shape[0], 28, 28).astype('float32')
#     ds = (ds - 127.5) / 127.5 # Normalize the images to [-1, 1]
#
#     #укажем размер буфера и батча, папки для сохранения чепоинтов
#     BUFFER_SIZE = len(datasets_list[i])
#     BATCH_SIZE = 256
#     checkpoint_dir =  './training_checkpoints'
#     checkpoint_prefix = os.path.join(checkpoint_dir,class_names_list[i], "ckpt")
#     print(checkpoint_prefix)
#
#     # Batch and shuffle the data
#     train_dataset = tf.data.Dataset.from_tensor_slices(ds).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#     print(train_dataset)
#
#     #создадим модель нашей нейронной сети
#     generator = make_generator_model()
#     discriminator = make_discriminator_model()
#
#
#     cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#     generator_optimizer = tf.keras.optimizers.Adam(1e-4)
#     discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#
#     checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                      discriminator_optimizer=discriminator_optimizer,
#                                      generator=generator,
#                                      discriminator=discriminator)
#
#     #seed = tf.random.normal([num_examples_to_generate, noise_dim])
#
#
#     #начинаем процесс обучения
#     train(train_dataset, EPOCHS, checkpoint,cross_entropy)
#     break

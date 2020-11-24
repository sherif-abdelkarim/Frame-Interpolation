
#import dependencies
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras import layers
import os 
from pathlib import Path
import h5py
import time
import numpy as np
from IPython import display
import sys

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



#sys.argv[1]
path = Path('/home/pattersonlab/Documents/Alex/Tomography_ml/Training_own_models_ab/full')
files = sorted(path.glob('*hdf5'))

for file in files:
    f = h5py.File(file, 'r')
    x_train = f['x_train']
    x_train = np.array(x_train)
    shape_train = x_train.shape[0]
    shape_train = int(shape_train/64.)*64 # ssim needs to have the same batch size for all the mini batches
    x_train = x_train[:shape_train,:,:,:].reshape((shape_train, 48, 48,2)) # reshape((195200, 1, 48, 48,2))
    y_train = f['y_train']
    y_train = np.array(y_train)
    y_train = y_train[:shape_train,:,:].reshape((shape_train, 48, 48,1)) # reshape((195200, 1, 48, 48,2))
    x_val = f['x_val']
    x_val = np.array(x_val)
    y_val = f['y_val']
    y_val = np.array(y_val)
    
    BUFFER_SIZE = y_train.shape[0]
    BATCH_SIZE = 64
    
    
    train_inputs3 = tf.data.Dataset.from_tensor_slices(x_train).batch(BATCH_SIZE)
    train_grounds3 = tf.data.Dataset.from_tensor_slices(y_train).batch(BATCH_SIZE)
    #train_inputs2 = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    #train_grounds2 = tf.data.Dataset.from_tensor_slices(y_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    #train_inputs = tf.data.Dataset.from_tensor_slices(x_train)
    #train_grounds = tf.data.Dataset.from_tensor_slices(y_train)
    
    
    def make_generator_model():
        model = tf.keras.Sequential()
        #block0 - input
        #model.add(layers.Input(shape=(48,48,2)))
        #block1
        model.add(layers.Conv2D(filters=32, kernel_size=(2,2), strides=(1,1), padding='same',use_bias=False,
                            input_shape=[48, 48, 2]))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        #block2
        model.add(layers.Conv2D(filters=64, kernel_size=(2,2), strides=(2,2),padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        #block3
        model.add(layers.Conv2D(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        #block4
        model.add(layers.Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', 
                                        use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.5))
        #block5
        model.add(layers.Conv2DTranspose(filters=32, kernel_size=(2,2), strides=(2,2), padding='same', 
                                        use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.5)) 
        #block6 - output layer
        model.add(layers.Conv2DTranspose(filters=1, kernel_size=(2,2), strides=(1,1), padding='same', 
                                        use_bias=False, activation='tanh'))

        return model

    generator = make_generator_model()
    
    
    def make_discriminator_model():
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same',
                            input_shape=[48,48,1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        
        return model
    
    discriminator = make_discriminator_model()
    
    
    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    
    def discriminator_loss(real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    
    def generator_loss(fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    
    checkpoint_dir = f'./{file.name[:-5]}_training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    
    
    EPOCHS = 100
    img_dimension = [48,48]
    noise_dim = 100
    num_examples_to_generate = 16

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    
    
    
    @tf.function
    def train_step(train_inputs, train_grounds):
        noise = tf.random.normal([BATCH_SIZE, 48, 48, 1])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(train_inputs, training=True)
            
            real_output = discriminator(train_grounds, training=True)
            fake_output = discriminator(generated_images, training=True)
            
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
    def train(input_dataset, ground_dataset, epochs):
        for epoch in range(epochs):
            #if epoch % 20 == 0:
            #    time.sleep(20)
            print(epoch)
            start = time.time()
            
            for input_batch, input_ground  in zip(input_dataset, ground_dataset):
                train_step(input_batch, input_ground)
            checkpoint.save(file_prefix = checkpoint_prefix)

            
            # Produce images for the GIF as we go
            """
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                    epoch + 1,
                                    seed)


            """
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            """
            # Generate after the final epoch
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                epochs,
                                seed)
        """
        
    def generate_and_save_images(model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    train(train_inputs3, train_grounds3, EPOCHS)
    
    checkpoint.save(file_prefix = checkpoint_prefix)
    
    generator.save(f'./{file.name[:-5]}_generator_model.h5')
    discriminator.save(f'./{file.name[:-5]}_step_20_discriminator_model.h5')
    
    

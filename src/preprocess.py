import tensorflow as tf
import numpy as np
import os
from PIL import Image
import tensorflow_addons as tfa


def load_add_preprocess_image(image_path, img_size = (256,256)):
    """
    Load and preprocess an image.
    Args:
        image_path (str): Path to the image file.
        img_size (tuple): Target size of the image (height, width).
    Returns:
        image (tf.Tensor): Preprocessed image as a TensorFlow tensor.
    """


    img = tf.io.read_file(image_path) # converts the image to binary format
    img = tf.image.decode_image(img) # binary string into image tensor
    img = tf.image.resize(img,img_size)
    img = img/255.0 # normailze the image
    return img

def augument_image(image):
    # Randomly flip the image horizontally
    image = tf.image.random_flip_left_right(image)

    angle = tf.random.uniform([],-20,20) * np.pi/100
    image = tfa.image.rotate(image,angle)

    image = tf.image.random_brightness(image, max_delta = 0.2)
    image = tf.image.random_contrast(image, lower = 0.2, upper =1.2)

    image = tf.image.random_crop(image, size=[224,224,3])
    image = tf.image.resize(image,[ 256,256])

    return image
    

def preprocess_and_augment_dataset(folder,output_folder, img_size = (256,256),augument_factor = 2):
    """
    Preprocess and augment images in a folder.
    Args:
        folder (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        img_size (tuple): Target size of the images.
        augment_factor (int): Number of augmented images to generate per original image.
    """


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_paths = [os.path.join(folder, fname) for fname in os.listdir(folder) ]

    for i , image_path in enumerate(image_paths):
        image  = load_add_preprocess_image(image_path, img_size)

        output_path = os.path.join(output_folder, f"original_{i}.jpg")

        for j in range(augument_factor):
            augmented_image = augument_image(image)
            output_path  = os.path.join(output_folder, f"augmented_{i}_{j}.jpg")

            tf.keras.preprocessing.image.save_img(output_path,augmented_image.numpy())



preprocess_and_augment_dataset("data/VincentVanGogh", "cleandata/augmented_vangogh", (256, 256), 2)
preprocess_and_augment_dataset("data/Monat", "cleandata/augmented_monet", (256, 256), 2)
preprocess_and_augment_dataset("data/ContentImage", "cleandata/augmented_content", (256, 256), 2)
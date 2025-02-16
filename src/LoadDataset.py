import os
import tensorflow as tf

# Load dataset directly from preprocessed 'cleandata' folder
def load_dataset(folder):
    """
    Load a dataset of preprocessed images from the 'cleandata' folder.
    Args:
        folder (str): Path to the folder containing preprocessed images.
    Returns:
        dataset (tf.data.Dataset): Dataset of preprocessed images.
    """
    image_paths = [os.path.join(folder, fname) for fname in os.listdir(folder) if fname.endswith('.jpg') or fname.endswith('.png')]
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda x: tf.image.decode_image(tf.io.read_file(x), channels=3), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset




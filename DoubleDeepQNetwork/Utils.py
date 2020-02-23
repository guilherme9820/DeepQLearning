from skimage import transform
import os
import numpy as np
import re
import glob


class WeightSaver:

    def __init__(self, save_freq, weights_dir, max_save=10):
        self._save_freq = save_freq
        self._max_save = max_save
        self._weights_dir = weights_dir

    def __call__(self, model, step):

        # Template for weight file name
        template = "weights_ep_{}.h5"

        if step % self._save_freq == 0:

            # Searches for already existing weight files
            weight_files = glob.glob(self._weights_dir + '*.h5')

            if weight_files:

                # Sorts the list of weight files by last modification date (older to newer)
                weight_files.sort(key=os.path.getmtime, reverse=False)

                # If the maximum of weight files was reached then delete the older one
                if len(weight_files) == self._max_save:
                    os.remove(weight_files[0])

                # Gets the number of newest weight file
                last_saved = int(re.split("[./_]", weight_files[-1])[-2])

                file_name = self._weights_dir + template.format(last_saved + self._save_freq)
            else:
                file_name = self._weights_dir + template.format(0)

            model.save_weights(file_name)


def restore_last_saved(model, agent, weights_dir):

    # Searches for already existing weight files
    weight_files = glob.glob(weights_dir + '*.h5')
    if weight_files:

        # Sorts the list of weight files by last modification date (newer to older)
        weight_files.sort(key=os.path.getmtime, reverse=True)
        model.restore_weights(agent, weight_files[0])
        print(f"Restored weights from file: {weight_files[0]}")


def preprocess_frames(frames, new_size=[84, 84]):
    """ Crop unnecessary pixels, resize image and apply dimensionality redution to save memory"""

    channels = len(frames)
    height, width = new_size

    # Removes ceiling from image because it contains no util information
    # and normalize it
    cropped = np.asarray([frame[30:-10, 30:-30] / 255 for frame in frames])

    # Resize image
    resized = transform.resize(cropped, [channels, height, width])

    # Apply dimensionality reduction over all frames
    compressed_frames = [dimensionality_reduction(frame) for frame in resized]

    return np.stack(compressed_frames, axis=2).astype(np.float32)


def dimensionality_reduction(image, threshold=0.8):
    """ Dimensionality reduction using Singular Value Decomposition (SVD) """

    # If image is not a black screen (all elements are not zero)
    if image.any():

        # Performs singular value decomposition
        u, s, vh = np.linalg.svd(image, full_matrices=False)

        # Calculates cumulative energy of the singular values.
        cumsum = np.cumsum(s) / np.sum(s)

        # Gets the number of singular values that have a cumulative
        # energy at least equal or above the specified threshold
        quantity = np.where(cumsum >= threshold)[0][0]

        # Return the reconstructed image in a compressed form
        return np.dot(np.multiply(u[:, :quantity], s[:quantity]), vh[:quantity, :])
    else:
        return image

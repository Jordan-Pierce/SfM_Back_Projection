import os
import glob

import cv2
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from skimage.morphology import disk
from sklearn.model_selection import train_test_split

import segmentation_models as sm
from segmentation_models.losses import *
from segmentation_models.metrics import *

sm.set_framework('keras')

import keras
from keras.callbacks import *
from keras.utils import to_categorical
from keras.optimizers import Adam

keras.backend.set_image_data_format('channels_last')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------

# labels for each class category of interest and their color values
class_colors = {'white': [255, 255, 255],
                'black': [0, 0, 0],
                'red': [158, 11, 15],
                'green': [0, 128, 0],
                'blue': [0, 162, 232],
                'orange': [247, 148, 29]}

# Define a dictionary mapping class colors to integer labels
class_categories = {'white': 0,
                    'black': 1,
                    'red': 2,
                    'green': 3,
                    'blue': 4,
                    'orange': 5}

# Get the number of classes in the dataset
n_classes = len(list(class_categories.values()))

# Initialize an array of class weights, where all classes have equal weight
class_weights = np.array([1 for _ in range(n_classes)])

# Set the weights for the first two classes (white and black) to 0,
# as these classes will not be used
class_weights[0:2] = 0

# Dimensions of the input images
HEIGHT, WIDTH = 1619, 2159
DOWNSIZE_RATIO = .40

# Declare a variable `BACKBONE` and set its value to 'efficientnetb0'.
BACKBONE = 'efficientnetb0'

# Setting the batch size, num epochs, and learning rate
BATCH_SIZE = 1
NUM_EPOCHS = 25
LR = .002
WORKERS = 8

# Whether to save the predictions from a trained model, and to display them.
SAVE = True
DISPLAY = True

NUM_ITERATIONS = 10


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def uncolor_mask(mask, class_colors=class_colors):
    """
       Converts a colored mask to a binary mask.

       Args:
           mask (numpy.ndarray): The colored mask.
           class_colors (dict): A dictionary mapping class labels to RGB tuples.

       Returns:
           numpy.ndarray: The binary mask.
       """
    uncolored_mask = np.zeros(shape=mask.shape[0:2], dtype=np.uint8)

    for integer, color in enumerate(list(class_colors.values())):
        y, x = np.where(np.all(mask == color, axis=-1))
        uncolored_mask[y, x] = integer

    return uncolored_mask


def color_mask(mask, class_colors=class_colors):
    """
   Converts a binary mask to a colored mask.

   Args:
       mask (numpy.ndarray): The binary mask.
       class_colors (dict): A dictionary mapping class labels to RGB tuples.

   Returns:
       numpy.ndarray: The colored mask.
   """

    colored_mask = np.zeros(shape=(mask.shape[0], mask.shape[1], 3),
                            dtype=np.uint8)

    for integer, color in enumerate(list(class_colors.values())):
        colored_mask[mask == integer] = color

    return colored_mask


def get_resized_dimensions(dimensions, downsizing_ratio):
    """
    Computes the new dimensions that the input image should be resized to,
    such that both the height and width are evenly divisible by 32,
    while maintaining the original aspect ratio.

    Args: image: The input image. downsizing_ratio: The ratio by which the
    dimensions of the input image should be reduced.

    Returns:
        A tuple containing the new height and width of the image.
    """
    # Get the dimensions of the input image
    height, width = dimensions

    # Compute the new height and width of the image based on the downsizing
    # ratio
    new_height = int(height * downsizing_ratio)
    new_width = int(width * downsizing_ratio)

    # Make sure that the new height and width are evenly divisible by 32
    new_height = (new_height + 31) // 32 * 32
    new_width = (new_width + 31) // 32 * 32

    return new_height, new_width


def get_second_max_mask(mask, threshold=0.5):
    """
    Given a segmentation mask with dimensions (height, width, 6), where the
    first 2 channels are background classes, this function finds the indices
    with the second highest probabilities that are not one of the background
    classes, and thresholds them so that only values that are greater than
    the specified threshold are accepted.

    Args: mask (np.ndarray): The segmentation mask with dimensions (height,
    width, 6) threshold (float): The threshold value to use when filtering
    the second highest probabilities. Default value is 0.5.

    Returns: np.ndarray: A binary mask with the same dimensions as the input
    mask, where the values are either one of the background classes or the
    second highest probabilities that are greater than the threshold.
    """

    # Get the indices with the highest probabilities
    max_mask = np.argmax(mask, axis=2)

    # Get the indices with the highest probabilites that are background classes
    background_indices = np.logical_or(max_mask == 0, max_mask == 1)

    # Get the indices with the second highest probabilities
    sorted_mask = np.argsort(mask, axis=2)
    second_max_indices = sorted_mask[:, :, -2]

    # Get the indices where background is highest class and second highest
    # probabilities are
    second_choice_indices = np.logical_and(second_max_indices,
                                           background_indices)

    # Get the probabilities of the second highest class categories
    second_probs = np.sort(mask, axis=2)[:, :, -2]

    # From the second highest class categories probabilities, get the indices
    second_choice_probs = np.take(second_probs, second_choice_indices)

    # Get the indices where the second highest probabilities are higher than a
    # threshold
    second_choice_threhold = np.where(second_probs > threshold)

    # Create an empty mask containing zeros
    second_max_mask = np.zeros_like(second_probs)

    # Fill in the indices of the mask where the second highes probabilities
    # higher than a threhold are, with the class category values from the
    # second highest class categories
    second_max_mask[second_choice_threhold] = second_max_indices[
        second_choice_threhold]

    combined_mask = max_mask.copy()
    combined_mask[second_choice_threhold] = second_max_indices[
        second_choice_threhold]

    return combined_mask


def get_thresholds(mask):
    """
    Calculate the thresholds for the mask values by sorting the values in
    the mask along the specified axis, splitting the values into two groups
    based on whether they are above or below a certain value, and then
    computing a histogram for each group.
    """

    # Sort the values in the mask along the specified axis
    second_probs = np.sort(mask, axis=2)[:, :, -2].flatten()

    # Select values in the mask that are above a certain threshold
    high_values = np.flip(second_probs[second_probs > 0.1])

    # Compute a histogram for the high-value group
    n, h_bins = np.histogram(high_values, bins=10)

    # Select values in the mask that are below a certain threshold
    low_values = np.flip(second_probs[second_probs < 0.001])

    # Compute a histogram for the low-value group
    n, l_bins = np.histogram(low_values, bins=50)

    # Concatenate the histogram bins for the two groups and sort them in
    # reverse order
    bins = np.sort(np.concatenate([h_bins, l_bins]))[::-1]

    # Return the calculated thresholds
    return bins


def supervise_predictions(mask, pred, size=1):
    """
    Adjusts the predicted class labels in the specified segmentation mask to
    match the class labels in the corresponding ground truth mask.

    Args:
        mask (numpy.ndarray): The ground truth segmentation mask.
        pred (numpy.ndarray): The predicted segmentation mask.

    Returns:
        numpy.ndarray: The adjusted predicted segmentation mask.
    """
    d = disk(size)
    super_pred = np.ones(shape=mask.shape + (n_classes,), dtype=np.uint8)

    for channel in range(n_classes)[2:]:
        # Check if the predicted class label in a given pixel differs from
        # the corresponding ground truth label
        in_pred = pred == channel
        in_mask = mask == channel
        difference = np.logical_and(in_pred, ~in_mask)

        # Check whether the given pixel is surrounded by a majority of
        # pixels with the correct ground truth label
        neighborhoods = np.logical_and(difference,
                                       convolve(in_mask, d, mode='constant'))
        neighborhoods = np.logical_or(in_mask, neighborhoods)

        # Adjust the predicted class label of the given pixel to match the
        # #ground truth label
        super_pred[neighborhoods, channel] = channel

    return super_pred


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras.

    Args: dataframe (pandas.DataFrame): A Pandas dataframe containing image
    paths and mask paths. batch_size (int): The number of image-mask pairs
    to generate per batch. augment (bool): Whether to apply data
    augmentation to the generated images. shape (tuple): The shape of the
    generated images. n_classes (int, optional): The number of classes in
    the dataset. Defaults to len(list(class_categories)).
    """

    def __init__(self, dataframe, batch_size, augment, shape, n_classes=n_classes):
        # Store the specified dataframe, batch size, data augmentation
        # settings, and image shape
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.augment = augment
        self.shape = shape

    # This method returns the number of steps per epoch, which is the number
    # of batches in an epoch
    def __len__(self):
        return len(self.dataframe) // self.batch_size

    # This method shuffles the dataframe and resets the index at the end of
    # each epoch
    def on_epoch_end(self):
        self.dataframe = self.dataframe.reset_index(drop=True)

    # This method generates a batch of data, which is then fed to the model
    # during training
    def __getitem__(self, index):
        # Initialize lists to hold the processed images and masks for the
        # current batch
        processed_images = []
        processed_masks = []

        # Generate the specified number of image-mask pairs for the current
        # batch
        for _ in range(self.batch_size):
            # Read the image and mask for the current index from the dataframe
            the_image = plt.imread(self.dataframe['Images'][index])
            the_mask = plt.imread(self.dataframe['Masks'][index])

            # Convert the mask from a colored mask to a binary mask
            the_mask = uncolor_mask(the_mask)

            # Convert the binary mask to a one-hot encoded mask
            one_hot_mask = to_categorical(the_mask, self.n_classes)

            # Resize the image and mask to the specified shape
            processed_image = cv2.resize(the_image, self.shape, cv2.INTER_LINEAR)
            processed_mask = cv2.resize(one_hot_mask, self.shape, cv2.INTER_NEAREST)

            # If data augmentation is enabled, randomly flip the image and
            # mask horizontally
            if np.random.randint(2) > 0 and self.augment:
                processed_image = np.fliplr(processed_image)
                processed_mask = np.fliplr(processed_mask)

            # Append the processed data to a list
            processed_images.append(processed_image)
            processed_masks.append(processed_mask)

        # Convert the list of processed data into a numpy array
        batch_x = np.array(processed_images)
        batch_y = np.array(processed_masks)

        return (batch_x, batch_y)


def load_data(image_path, dense_path):
    """This function takes in two paths, globs the *.tif* files, and returns
    a dataframe containing both."""

    images = glob.glob(image_path + "*.tif*")
    masks = glob.glob(dense_path + "*.tif*")

    data = pd.DataFrame(list(zip(images, masks)), columns=['Images', 'Masks'])

    return data


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    ROOT = "..\\Project\\Images\\"
    IMAGE_PATH = ROOT + "Tif\\"
    SPARSE_PATH = ROOT + "Sparse\\"
    DENSE_PATH = ROOT + "Dense\\"
    WEIGHTS_PATH = ROOT + "Weights\\"
    DISPLAY_PATH = ROOT + "Display\\"

    # Creates a folder to hold the weights of the trained model
    os.makedirs(WEIGHTS_PATH, exist_ok=True)
    os.makedirs(DISPLAY_PATH, exist_ok=True)
    os.makedirs(DENSE_PATH, exist_ok=True)

    # Get a list of all images and masks in the directories
    data = load_data(IMAGE_PATH, SPARSE_PATH)

    # Split the data into training and validation sets
    train, valid = train_test_split(data, test_size=.25)

    # Reset the indices of the training and validation sets
    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)

    # Calculate the number of steps per epoch for the training and
    # validation sets
    steps_per_epoch_train = len(train) // BATCH_SIZE
    steps_per_epoch_valid = len(valid) // BATCH_SIZE

    # Get new dimensions for training that fit criteria of model
    # (N / 32 % == 0)
    height, width = get_resized_dimensions((HEIGHT, WIDTH), DOWNSIZE_RATIO)

    # Initialize generators for the training and validation sets, using the
    # specified batch size and image shape
    # Reset generators for the training and validation sets
    train_gen = DataGenerator(train, BATCH_SIZE, True, (width, height))
    valid_gen = DataGenerator(valid, BATCH_SIZE, False, (width, height))
    data_gen = DataGenerator(data, 1, False, (width, height))

    # Stops training if val_loss does not continue to decrease after N epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)

    # Reduces the learning rate by % if val_loss does not continue to
    # decrease after N epochs
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=.65,
                                  patience=3,
                                  verbose=1)

    # Saves model weights after epochs in which the val_loss decreases from
    # previously
    model_chkpt = ModelCheckpoint(
        filepath=WEIGHTS_PATH + "model-{epoch:03d}-{iou_score:03f}-{val_iou_score:03f}.h5",
        monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=1)

    # List containing callbacks
    callbacks = [early_stopping, reduce_lr, model_chkpt]

    # Define a U-Net model with the specified BACKBONE, input shape,
    # and activation function. Set the encoder weights to be "imagenet" and
    # set the encoder layers to be frozen. Use batch normalization in the
    # decoder part of the model.
    model = sm.Unet(input_shape=(None, None, 3),
                    backbone_name=BACKBONE,
                    encoder_weights='imagenet',
                    activation='softmax',
                    classes=n_classes,
                    encoder_freeze=True,
                    decoder_use_batchnorm=True)

    model.save_weights(WEIGHTS_PATH + 'initial_weights.h5')

    # Define a list of metrics to track during model training.
    metrics = ['accuracy', iou_score, precision, recall]

    # Define a categorical cross-entropy loss function using class weights.
    cce_loss = CategoricalCELoss(class_weights=class_weights)

    # Compile the model using the Adam optimizer with a specified learning
    # rate. Use a custom loss function that combines the categorical cross
    # entropy loss with the Jaccard loss.
    model.compile(optimizer=Adam(lr=LR),
                  loss=[cce_jaccard_loss + cce_loss],
                  metrics=metrics)

    # Here we're going to continuously train the model until the user
    # decided that the predictions output from the current generation of
    # model is filled in enough.
    CURR_ITERATION = 0

    while CURR_ITERATION < NUM_ITERATIONS:

        # Make a folder to store the dense labels of this current iteration
        CURR_DENSE_PATH = DENSE_PATH + str(CURR_ITERATION) + "/"
        os.makedirs(CURR_DENSE_PATH, exist_ok=True)

        # Training a model using the train generator, validating using the
        # valid generator
        history = model.fit_generator(generator=train_gen,
                                      steps_per_epoch=steps_per_epoch_train,
                                      epochs=NUM_EPOCHS,
                                      validation_data=valid_gen,
                                      validation_steps=steps_per_epoch_valid,
                                      verbose=1,
                                      callbacks=callbacks,
                                      workers=WORKERS)

        # Get a list of all the files in the `Weights` directory that have
        # the `.h5` extension and sort them based on their modification
        # time, with the most recently modified files appearing first; load
        # the weights of the most recently modified file into the `model`
        weights = sorted(glob.glob(WEIGHTS_PATH + "*.h5"), key=os.path.getmtime)[-1]
        print("Loading best weights: ", os.path.basename(weights))
        model.load_weights(weights)

        # Use the trained model to produce prediction for each of the
        # images, post-process those predictions and then save them for the
        # next generation of model.
        print("Updating sparse labels...", "\n")

        for _ in range(len(data)):

            # Get the next image and mask from the data generator Use the
            # model to predict the segmentation mask for the input image
            image, mask = data_gen.__getitem__(_)
            raw_prediction = model.predict(image)

            # Remove the added dimension from each
            image = image.squeeze()
            mask = mask.squeeze()
            raw_prediction = raw_prediction.squeeze()

            # Convert the ground truth mask to a tensor of class indices
            # Convert the raw prediction tensor to a tensor of class indices
            mask = np.argmax(mask, axis=2)
            init_pred = np.argmax(raw_prediction, axis=2)
            curr_pred = np.zeros_like(mask)
            final_pred = mask.copy()

            for threshold in get_thresholds(raw_prediction):

                curr_pred = get_second_max_mask(raw_prediction, threshold)

                # Overwrite the second preds that are not background with
                # the first preds
                for class_id in list(class_categories.values())[2:]:
                    curr_pred[init_pred == class_id] = class_id

                # Use the original mask with the predictions to remove class
                # predictions that are not near existing labels within the
                # mesh; set the black regions of the prediction to match the
                # mask's
                supervised_pred = supervise_predictions(final_pred,
                                                        curr_pred,
                                                        1)

                supervised_pred = np.argmax(supervised_pred, axis=2)
                supervised_pred[mask == 1] = 1

                # Overwrite the second preds that are not background with
                # the first preds
                for class_id in list(class_categories.values())[2:]:
                    final_pred[supervised_pred == class_id] = class_id

                final_pred[mask == 1] = 1

            # Resize the image, mask, and processed prediction to the
            # original dimensions
            image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
            init_pred = cv2.resize(init_pred, (WIDTH, HEIGHT),interpolation=cv2.INTER_NEAREST)
            final_pred = cv2.resize(final_pred, (WIDTH, HEIGHT),interpolation=cv2.INTER_NEAREST)

            # Save the processed prediction to the DENSE folder, using the
            # same filename. This means that for each generation of model
            # training, we'll be using the most updated labels to train on.
            if SAVE:
                dense_path = CURR_DENSE_PATH + os.path.basename(data.iloc[_]['Masks'])
                io.imsave(dense_path, color_mask(final_pred))

            # Display the image, mask, and final prediction side-by-side
            if DISPLAY and _ % 15 == 0:
                plt.figure(figsize=(20, 20))
                plt.subplot(1, 4, 1)
                plt.title("Image")
                plt.imshow(image)
                plt.subplot(1, 4, 2)
                plt.title("Previous Mask")
                plt.imshow(color_mask(mask))
                plt.subplot(1, 4, 3)
                plt.title("Original Prediction")
                plt.imshow(color_mask(init_pred))
                plt.subplot(1, 4, 4)
                plt.title("Supervised Prediction")
                plt.imshow(color_mask(final_pred))
                plt.savefig(DISPLAY_PATH + str(CURR_ITERATION) + "_" +
                            os.path.basename(data.iloc[_]['Masks']))
                plt.close()

        # Resetting the training, validation, and data generators
        data = load_data(IMAGE_PATH, CURR_DENSE_PATH)

        # Splitting data into training, validation
        train, valid = train_test_split(data, test_size=.25)
        train.reset_index(drop=True, inplace=True)
        valid.reset_index(drop=True, inplace=True)

        # Creating new generators
        train_gen = DataGenerator(train, BATCH_SIZE, True, (width, height))
        valid_gen = DataGenerator(valid, BATCH_SIZE, False, (width, height))
        data_gen = DataGenerator(data, 1, False, (width, height))

        # Load the initial weights to start fresh, avoid overfitting
        model.load_weights(WEIGHTS_PATH + 'initial_weights.h5')
        keras.backend.set_value(model.optimizer.learning_rate, LR)

        CURR_ITERATION += 1

        if CURR_ITERATION == NUM_ITERATIONS:
            NUM_ITERATIONS += int(input("How many more iterations?: "))

    print("Done.")
import os
import glob

import cv2
import rawpy
import imageio

# Set the output directory, create
output_dir = "../Project/Images/Tiff/"
os.makedirs(output_dir, exist_ok=True)

# Collect all the RAW images
images = glob.glob("../Project/Images/Raw/*.ARW")

# Loop through each image
for image in images:

    # Get the base name of the image, create output filename
    output_image = output_dir + os.path.basename(image)
    output_image = output_image.replace(".ARW", ".tiff")

    # Read in the RAW file as numpy array
    with rawpy.imread(image) as raw:
        rgb = raw.postprocess()

    # Resize image to 10MB (2159 X 1619), save as .tiff
    rgb = cv2.resize(rgb, (2159, 1619))
    imageio.imsave(output_image, rgb)

    if os.path.exists(output_image):
        print("Saved:", output_image)
    else:
        raise FileNotFoundError("Issue with creating: ", output_image)

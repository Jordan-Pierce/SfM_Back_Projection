import argparse
import os.path

import rasterio
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from utils import *
import Metashape as M
from PySide2 import QtWidgets

# Get the Metashape License stored in the environmental variable
M.License().activate(os.getenv("METASHAPE_LICENSE"))

# Checking compatibility
compatible_major_version = "2.0"
found_major_version = ".".join(M.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


def combine_mosaic_and_labels(args, app):
    """Combine VISCORE Orthomosaic and Labels."""

    def update_console():
        """This function updates the console in Metashape when applicable."""
        if app is not None:
            app.processEvents()

    # Create the output path
    output_path = args.orthomosaic_path.replace(".tif", "_combined.tif")

    # Load VISCORE mosaic
    print("NOTE: Loading VISCORE Orthomosaic...")
    update_console()

    with rasterio.open(args.orthomosaic_path) as src:
        mosaic = src.read()
        profile = src.profile

    # Load VISCORE labels
    print("NOTE: Loading VISCORE Labels...")
    update_console()

    image = Image.open(args.labels_path)
    labels = np.array(image)

    # Check that the dimensions match
    if mosaic.shape[1:] != labels.shape[:2]:
        raise ValueError("ERROR: Dimensions don't match")

    # Copy the metadata of the VISCORE mosaic
    print("NOTE: Combining VISCORE Orthomosaic and Labels...")
    update_console()

    new_profile = profile.copy()
    new_profile.update(count=6, dtype=rasterio.uint8)

    # Create a new array, carrying over the values of the VISCORE mosaic and labels
    new_mosaic = np.zeros(shape=(6, mosaic.shape[1], mosaic.shape[2]), dtype=np.uint8)
    new_mosaic[0, :, :] = mosaic[0, :, :]
    new_mosaic[1, :, :] = mosaic[1, :, :]
    new_mosaic[2, :, :] = mosaic[2, :, :]
    new_mosaic[3, :, :] = labels[:, :, 0]
    new_mosaic[4, :, :] = labels[:, :, 1]
    new_mosaic[5, :, :] = labels[:, :, 2]

    # Write the new array to the new file
    print("NOTE: Writing combined VISCORE Orthomosaic and Labels...")
    update_console()

    with rasterio.open(output_path, 'w', **new_profile) as dst:
        dst.write(new_mosaic)

    if os.path.exists(output_path):
        print("SUCCESS: Combined VISCORE Orthomosaic and Labels.")
    else:
        print("ERROR: Failed to combine VISCORE Orthomosaic and Labels.")


# ----------------------------------------------------------------------------------------------------------------------
# Commandline Interface
# ----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Combine VISCORE Orthomosaic and Labels.')

parser.add_argument('--orthomosaic_path', type=str, help='Path to the VISCORE orthomosaic.')
parser.add_argument('--labels_path', type=str, help='Path to the VISCORE labels.')

args = parser.parse_args()
app = QtWidgets.QApplication.instance()

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    combine_mosaic_and_labels(args, app)
    print("Done.")

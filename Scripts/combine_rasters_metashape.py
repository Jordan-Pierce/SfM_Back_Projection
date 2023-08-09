import os.path
import argparse

import rasterio
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from combine_rasters import combine_mosaic_and_labels

import Metashape as M
from PySide2 import QtWidgets

# Get the Metashape License stored in the environmental variable
M.License().activate(os.getenv("METASHAPE_LICENSE"))

# Checking compatibility
compatible_major_version = "2.0"
found_major_version = ".".join(M.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


# ----------------------------------------------------------------------------------------------------------------------
# Prompting the User in Metashape
# ----------------------------------------------------------------------------------------------------------------------
# "orthomosaic_path": The path to the VISCORE orthomosaic.
# "labels_path": The path to the VISCORE labels.

def main():

    # Create an empty argparser to pas to workflow
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Get the path to the Orthomosaic
    print("SELECT: This is the ORTHOMOSAIC created in VISCORE.")
    prompt = "Select the ORTHOMOSAIC: "
    args.orthomosaic_path = M.app.getOpenFileName(prompt)

    # Get the path to the Elevation (DEM)
    print("SELECT: This is the LABELS created in TagLab.")
    prompt = "Select the Labels: "
    args.labels_path = M.app.getOpenFileName(prompt)

    # Create the app to update console
    app = QtWidgets.QApplication.instance()

    # Run the combine_viscore_mosaic_and_labels function
    combine_mosaic_and_labels(args, app)


label = "Combine Rasters (Metashape)"
M.app.addMenuItem(label, main)
print(f"NOTE: Select the VISCORE Orthomosaic and Labels: {label}")

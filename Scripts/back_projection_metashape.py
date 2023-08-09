import os
import sys
import time
import argparse

from back_projection import back_projection_workflow

import Metashape as M
from PySide2 import QtWidgets

# Checking compatibility
compatible_major_version = "2.0"
found_major_version = ".".join(M.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

# ----------------------------------------------------------------------------------------------------------------------
# Prompting the User in Metashape
# ----------------------------------------------------------------------------------------------------------------------
# "doc_path": Path to the Metashape .psx file; ignored in metashape
# "chunk_number": The chunk to work on; 0-based indexing
# "labels_path": The path to the labeled orthomosaic (.tif, .png)
# "elevation_path": The path to the elevation (DEM, .tif)
# "orthomosaic_path": The path to the orthomosaic (.tif)
# "taglab_path": The path to the TagLab project (.json)
# "output_path": The output folder to store data products
# "classified_point_path": The output path of the classified point cloud (.ply)
# "labeled_frac": The % [0 - 1] of labeled pixels to sample from labeled orthomosaic
# "unlabeled_frac": The % [0 - 1] of unlabeled pixels to sample from labeled orthomosaic
# "dense_pcd_frac": The % [0 - 1] of dense point cloud to sample
# "model_type": Classifier ('rf', 'mlp', 'knn', 'svm')

def main():

    # Create an empty argparser to pas to workflow
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Set to the currently active chunk
    chunks = M.app.document.chunks
    try:
        args.chunk_number = [i for i in range(len(chunks)) if chunks[i].selected][0]
    except:
        print("ERROR: You must click on the desired chunk first to select it!")
        return

    # Get the path to the Orthomosaic
    print("SELECT: This is the ORTHOMOSAIC created in Metashape / VISCORE.")
    prompt = "Select the ORTHOMOSAIC: "
    args.orthomosaic_path = M.app.getOpenFileName(prompt)

    # Get the path to the Elevation (DEM)
    print("SELECT: This is the ELEVATION (DEM) created in Metashape.")
    prompt = "Select the ELEVATION (DEM): "
    args.elevation_path = M.app.getOpenFileName(prompt)

    # Get the path to the Labeled Orthomosaic
    print("SELECT: This is the LABELED ORTHOMOSAIC created in TagLab.")
    prompt = "Select the LABELED ORTHOMOSAIC: "
    args.labels_path = M.app.getOpenFileName(prompt)

    # Get the path to the TagLab JSON file
    print("SELECT: This is the TagLab Project JSON created by TagLab.")
    prompt = "Select the TagLab JSON file: "
    args.taglab_json = M.app.getOpenFileName(prompt)

    # Ask where to place output data
    print("SELECT: This is the FOLDER where all OUTPUT should be saved.")
    prompt = "Select the OUTPUT FOLDER: "
    args.output_path = M.app.getExistingDirectory(prompt) + "/"

    # Set sample fraction of labeled pixels
    print("SELECT: Percentage of labeled pixels to sample from labeled orthomosaic.")
    prompt = 'Enter a value between 0 and 1'
    args.labeled_frac = M.app.getFloat(value=0.01, label=prompt)

    # Set sample fraction of unlabeled pixels
    print("SELECT: Percentage of unlabeled pixels to sample from labeled orthomosaic.")
    prompt = 'Enter a value between 0 and 1'
    args.unlabeled_frac = M.app.getFloat(value=0.001, label=prompt)

    # Set the sample fraction for dense point cloud
    print("SELECT: Percentage of dense point cloud to sample.")
    prompt = 'Enter a value between 0 and 1'
    args.dense_pcd_frac = M.app.getFloat(value=0.1, label=prompt)

    # Set model type
    print("SELECT: This is model type (RandomForest, Neural Network, KNearestNeighbor).")
    prompt = 'One of: rf, mlp, knn'
    args.model_type = M.app.getString(value='rf', label=prompt)

    # Setting where to save output files
    args.model_path = args.output_path + args.model_type + "_model.joblib"
    args.dense_point_path = args.output_path + "Dense_Point_Cloud.ply"
    args.training_point_path = args.output_path + "Training_Point_Cloud.ply"
    args.classified_point_path = args.output_path + "Classified_Point_Cloud.ply"

    # Ignored when used with Metashape GUI
    args.doc_path = None

    # Create the app to update console
    app = QtWidgets.QApplication.instance()

    t0 = time.time()
    back_projection_workflow(args, app)
    print("Total time: ", (time.time() - t0) / 60, " minutes!")
    print("Done.")


label = "Back Projection (Metashape)"
M.app.addMenuItem(label, main)
print(f"NOTE: Select the desired chunk, and then run: {label}")

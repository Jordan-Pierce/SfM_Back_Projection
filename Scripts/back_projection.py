import os
import sys
import time
import joblib
import argparse

import numpy as np
from tqdm import tqdm

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


def back_projection_workflow(args, app=None):
    """This function takes is a set of arguments from a commandline and runs the back
    projection workflow."""

    def update_console():
        """This function updates the console in Metashape when applicable."""
        if app is not None:
            app.processEvents()

    # Set the Metashape document, open it.
    if args.doc_path is None:
        doc = M.app.document
    else:
        doc = M.Document()
        doc.open(args.doc_path, True, True)

    # Set the output paths
    os.makedirs(args.output_path, exist_ok=True)
    model_path = args.output_path + args.model_type + "_model.joblib"
    dense_point_path = args.output_path + "Dense_Point_Cloud.ply"
    training_point_path = args.output_path + "Training_Point_Cloud.ply"
    classified_point_path = args.output_path + "Classified_Point_Cloud.ply"

    # A dict holding shared information from metashape.
    m_dict = {'chunk': doc.chunks[args.chunk_number],
              'orthomosaic': doc.chunk.orthomosaic,
              'elevation': doc.chunk.elevation}

    print("Chunk Name: ", m_dict['chunk'].label)
    print("Coordinate System: ", m_dict['chunk'].crs)
    update_console()

    # Open the elevation (DEM), orthomosaic, and labeled orthomosaic provided.
    try:
        # Open the DEM
        print("NOTE: Loading Elevation...")
        update_console()
        elevation = imread(args.elevation_path)

        # Open the orthomosaic, keep only the RGB channels, excluding alpha
        print("NOTE: Loading Orthomosaic...")
        update_console()
        orthomosaic = imread(args.orthomosaic_path)[:, :, 0:3]

        # Open the labeled orthomosaic, keepy only the RGB channels, excluding alpha
        print("NOTE: Loading Labeled Orthomosaic...")
        update_console()
        labels = imread(args.labels_path)[:, :, 0:3]

    except Exception as e:
        print("ERROR: Unable to open elevation, orthomosaic, or labeled orthomosaic!\n\n\n", e)

    try:

        # Ensure that the orthomosaic within the Metashape project have the same dimensions
        # of the orthomosaic being imported into this script.
        print("NOTE: Metashape Orthomoasic (", m_dict['orthomosaic'].height, m_dict['orthomosaic'].width, ", 3)")
        print("NOTE: Imported Orthomosaic ", orthomosaic.shape)
        print("NOTE: Labeled Orthomosiac ", labels.shape)
        update_console()

        h_dif = orthomosaic.shape[0] == m_dict['orthomosaic'].height == labels.shape[0]
        w_dif = orthomosaic.shape[1] == m_dict['orthomosaic'].width == labels.shape[1]

        ORTHOMOSAIC_DIMENSION_CHECK = h_dif and w_dif

        print("NOTE: Metashape Elevation (", m_dict['elevation'].height, m_dict['elevation'].width, ")")
        print("NOTE: Imported Elevation ", elevation.shape)
        update_console()

        h_dif = elevation.shape[0] == m_dict['elevation'].height
        w_dif = elevation.shape[1] == m_dict['elevation'].width

        ELEVATION_DIMENSION_CHECK = h_dif and w_dif

        if ORTHOMOSAIC_DIMENSION_CHECK != True or ELEVATION_DIMENSION_CHECK != True:
            raise ValueError

    except Exception as e:
        print("ERROR: Imported Orthomosaic and Labeled Orthomosaic must match the dimensions of Metashape Orthomosaic!")
        print("ERROR: Imported Elevation must match the dimensions of Metashape Elevation!\n\n\n", e)
        return False

    # Export mesh from metashape as point cloud, import using open3d.
    # Extract the points (XYZ) and colors (RGB) and store in variables.
    print("NOTE: Loading dense point cloud...")
    update_console()
    original_o3d = get_metashape_point_cloud(dense_point_path, m_dict)
    original_o3d = original_o3d.random_down_sample(args.dense_pcd_frac)

    pc_pts = np.array(original_o3d.points)
    pc_rgb = np.array(np.array(original_o3d.colors) * 255).astype(np.uint8)

    # Find the labels used in the TagLab project (.json) and store
    # the class names and corresponding color values in variables
    print("NOTE: Calculating class colors...")
    update_console()
    class_colors, class_names, class_indices = get_class_colors(args.taglab_json)

    # Sample the orthomosaic, getting the yxzrgbl for each index
    print("NOTE: Sampling labeled orthomosaic...")
    update_console()
    pixels = get_samples(orthomosaic, labels, args.labeled_frac, args.unlabeled_frac)

    # Given the pixels, project from 2D orthomosaic space into 3D coordinates
    print("NOTE: Projecting points...")
    update_console()
    labeled_pts, labeled_labs, valid_indices = project_pixels(pixels, elevation, m_dict)
    labeled_conf = np.full(shape=len(labeled_pts), fill_value=100).astype(int)

    # Grab the YXZ and labels for each projected point; calculate the color
    # of each point from the orthomosaic.
    print("NOTE: Getting point colors...")
    update_console()
    valid_y = pixels.T[0][valid_indices]
    valid_x = pixels.T[1][valid_indices]
    labeled_rgb = orthomosaic[valid_y.astype(int), valid_x.astype(int)]

    # Store in an open3d point cloud object
    labeled_o3d = o3d.geometry.PointCloud()
    labeled_o3d.points = o3d.utility.Vector3dVector(labeled_pts)
    labeled_o3d.colors = o3d.utility.Vector3dVector(labeled_labs / 255.0)

    # Sanity check ensuring that the coordinates are roughly the same.
    # If these are not close (within 1.0 unit), then the dimensions of the
    # Metashape and Imported are different and therefore incorrect.
    print("NOTE: Dense point cloud ", pc_pts.shape, pc_pts.mean(axis=0))
    print("NOTE: Labeled point cloud ", labeled_pts.shape, labeled_pts.mean(axis=0))

    # Calculate additional features for each of the point clouds.
    print("NOTE: Calculating features...")
    update_console()
    pc_norms = calculate_features(original_o3d)
    labeled_norms = calculate_features(labeled_o3d)

    # Stack the points, colors, and features, then create a training,
    # validation and testing sets.
    print("NOTE: Preparing training, testing data...")
    update_console()
    train_x = np.column_stack((labeled_pts, labeled_rgb, labeled_norms))
    train_y = map_labels(labeled_labs, class_colors)
    test_x = np.column_stack((pc_pts, pc_rgb, pc_norms))

    # Train a ML model using the training set
    print("NOTE: Training model...")
    update_console()
    model = create_classifier(args.model_type)
    model.fit(train_x, train_y)

    # Make predictions using the trained model on all points in the original
    # cloud. Get the class predictions, and corresponding confidence values
    # (these need to be scaled to [0 - 255] for Metashape). Convert the
    # predictions to label color values.
    print("\nNOTE: Making predictions...")
    update_console()

    # Define the batch size for predictions; if the point cloud is very large
    # many users will have OOM errors, so reduce batch size, aggregate results.
    batch_size = 10000000

    # Make predictions in batches
    probabilities = []

    for i in tqdm(range(0, len(test_x), batch_size)):
        x_batch = test_x[i:i + batch_size]
        batch_predictions = model.predict_proba(x_batch)
        probabilities.extend(batch_predictions)
        update_console()

    # Create an array, extract top-1, get label colors, store probabilities,
    # convert top-1 to scientific names.
    probabilities = np.array(probabilities)
    test_y = np.argmax(probabilities, axis=1)
    pc_labs = np.array([class_colors[_] for _ in test_y])
    pc_conf = (np.max(probabilities, axis=1) * 100).astype(np.uint8)

    # Writing training point cloud to disk
    print("NOTE: Writing training point cloud to disk...")
    update_console()
    write_ply(labeled_pts,
              labeled_rgb,
              labeled_norms,
              labeled_labs,
              labeled_conf,
              train_y,
              training_point_path)

    # Combine the training and classified point cloud together
    print("NOTE: Combining point clouds...")
    update_console()
    combined_pc = np.row_stack((pc_pts, labeled_pts))
    combined_rgb = np.row_stack((pc_rgb, labeled_rgb))
    combined_labs = np.row_stack((pc_labs, labeled_labs))
    combined_norms = np.row_stack((pc_norms, labeled_norms))
    combined_conf = np.concatenate((pc_conf, labeled_conf))
    combined_indices = np.concatenate((test_y, train_y))

    # Write combined point cloud to disk
    print("NOTE: Writing classified point cloud to disk...")
    update_console()
    write_ply(combined_pc,
              combined_rgb,
              combined_norms,
              combined_labs,
              combined_conf,
              combined_indices,
              classified_point_path)

    # Export the classifier to disk
    print("NOTE: Exporting classifier to disk...")
    update_console()
    joblib.dump(model, model_path)

    # Import the point clouds into Metashape
    print("NOTE: Importing point clouds into Metashape...")
    update_console()
    # Importing the training point cloud
    m_dict['chunk'].importPointCloud(training_point_path,
                                     crs=m_dict['chunk'].crs)

    # Importing the classified point cloud
    m_dict['chunk'].importPointCloud(classified_point_path,
                                     crs=m_dict['chunk'].crs)


# ----------------------------------------------------------------------------------------------------------------------
# Commandline Interface
# ----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='SfM Back Projection Workflow.')

parser.add_argument('--doc_path', type=str,
                    default="B://MIR/EDR_R5-3/Project/EDR_R5-3_2021-12-08.psx",
                    help='Path to the document')

parser.add_argument('--chunk_number', type=int,
                    default=2,
                    help='The chunk within the Metashape project to use; 0-based indexing.')

parser.add_argument('--labels_path', type=str,
                    default="B://MIR/EDR_R5-3/Output/labels.png",
                    help='Path to the labels file')

parser.add_argument('--elevation_path', type=str,
                    default="B://MIR/EDR_R5-3/Output/elevation.tif",
                    help='Path to the elevation file')

parser.add_argument('--orthomosaic_path', type=str,
                    default="B://MIR/EDR_R5-3/Output/orthomosaic.tif",
                    help='Path to the orthomosaic file')

parser.add_argument('--taglab_json', type=str,
                    default="B://MIR/EDR_R5-3/Output/EDR_R5-3_2023-03-22.json",
                    help='Path to the taglab JSON file')

parser.add_argument('--output_path', type=str,
                    default="B://MIR/EDR_R5-3/Output/",
                    help='Path to output the data from workflow.')

parser.add_argument('--labeled_frac', type=float, default=0.01,
                    help='Percentage of labeled pixels to be sampled from the labeled orthomosaic')

parser.add_argument('--unlabeled_frac', type=float, default=0.001,
                    help='Percentage of unlabeled pixels to be sampled from the labeled orthomosaic')

parser.add_argument('--dense_pcd_frac', type=float, default=0.25,
                    help='Percentage of the dense point cloud to be sampled')

parser.add_argument('--model_type', type=str, default='knn',
                    help='Type of model to use')

args = parser.parse_args()
app = QtWidgets.QApplication.instance()

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()
    back_projection_workflow(args, app)
    print("Total time: ", (time.time() - t0) / 60, " minutes!")
    print("Done.")

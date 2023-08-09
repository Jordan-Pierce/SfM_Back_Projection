import os
import sys
import json
import struct

import numpy as np
import open3d as o3d
from skimage.io import imread
import matplotlib.pyplot as plt
from plyfile import PlyElement, PlyData

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from skimage.transform import resize


def get_class_colors(taglab_json):
    """Find all the unique values in JSON file output by TagLab."""

    # Define a function to search for values
    def find_class_name(data):

        class_names = []
        # check if the data is a dictionary
        if isinstance(data, dict):
            for key, value in data.items():
                # check if the key is "class name"
                if key == "class name":
                    class_names.append(value)
                else:
                    # if it's not a "class name" key, call the function
                    # recursively
                    class_names += find_class_name(value)
        # check if the data is a list
        elif isinstance(data, list):
            for item in data:
                # call the function recursively
                class_names += find_class_name(item)
        return list(set(class_names))

    # Read the JSON file provided by TagLab
    with open(taglab_json) as f:
        taglab_meta = json.load(f)

    class_names = find_class_name(taglab_meta)
    class_colors = [taglab_meta['labels'][cn]['fill'] for cn in class_names]

    # Add the unlabeled class category
    class_names.insert(0, "Unlabeled")
    class_colors.insert(0, [0, 0, 0])

    # Create a dictionary representing class colors
    class_colors = dict(zip(range(len(class_names)), class_colors))
    class_names = dict(zip(class_names, list(class_colors.values())))
    class_indices = {i: k for i, k in enumerate(class_names.keys())}

    # Creating a label mapping for posterity
    label_mapping = {"TagLab_File": os.path.basename(taglab_json),
                     "Class_Categories": {}}

    for class_index in list(class_indices.keys()):
        label_mapping['Class_Categories'][class_index] = {"Class_Category": class_indices[class_index],
                                                          "Class_Color": class_colors[class_index]}
    # Writing label mapping as json
    label_mapping_path = taglab_json.replace(".json", "_") + "Label_Mapping.json"
    with open(label_mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=4)

    print(f"NOTE: Classes found {len(class_names)}")

    return class_colors, class_names, class_indices


def get_metashape_point_cloud(path, m_dict):
    """Exports point cloud from Metashape, reads it with open3d, returns."""

    # Only create if it doesn't already exists
    if not os.path.exists(path):
        m_dict['chunk'].exportPointCloud(path=path,
                                         save_point_color=True,
                                         save_point_classification=True,
                                         save_point_normal=True,
                                         save_point_confidence=True,
                                         crs=m_dict['chunk'].crs)

    point_cloud = o3d.io.read_point_cloud(path)

    if point_cloud.is_empty():
        print("ERROR: Exported Metashape point cloud is empty!")
        sys.exit()

    return point_cloud


def get_samples(orthomosaic, labels, labeled_frac, unlabled_frac):
    """This function takes in the labeled orthomosaic and returns a sample of
    y, x, pixels and their corresponding labels."""

    # Find all the pixels in the orthmosaic that are not white, which are
    # valid data pixels.
    white = np.all(orthomosaic == [255, 255, 255], axis=-1)
    y, x = np.where(~white)

    # From the labeled orthomosaic, return only a sample of colored labels
    clr_i = np.where(np.any(labels[y, x] != [0, 0, 0], axis=-1))[0]
    size = int(len(clr_i) * labeled_frac)
    clr_i = np.random.choice(clr_i, size=size, replace=False)
    clr_y, clr_x = y[clr_i], x[clr_i]
    clr_l = labels[y, x][clr_i]

    # From the labeled orthomosaic, return only a sample of black labels
    blk_i = np.where(np.all(labels[y, x] == [0, 0, 0], axis=-1))[0]
    size = int(len(blk_i) * unlabled_frac)
    blk_i = np.random.choice(blk_i, size=size, replace=False)
    blk_y, blk_x = y[blk_i], x[blk_i]
    blk_l = labels[y, x][blk_i]
    print(f"NOTE: Labeled pixels - {len(clr_i)}")
    print(f"NOTE: Unlabeled pixels - {len(blk_i)}")

    blk = np.column_stack((blk_y, blk_x, blk_l))
    clr = np.column_stack((clr_y, clr_x, clr_l))

    return np.row_stack((blk, clr))


def project_pixels(pixels, elevation, m_dict):
    """This function takes in the sample pixels from the (labeled) orthomosaic, the elevation,
    and the Metashape dictionary; it returns the 3D locations of the pixels within the labeled orthomosaic."""

    ortho = m_dict['orthomosaic']
    dem = m_dict['elevation']

    y, x, R, G, B = pixels.T

    # Converting the pixels indices in the orthomosaic to 3D locations.
    x_3d = ortho.left + ortho.resolution * x.astype(int)
    y_3d = ortho.top - ortho.resolution * y.astype(int)

    # Map the pixel indices in the orthomosaic to the corresponding indices in the DEM
    x_dem = np.round((x_3d - dem.left) / dem.resolution).astype(int)
    y_dem = np.round((dem.top - y_3d) / dem.resolution).astype(int)

    # Plot the points sampled and DEM to check that the mapping is correct
    if False:
        plt.figure()
        plt.scatter(x_dem, y_dem, s=0.1)
        plt.imshow(elevation)
        plt.show()

    # Get the z coordinates for all pixels in the orthomosaic, from the DEM.
    # Indices that are not valid will be removed.
    z_3d = elevation[y_dem, x_dem]
    indices = np.where(z_3d > -32767)[0]

    # Keep only the valid indices
    X = x_3d[indices]
    Y = y_3d[indices]
    Z = z_3d[indices]
    R = R[indices]
    G = G[indices]
    B = B[indices]

    # Stack and return XYZ, RGB
    xyz = np.column_stack((X, Y, Z))
    rgb = np.column_stack((R.astype(int), G.astype(int), B.astype(int)))

    return xyz, rgb, indices


def calculate_features(point_cloud_o3d):
    """Calculate features from a point cloud to be using in training.
    Currently the features include: Normals."""

    # Calculate features (normals) for both point clouds; set search
    # parameters based on properties of the point cloud later. Low values for
    # radius result in less accurate models, but high values take longer...
    search_params = o3d.geometry.KDTreeSearchParamHybrid(radius=10,
                                                         max_nn=100)

    # If the point cloud already has normals, grab them, else create them.
    if not point_cloud_o3d.has_normals():
        point_cloud_o3d.estimate_normals(search_param=search_params)
        normals = np.array(point_cloud_o3d.normals)
    else:
        normals = np.array(point_cloud_o3d.normals)

    return normals


def map_labels(labels, class_colors):
    """This function maps RGB values representing the label color,
    to a single integer value that serves as the training label"""

    m = [list(class_colors.values()).index(l.tolist()) for l in labels]
    m = np.array(m)

    return m


def create_classifier(model_type):
    """Returns a scikit-learn classifier based on the user's input"""

    # Random Forest classifier
    if model_type == "rf":
        clf = RandomForestClassifier(n_estimators=100,
                                     warm_start=True,
                                     n_jobs=-1,
                                     verbose=True)

    # Support Vector Machine classifier with a radial basis function kernel
    elif model_type == "svm":
        clf = SVC(kernel='rbf',
                  gamma='auto',
                  C=1.0,
                  decision_function_shape='ovr',
                  verbose=True)

    # MLP classifier
    elif model_type == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(100, 2),
                            activation='relu',
                            solver='adam',
                            max_iter=100,
                            verbose=True,
                            warm_start=True,
                            early_stopping=True,
                            validation_fraction=0.3,
                            tol=1e-4)

    # K-Nearest Neighbors classifier
    elif model_type == "knn":

        clf = KNeighborsClassifier(n_neighbors=100,
                                   n_jobs=-1)

    else:
        raise ValueError("Invalid model_type")

    return clf


def write_ply(points, colors, normals, labels, confidences, indices, output_file):
    """Write the classified point cloud as a .ply file"""

    dt = np.dtype([('x', 'f4'),
                   ('y', 'f4'),
                   ('z', 'f4'),
                   ('r', 'u1'),
                   ('g', 'u1'),
                   ('b', 'u1'),
                   ('nx', 'f4'),
                   ('ny', 'f4'),
                   ('nz', 'f4'),
                   ('red', 'u1'),
                   ('green', 'u1'),
                   ('blue', 'u1'),
                   ('confidence', 'u1'),
                   ('class', 'u1')])

    point_cloud = np.empty(len(points), dtype=dt)

    point_cloud['x'] = points.T[0]
    point_cloud['y'] = points.T[1]
    point_cloud['z'] = points.T[2]
    point_cloud['r'] = colors.T[0]
    point_cloud['g'] = colors.T[0]
    point_cloud['b'] = colors.T[0]
    point_cloud['nx'] = normals.T[0]
    point_cloud['ny'] = normals.T[1]
    point_cloud['nz'] = normals.T[2]
    point_cloud['red'] = labels.T[0]
    point_cloud['green'] = labels.T[1]
    point_cloud['blue'] = labels.T[2]
    point_cloud['confidence'] = confidences
    point_cloud['class'] = indices

    element = PlyElement.describe(point_cloud, 'vertex')
    PlyData([element]).write(output_file)

    if os.path.exists(output_file):
        print(f"File created successfully: {output_file}")
    else:
        print(f"Could not create file: {output_file}")

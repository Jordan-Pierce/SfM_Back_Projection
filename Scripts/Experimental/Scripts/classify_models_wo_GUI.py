import os

import Metashape as M
M.License().activate(os.getenv('METASHAPE_LICENSE'))


# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------
CWD = "C:/Users/jordan.pierce/Documents/GitHub/SfM_Back_Projection"
DOC_PATH = CWD + "/Project/back_project.psx"
IMAGE_DIR = CWD + "/Project/Images/Tif/"
LABEL_DIR = CWD + "/Project/Images/Dense/0/"

# Set the Metashape document, open it.
doc = M.Document()
doc.open(DOC_PATH)

# Ask where the images are
prompt = "Enter the Directory Containing Images: "
# image_dir = M.app.getExistingDirectory(prompt) + "/"

# Ask where to place the labeled images
prompt = "Enter the Directory Containing Labels: "
# output_dir = M.app.getExistingDirectory(prompt) + "/"

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Get a list of chunks
    chunks = doc.chunks
    classified_chunk = None

    # Search to see if there is already a classified chunk
    for chunk in chunks:
        if chunk.label == "Classified":
            classified_chunk = chunk
            break

    # If not, create it, name it, and add it to the document
    if classified_chunk is None:
        classified_chunk = chunks[0].copy()
        classified_chunk.label = "Classified"

    # Update all the photo paths in the classified chunk to be the labels
    for camera in classified_chunk.cameras:
        classified_photo = LABEL_DIR + os.path.basename(camera.photo.path)
        camera.photo.path = classified_photo[:-1]

    # Save the document
    doc.save()

    # Classify (colorize) the dense point cloud using the labels.
    # Update the point cloud to apply the new colorization settings
    classified_chunk.colorizeDenseCloud(M.ImagesData)
    doc.save()

    # Edit dense point cloud colors

    # Classify (colorize) the mesh using the classified dense point cloud.
    # Update the mesh to apply the new colorization settings
    classified_chunk.colorizeModel(M.DenseCloudData)
    doc.save()

    print("Done.")
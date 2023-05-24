### First Release
The first release of the SfM Back Projection Workflow! This release includes the ability to do all that is specified in the readme, but in summary:
- A `setup.py` script that will install all necessary dependencies
- Project labels from labeled orthomosaics made in TagLab to create a labeled sparse point cloud (used for training)
- Train a machine learning algorithm (SciKit-Learn's KNN, RandomForest, MLP) using the labeled sparse point cloud
- Classify the dense point cloud using the trained model
- Export trained model (`.joblib` file)
- Tutorials to help walk a user through the entire process

### Future Releases
In future iterations we hope to:
- Add a deep learning algorithm for point cloud semantic segmentation
- Have to ability to back project labels from the classified point cloud / mesh to create segmentation masks for each image
- Update how the user provides parameters for the script within Metashape (single panel)

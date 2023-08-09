### Classifying 3D Models by Back-Projecting and Projecting Labels

This repository is used classify 3D models (point clouds, meshes) 
created in Agisoft Metashape. The process is described below:

#### 1.) Back-Projecting Sparse Labels 

Labels from a labeled othromosaic (i.e., TagLab) are back-projected to source images, creating 
sparse labels (see below).  

<p align="center">
  <img src="./Figures/back_projection.gif" alt="">
</p>

#### 2.) Converting Sparse Labels to Dense Labels using an Autoencoder

An autoencoder (e.g., UNet) is trained to perform semantic segmentation by being trained 
directly on the sparse labels. Once trained, the model **adds** to the sparse labels, which are 
then used to train the next generation of model. This process continues until the user is 
satisfied with the resulting dense labels.

<p align="center">
  <img src="./Figures/autoencoder.gif" alt="">
</p>

#### 3.) Projecting Dense Labels

The last step involves projecting the dense labels on to the dense point cloud and mesh to 
created classified versions. This process is done entirely in Metashape.

<p align="center">
  <img src="./Figures/mesh_rotation.gif" alt="">
</p>


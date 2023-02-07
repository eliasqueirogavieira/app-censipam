
# App Censipam

This project provides a Python environment that downloads SAR images (Sentinel-1 satellite) through zip files, pre-process it using SNAP, post-process it using SAR enhancing techniques and then feed it to a pre-trained CNN ([U-Net](https://github.com/eliasqueirogavieira/unet-sentinel) implemented) to detect newer deforestation in a georeferenced (CRS - WGS84) TIFF images.

This application performs the task of classifying the same region on different dates, usually separated by months, then performs the difference between the two and saves the output in a shapefile (.shp), using the same CRS as the original TIFF image, so it is compatible with QGIS, PostGIS, ArcGIS, etc.

```bash
  pred.shp = pred_after.shp - pred_before.shp
```
### 1 - Install Anaconda

Install conda as instructed in [Anaconda](https://docs.anaconda.com/anaconda/install/linux/).

Download the installer

```bash
    wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
```

Run the installer. You will be asked to provide a installation folder,  for instance: **$HOME/conda**

```bash
    bash Anaconda3-2022.10-Linux-x86_64.sh
```

Or install conda from the mini version already downloaded here

```bash
    bash installation/Miniconda3-py39_4.11.0-Linux-x86_64.sh
```

In order to run conda commands in a terminal, save the conda installation path to **$PATH** environment variable. After that, you may need to close and reopen the terminal.


### 2 - Create the environment using conda

For convience all dependecies are provided in the file [**censipam.yml**](./app/censipam.yml). To create the environment with all packages needed to run the application,  run the following command:

```bash
    conda env create -f app/installation/censipam.yml
```

Once the environment is created, you may need to activate the environment:

```bash
    conda activate
```

Otherwise you will need to call the python by the full path like ~/conda/envs/<env_created>/bin/python

### 3 - Install ESA-SNAP

The application requires the installation of the ESA-SNAP for pre-processing the SAR images. The ESA-SNAP may be downloaded from [ESA-SNAP](https://step.esa.int/main/download/snap-download/) or if the link is still valid:

** You might need the get the link to latest version at the ESA-SNAP website.

```bash
    wget https://download.esa.int/step/snap/8.0/installers/esa-snap_all_unix_8_0.sh
```

Run the installer

```bash
    bash esa-snap_all_unix_8_0.sh
```

Stick with default installation options


### 3 - Clone the application repository


Clone the application repository. You need only the **/app** folder

```bash
    git clone https://github.com/eliasqueirogavieira/app-censipam
```

#### 3-1 - Additional package

You also must to install the package inside the folder *packages/segmentation/*. To do that, go the folder and execute the following command:

```bash
    conda activate <conda_env_created> # <conda_env_created> the conda env created previously
    pip install -e .
```



### 4 - Running the App - Sentinel data


Go the folder /appa and execute the application by runnning:

```bash
    ~/conda/envs/<env_created>/bin/python main_prediction.py \
                --data_path <folder>
                --root_output <folder>
                --config configs/cfg_sentinel.yml
```

* **data_path**: is the root folder with *.zip files of the scenes

* **root_output**: is the root folder where to save the processing results and the serialized object that keeps track of what have been processed. This folder will consume considerable disk space, choose thougly.

* **config**: this is a file with several settings, take a look at the file to see its content. Make sure the files indicated there exist and are properly set up. In **config** there are definitions for ESA-SNAP processing like the *xml* graph files and the model settings.

Depending on the number of *.zip* files encountered in *data_path*, the execution will take a while. It might be better to first pass a **data_path** with a few files to test. The application will: i) process each zip file with ESA-SNAP; ii) find out the image pairs; iii) process the pairs together with ESA-SNAP; and iv) execute the learned model to generate the predictions.


#### 4-1 Structure of the *root_ouput folder*

The **root_ouput folder** will be have:
    
  * **partial_sentinel**: keeps a folder for each zip file processed. Each folder's name is the name of the zip found
    
  * **app_results**: keeps a folder for each image pair found and processed. Each folder's name is a combination of image pair. 
      * **stacked.tif**: (VV1, VH1, VV2, VH2) stacked image 
      * **pred.shp**: shapefile with prediction

  * **sentinel_obj**: a serialized object to keep track of what have been processed




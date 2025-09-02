![Logo](logo/GBA-UBF.png)

# GBA-UBF
This is the Dataset of the **Greater Bay Area Urban Building Function** (**GBA-UBF**) proposed by our paper "A Large-Scale and Fine-Grained Building Function Classification Dataset in the Greater Bay Area".

**GBA-UBF** a building-level function dataset for **the Guangdong–Hong Kong–Macao Greater Bay Area**. GBA-UBF provides **city-wide**, **building-scale** labels that overcome the coarse granularity of parcel maps. It covers 4 million buildings across six core GBA cities (**Guangzhou**, **Shenzhen**, **Zhongshan**, **Zhuhai**, **Foshan**, **HongKong SAR**) with five unified classes: **Residential**, **Commercial**, **Public Services**, **Technology and Industry**, **Educational and Cultural**.

## GBA-UBF Preview
<p align="center">
  <img src="pictures/overallshow.png" alt="GBA-UBF Preview" width="900"/><br>
  <em>Figure 1. A preview of the GBA-UBF dataset.</em>
</p>

<p align="center">
  <img src="pictures/3d.png" alt="GBA-UBF 3D Visualisation" width="800"/><br>
  <em>Figure 2. Representative GBA-UBF Dataset 3D Visualization and Corresponding Shapefile Overlay with High-Resolution Remote Sensing Imagery across six cities.</em>
</p>

<p align="center">
  <img src="pictures/com.png" alt="Comparison of GBA-UBF with Existing Land Use Datasets" width="600"/><br>
  <em>Figure 3. The Comparison between the GBA-UBF Dataset and EULUC-China Dataset.</em>
</p>

## Dataset Download
Please enjoy this dataset we have provided. Click [here](demo/) to view one examples of the study aera in Guangzhou.

## Dataset structure
````
dataset
|--GBA-UBF
|  |--Guangzhou_1
|  |  |--Residential_shp
|  |  |--Commercial_shp
|  |  |--PublicServices_shp
|  |  |--TechnologyIndustry_shp
|  |  |--EducationalCultural_shp
|  |--Shenzhen_1
|  |  |--Residential_shp
|  |  |--Commercial_shp
|  |  |--PublicServices_shp
|  |  |--TechnologyIndustry_shp
|  |  |--EducationalCultural_shp
...
````
Each shapefile also contains building height information.

## Organizations
<p align="center">
  <img src="logo/SYSU.png" alt="SYSU" width="300"/>
  <img src="logo/SGES.png" alt="SGES" width="500"/>
</p>

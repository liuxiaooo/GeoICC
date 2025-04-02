# GeoICC
The dataset used for pre-training GeoGPT-2 and GeoViT cannot be publicly released due to confidentiality reasons, but we have provided pre-trained weight files. 
Please unzip 
```
geogpt pre-trained weights.zip
```
into the geogpt folder.
and unzip 
```
geovit pre-trained weights.zip
```
into the geovit folder.
For fine-tuning GeoICC on the image-caption pair dataset, please unzip 
```
imagecaptiondata.zip
```
to 
```
geoicc/bigdata/geoimage_data
```
To set up the environment and install the required libraries, please run the following command:
 
```bash
pip install -r requirements.txt

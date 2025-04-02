# GeoICC
The dataset used for pre-training GeoGPT-2 and GeoViT cannot be publicly released due to confidentiality reasons, but we have provided pre-trained weight files. 
You can download the zip files `geogpt pre-trained weights.zip`, `geovit pre-trained weights.zip` and `imagecaptiondata.zip` from [link2]([https://drive.google.com/open?id=1eCdz62FAVCGogOuNhy87Nmlo5_I0sH2J](https://huggingface.co/iiliya/geovit/resolve/main/geovit%20pre-trained%20weights.zip?download=true)).
Please unzip 
```
geogpt pre-trained weights.zip
```
into the geogpt folder.
Please  unzip 
```
geovit pre-trained weights.zip
```
into the geovit folder.
For fine-tuning GeoICC on the image-caption pair dataset, please unzip 
```
imagecaptiondata.zip
```
into the geoicc folder.
To set up the environment and install the required libraries, please run the following command:
 
```bash
pip install -r requirements.txt

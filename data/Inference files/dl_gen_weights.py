import urllib.request
import pathlib
import os
import zipfile
model_path = pathlib.Path('..','src','Generator Models')

urls = [
'https://github.com/xefonon/SoundFieldGAN/releases/download/GANweights/PlanewaveGAN_G_weights.zip'
]

weights_url = urls[0]
filename = weights_url.split('/')[-1]
print("downloading", filename, "...")
urllib.request.urlretrieve(weights_url, os.path.join(str(model_path), filename))
print("Done.")

# unzip the weights file
with zipfile.ZipFile(os.path.join(str(model_path), filename), 'r') as zip_ref:
    zip_ref.extractall(str(model_path))
print("Done with unzipping.")
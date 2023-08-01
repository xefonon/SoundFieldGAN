import urllib.request
import pathlib
import os

datapath = pathlib.Path("..", "data", "Inference files")

urls = [
  'https://github.com/xefonon/SoundFieldGAN/releases/download/dataset/MeshRIR_set.npz',
]


input_data = urls[0]
filename = input_data.split('/')[-1]
print("downloading", filename, "...")
urllib.request.urlretrieve(input_data, os.path.join(str(datapath), filename))
print("Done.")
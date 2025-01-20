
import glob
import os

from PIL import Image

path = os.getcwd() + '/Dataset/SpeckleNoise/'
print(path)
i = 0
for samplep in glob.glob(path + '/*.jpg'):
    sample = Image.open(samplep)
    if sample.size != (224, 224, 3):
        sample = sample.resize((224, 224))
        sample.save(samplep)  # resize and replace\n", name = = "x ("+i+")"
    if i % 100 == 0:
        print(i, sample.size)
    i = 1 + i

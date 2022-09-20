from PIL import Image
import glob
import re
import os

directory = r'/Users/marc/PycharmProjects/Masterarbeit_OCR/venv/Testfiles'
c=18
for filename in os.listdir(directory):
    if filename.endswith(".jpeg"):
        im = Image.open(filename)
        name='img'+str(c)+'.png'
        rgb_im = im.convert('RGB')
        rgb_im.save(name)
        c+=1
        print(os.path.join(directory, filename))
        continue
    else:
        continue
import os
from datetime import datetime
import time
from PIL import Image


path = "/Users/marc/PycharmProjects/Masterarbeit_OCR/venv/The Times"
c=1
for filename in os.listdir(path):
    inputPath = os.path.join(path, filename)
    if inputPath == path + "/.DS_Store":
        continue
    img = Image.open(inputPath)
    creationdatetimestamp = os.path.getmtime(inputPath)
    creationdate = datetime.fromtimestamp(creationdatetimestamp)
    creationdatestring = str(creationdate)
    print(creationdate)
    name = creationdatestring+"_"+"TheTimes"+"_"+str(c)+".png"
    rgb_im = img.convert("RGB")
    rgb_im.save(name)
    c+=1
    print(os.path.join(path, filename))
    continue





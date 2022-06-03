import pandas as pd
import numpy as np
import cv2
import pytesseract
import glob
import os


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/5.1.0/bin/tesseract'

#image = cv2.imread('telegraph_new.jpeg')
image = cv2.imread('/Users/marc/PycharmProjects/Masterarbeit_OCR/venv/Daily Express/_106419769_dailyexpress.jpg')
images = [cv2.imread(file) for file in glob.glob("/Users/marc/PycharmProjects/Masterarbeit_OCR/venv/Testfiles/*.jpg")]


#text = pytesseract.image_to_string(image)
text = pytesseract.image_to_string(images)
print(text)
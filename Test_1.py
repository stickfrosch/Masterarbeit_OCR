from PIL import Image
import pytesseract as pt
import os

def main():
    # path for the folder for getting the raw images
    path = "/Users/marc/PycharmProjects/Masterarbeit_OCR/venv/Testfiles"

    # link to the file in which output needs to be kept
    fullTempPath = "/Users/marc/PycharmProjects/Masterarbeit_OCR/venv/outputFile.txt"

    # iterating the images inside the folder
    for imageName in os.listdir(path):
        inputPath = os.path.join(path, imageName)
        img = Image.open(inputPath)

        # applying ocr using pytesseract for python
        text = pt.image_to_string(img, lang="eng")

        # saving the  text for appending it to the output.txt file
        # a + parameter used for creating the file if not present
        # and if present then append the text content
        file1 = open(fullTempPath, "a+")

        # providing the name of the image
        file1.write("New_Document" + "\n" + imageName + "\n")

        # providing the content in the image
        file1.write(text + "\n")
        file1.close()

        # for printing the output file
    file2 = open(fullTempPath, 'r')
    print(file2.read())
    file2.close()


if __name__ == '__main__':
    main()


# with open("/Users/marc/PycharmProjects/Masterarbeit_OCR/venv/outputFile.txt",'r') as file3:
#     for line in file3:
#         grade_data = line.strip().split('New_Document')
#         print(grade_data)
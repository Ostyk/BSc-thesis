import os
from PIL import Image
#% matplotlib notebook
#from ipywidgets import FloatProgress
#from IPython.display import display

def resize_images_to_alexnet(file):
    img = Image.open(file)
    img = img.resize((227,227), Image.ANTIALIAS)
    img.save(file)

print("CURRENT PATH:", os.getcwd())
#f = FloatProgress(min=0, max=len(os.listdir()))
#display(f)
im=0
for root, dirs, files in os.walk('data/ISBI2016_ISIC_Part3_Test_Data'):
    for i in files:
        path = root + "/" + i
        try:
            resize_images_to_alexnet(path)
            im+=1
        except IOError:
            print("cannot identify image file %r")
        #f.value+=1
print("Succesfully converted {} images to 227x227".format(im))

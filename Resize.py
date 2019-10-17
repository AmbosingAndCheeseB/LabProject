import glob
from PIL import Image

images = glob.glob('./image(temp)/*.jpg')

def img_resize():

    i =6
    for file in images:
        im = Image.open(file)
        resize_image = im.resize((240, 320))
        resize_image.save('./test_image(temp)/'+str(i)+'.jpg')
        i = i+1


img_resize()

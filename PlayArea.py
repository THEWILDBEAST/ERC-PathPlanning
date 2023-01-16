from PIL import Image,ImageOps
import numpy as np

img=Image.open('cspace.png')

img=ImageOps.grayscale(img)

npimg=np.array(img)
npimg = ~ npimg

npimg[npimg>0]=1
np.save('cspace.npy')

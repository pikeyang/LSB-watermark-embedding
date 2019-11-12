import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg
def binary_to_int(x):
    result=0
    for i in range(len(x)):
        result+=x[i]*2**i
    return result
def set_bit(v, x):
  """Set the index:th bit of v to 1 if x is truthy, else to 0, and return the new value."""
  mask = 1 << 0   # Compute mask, an integer with just bit 'index' set.
  v &= ~mask          # Clear the bit indicated by the mask (if x is False)
  if x:
    v |= mask         # If x was True, set the bit indicated by the mask.
  return v            # Return the result, we're done.
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
class LSB():
    def __init__(self,img):
        self.image=img
        self.watermark=""

    def inject(self,wm):
        self.watermark=wm
        pic=self.image
        binary=[]
        chars=list(self.watermark)
        for c in chars:
            x=ord(c)
            for i in range(8):
                binary.append(x%2)
                x//=2
        arr=np.array(binary)
        arr=arr.reshape(len(binary)//8,8)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                pic[i,j,0]=set_bit(pic[i,j,0],arr[i,j])
        return pic,arr.shape[0]
    def extract(self,length,width):
        pic=self.image
        chars=[]
        b=0
        for i in range(length):
            binary=[]
            for j in range(width):
                b=pic[i,j,0]%2
                binary.append(b)
            x=binary_to_int(binary)
            c=chr(x)
            chars.append(c)
        return chars


img = np.array(Image.open('messi.jpg'),dtype='uint8')
plt.imshow(img)
lsb=LSB(img)
temp,n=lsb.inject("hello world")
img2=Image.fromarray(img.astype('uint8'))
plt.imshow(img2)
plt.show()
message=lsb.extract(n,8)
print(message)


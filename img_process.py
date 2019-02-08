from __future__ import print_function
import sys
import pandas as pd
import os
import random
import json
import numpy as np
from PIL import Image


#im = Image.open("data/IMG_0089.jpg")


'''Crop to Square'''
def crop(im):
	maxsize = (512,512)
	im = im.crop((0,504,3024,3528))
	#im = im.thumbnail(maxsize, Image.BICUBIC)
	im = im.resize(maxsize,Image.BICUBIC)
	im = im.convert('L')
	return im

'''Get Pixel Values'''
def img_to_array(im):
	dat = list(im.getdata())
	pix_values = np.array(dat).reshape((512,512,1))
	return pix_values
'''
im = crop(im)
pix = img_to_array(im)
print(pix)
'''

#df = pd.DataFrame()
if __name__ == '__main__':
	CACHE = json.load(open('dats.json', 'r'))
	s = 0
	t = 0
	for infile in os.listdir("data"):
	    if infile == '.DS_Store':
		    continue
	    if infile in CACHE:
	        t += 1
	        if t > 300:
	            print('Finished labeling')
	            break
	        continue
	    else:
	    	im = Image.open(os.path.join("data", infile))
	    	im.show()
	    	suit = int(input('Suit: '))
	    	value = int(input('Card Value: '))
	    	label = (suit,value)
	    	CACHE[infile] = label
	    	im.close()
	    	s += 1
	    if s>20:
	    	break

	with open('dats.json', 'w') as data:
	    json.dump(CACHE, data)


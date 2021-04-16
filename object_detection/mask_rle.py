import numpy as np
from pycocotools import mask as m
import cv2, json

def main():

	img = cv2.imread('/home/jon/Desktop/kope100-mask.png')
	print(img.shape)
	mask = img[...,0]
	mask[mask>0]=1
	print(mask.shape)

	mask = np.asfortranarray(mask)

	# Create bool array
	# n_a = np.random.normal(size=(10, 10))
	# b_a = np.array(n_a > 0.5, dtype=np.uint8, order='F')

	# print(n_a)
	# print(mask)

	# Encode bool array
	rle = m.encode(mask)
	# d_a = m.decode(rle)

	# print(rle)
	# print(d_a)

	# cv2.imshow("a", mask*255)
	# cv2.imshow("b", d_a*255)
	# cv2.waitKey(0)

	# print(np.all(mask==d_a))

	rle['counts'] = rle['counts'].decode('ascii')

	with open('rle.json', 'w') as outfile:
		json.dump(rle, outfile)

	with open('rle.json') as f:
		data = json.load(f)

	# print(data)

	json_mask = m.decode(data)
	print(np.all(mask==json_mask))

	# Decode byte string rle encoded mask
	# sz = pred['mask']['size']
	# c = pred['mask']['counts'][2:-1]
	# es = str.encode(c)
	# t = {'size': [450, 800], 'counts': es}
	# dm = m.decode(t)


if __name__=="__main__":
	main()
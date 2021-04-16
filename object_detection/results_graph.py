
import os, math
import cv2
import json
import shutil
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from matplotlib.colors import hsv_to_rgb, to_hex, rgb_to_hsv, to_rgb

def darken_color(clr, f):
	# print(clr)
	# print(to_rgb(clr))

	c = rgb_to_hsv(to_rgb(clr))
	# print(c)
	c[2]*=f
	# print(hsv_to_rgb(c))
	# print(to_hex(hsv_to_rgb(c)))
	return to_hex(hsv_to_rgb(c))
    # hsv = np.concatenate([np.random.rand(2), np.random.uniform(min_v, max_v, size=1)])
    # return to_hex(hsv_to_rgb(hdv))

def main():

	res_path = 'results.json'

	with open(res_path) as f:
		data = json.load(f)

	print(data)
	# w = 0.12

	# for i, d in enumerate(list(data.keys())):
	# 	dat0 = data[d][0]
	# 	print(dat0)
	# 	plt.bar(i+i*w, dat0, label=d)

	# Numbers of pairs of bars you want
	N = len(data.keys())

	# Data on X-axis

	# Specify the values of blue bars (height)
	blue_bar = (23, 25, 17, 10)
	# Specify the values of orange bars (height)
	orange_bar = (19, 18, 14, 12)

	red_bar = (10, 11, 24, 22)

	# Position of bars on x-axis
	ind = np.arange(N)

	# Figure size
	ax = plt.figure(figsize=(10,5))

	# Width of a bar 
	width = 0.2

	# clrs = {'mrcnn':[1.0,0.0,0.0], 'yolo':[0.0,1.0,0.0], 'fcos': [0.0,0.0,1.0], 'ssd':[1.0,0.5,0.0]}
	# clrs = {'mrcnn':'#d9ecf2', 'yolo':'#f56a79', 'fcos': '#ff414d', 'ssd':'#1aa6b7'}
	# clrs = {'mrcnn':'#eeecda', 'yolo':'#f08a5d', 'fcos': '#b83b5e', 'ssd':'#6a2c70'}
	clrs = {'mrcnn':'#eeecda', 'yolo':'#f08a5d', 'fcos': '#b83b5e', 'ssd':'#6a2c70'}
	plt.rcParams.update({'font.size': 22})

	# Plotting

	detectors = ['mrcnn', 'fcos', 'yolo', 'ssd']
	detectors_nice = ['Mask R-CNN', 'FCOS', 'YOLOv4', 'SSD']

	# for i, d in enumerate(list(data.keys())):
	for i, d in enumerate(detectors):
		dat0 = data[d][0]
		dat1 = data[d][1]
		dat2 = data[d][2]
		
		# plt.bar(ind + i*width, dat2 , width, label=d, color=clrs[d], linewidth=2)
		# plt.bar(ind + i*width, dat1 , width, color=clrs[d], linewidth=2)
		# plt.bar(ind + i*width, dat0 , width, color=clrs[d], linewidth=2)



		plt.bar(ind + i*width, dat2 , width, label=detectors_nice[i], color=clrs[d], linewidth=2)
		plt.bar(ind + i*width, dat1 , width, color=darken_color(clrs[d],0.9), linewidth=2)
		plt.bar(ind + i*width, dat0 , width, color=darken_color(clrs[d],0.8), linewidth=2)

		# darken flaot colors
		# plt.bar(ind + i*width, dat2 , width, label=d, color=np.asarray(clrs[d]), linewidth=2)
		# plt.bar(ind + i*width, dat1 , width, color=np.asarray(clrs[d])*0.8, linewidth=2)
		# plt.bar(ind + i*width, dat0 , width, color=np.asarray(clrs[d])*0.7, linewidth=2)

	# plt.bar(ind, blue_bar , width, label='Blue bar label')
	# plt.bar(ind + width, orange_bar, width, label='Orange bar label')
	# plt.bar(ind + 2*width, red_bar, width, label='Red bar label')

	plt.ylabel('F1 score')
	# plt.title('Accuracy per experiment for different detectors and object sizes')

	# xticks()
	# First argument - A list of positions at which ticks should be placed
	# Second argument -  A list of labels to place at the given locations
	plt.xticks(ind +1.5*width, ('All', 'S', 'M', 'L'))


	# Finding the best position for legends and putting it
	# plt.legend( loc='upper left')
	plt.rcParams.update({'font.size': 30})
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  fancybox=True, shadow=True, ncol=5)

	plt.show()

	# plt.bar(range(len(data['mrcnn'][0])), [data['mrcnn'][0],data['mrcnn'][1]])
	# plt.bar(range(len(data['yolo'][0])), data['yolo'][0])
	# plt.legend()
	# plt.show()


if __name__ == "__main__":
	main()
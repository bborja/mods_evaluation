import os, math
import cv2
import json
import shutil
import argparse
import subprocess
import numpy as np

def main():

	res_path = 'results.json'

	with open(res_path) as f:
		data = json.load(f)

	print(data)

	res_s1 = []
	res_s2 = []
	res_s3 = []
	idx = 0

	for idx in range(4):

		for d in data.keys():
			print(d)
			print(data[d])
			r = data[d]
			res_s1.append(r[0][idx])
			res_s2.append(r[1][idx])
			res_s3.append(r[2][idx])

		print(res_s1)
		print(res_s2)
		print(res_s3)

		# print(np.asarray(res_s1)/res_s1[0])
		# print(np.asarray(res_s2)/res_s2[0])
		# print(np.asarray(res_s3)/res_s3[2])
		print(round(np.mean(res_s1),3), round(np.std(res_s1),3))
		print(round(np.mean(res_s2),3), round(np.std(res_s2),3))
		print(round(np.mean(res_s3),3), round(np.std(res_s3),3))

if __name__ == "__main__":
	main()
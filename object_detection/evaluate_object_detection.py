import warnings
warnings.filterwarnings("ignore", module="matplotlib")

import os, glob, sys
import cv2
import json
import shutil
import argparse
import subprocess, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_gt_file, code_mask_to_labels, code_labels_to_colors, resize_image
from detection_utils import read_json
from visualize_image import visualize_single_image, visualize_image_for_video
from PIL import Image
from collections import defaultdict

# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from pycocotools.cocoeval import Params

from pycoco.coco import COCO
from pycoco.cocoeval import COCOeval, Params


import skimage.io as io
import random

OUTPUT_PATH = './results/video_output/'
RESULTS_PATH = './results'
DATA_PATH = '/media/jon/disk1/viamaro_data/modd3_full'
clss = ['ship', 'person', 'other']
cls_clr = {'ship': (0,0,1), 'person':(1,0.5,0), 'other':(1,0,0)}


def get_arguments():
	""" Parse all the arguments provided from the CLI
	Returns: A list of parsed arguments
	"""
	parser = argparse.ArgumentParser(description='Marine Obstacle Detection Benchmark.')
	parser.add_argument("--data-path", type=str, default=DATA_PATH,
						help="Absolute path to the folder where MODB sequences are stored.")
	parser.add_argument("--output-path", type=str, default=OUTPUT_PATH,
						help="Output path where the results and statistics of the evaluation will be stored.")
	parser.add_argument("--results-path", type=str, default=RESULTS_PATH,
						help="Absolute path to the folder where evaluation results are stored.")
	parser.add_argument("--method-name", type=str, required=False,
						help="<Required> Method name. This should be equal to the folder name in which the "
							 "segmentation masks are located.")

	return parser.parse_args()

def find_seq_id(name, lst):
	seq_id = next((index for (index, d) in enumerate(lst) if d["path"] == name), None)
	
	return seq_id

def load_calibration(filename):

	if os.path.isfile(filename):

		fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
		M1 = fs.getNode("M1").mat()
		M2 = fs.getNode("M2").mat()
		D1 = fs.getNode("D1").mat()
		D2 = fs.getNode("D2").mat()
		R = fs.getNode("R").mat()
		T = fs.getNode("T").mat()
		return M1, M2, D1, D2, R, T
	else:
		print("calibration file not found!")

def mask_from_sea_edge(sea_edge, size):
	mask = np.zeros(size[0:2])

	for e in sea_edge:
		#print(e)
		axx = e['x_axis']
		axy = e['y_axis']

		#print(axx,axy)
		if type(axx) is float or type(axy) is float or type(axx) is int or type(axy) is int:
			continue
		if axx and axy:
			axx.insert(0, int(axx[0]))
			axy.insert(0, 0)
			axx.append(int(axx[-1]))
			axy.append(0)
			c = np.array([[int(x),int(y)] for x,y in zip(axx,axy)])
			cv2.fillPoly(mask, pts =[c], color=(255,255,255))
	return mask

def danger_zone_to_mask(roll, pitch, height, rnge, M, D, w, h):
	mask = np.ones([h,w],dtype=np.uint8)*255

	A,B,C,D = plane_from_IMU(roll, pitch, height)

	N = 1000
	r = np.linspace(0,180,N)
	x = np.sin(np.radians(r))*rnge
	y = np.cos(np.radians(r))*rnge
	#z = np.zeros(N)
	z = np.zeros(N)+(-1/C)
	z = z*(A*x+B*y+D)

	points = np.transpose(np.array([-y,-z, x]))
	pp, _ = cv2.projectPoints(points, np.identity(3), np.zeros([1,3]), M, distCoeffs=D)
	
	poly = []
	
	for p in pp:
		x = p[0,0]
		y = p[0,1]
		if x>0 and y>0 and x <= w and y <= h:
			poly.append([int(x),int(y)])
			
	poly.insert(0,[0,poly[0][1]])
	poly.insert(0,[0,h])
	poly.append([w, poly[-1][1]])
	poly.append([w,h])
	poly.append([0,h])
			
	cv2.fillPoly(mask, np.array([poly]), color=0)

	return mask

def in_mask(mask, rect):
	rect = [int(round(x)) for x in rect]
	# print(mask)
	roi = mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
	# print("ROI",roi)
	s = np.sum(roi,-1)
	s = np.sum(s)
	a = rect[2]*rect[3]
	# print(s,a,s/a)
	iou = s/a
	# print("IOU", iou)

	return iou>0.5

def iou_overlap(rect, gt, thr=0.3):
	res = []
	# print(rect)
	# print(gt)


	for g in gt:
		gt_bb = g['bbox']
		iou = get_iou(gt_bb, rect)
		res.append(iou)

	# print(res)
	return [x if x>thr else 0 for x in res]

def build_coco_gt(data_path, out_fn, mode='full', ignore_class=False, indices=None):
	gt = read_json(data_path+'/modd3.json')
	data = gt['dataset']

	if indices:
		data['sequences']=[x for x in data['sequences'] if x['id'] in indices]

	coco = {}
	info = {'year': 2020}
	coco['info']=info
	step = 1

	images = []
	annotations = []
	categories = [{'id': 0, 'name': 'ship', 'supercategory': 'obstacle'},{'id': 1, 'name': 'person', 'supercategory': 'obstacle'},{'id': 2, 'name': 'other', 'supercategory': 'obstacle'}]
	coco['categories']=categories

	im_id = 0
	ann_id = 0
	frames_per_seq = []

	for s in data['sequences']:
		frames_per_seq.append(len(s['frames']))
		# print(s['path'])
		# print(s)
		seq_exhaustive = True if s['exhaustive']==1 else False
		# print(seq_exhaustive)

		# if 'mask' in s.keys():
			# print(s['mask'])
			# seq_mask = Image.open(data_path+'/'+s['mask'])
			# plt.imshow(seq_mask)
			# plt.waitforbuttonpress()
		
		seq_nm = s['path'].split('/')[1].split('-')[0]
		calib_fn = data_path+'/calibration/calibration-'+seq_nm+'.yaml'
		M1, M2, D1, D2, R, T = load_calibration(calib_fn)

		for f in s['frames']:
			fn = data_path+s['path']+f['image_file_name']
			I = Image.open(fn)
			(width, height)=I.size

			frame_exhaustive = seq_exhaustive
			if 'exhaustive' in f.keys():
				frame_exhaustive = True if f['exhaustive']==1 else False

			if not seq_exhaustive:
				frame_exhaustive = False

			if frame_exhaustive:
				mask = np.zeros((height, width), dtype=np.uint8)
				if mode=='edge':
					mask+= mask_from_sea_edge(f['water_edges'],(height, width)).astype(np.uint8)
				elif mode=='dz':
					mask+= danger_zone_to_mask(f['roll'], f['pitch'], 0.5, 30, M1, D1, width, height)		

				if 'mask' in s.keys():
					mask+= cv2.imread(data_path+'/'+s['mask'], 0)
				mask[mask>0]=1
			else:
				mask = np.ones((height, width), dtype=np.uint8)

			# plt.clf()
			# plt.imshow(mask)
			# plt.pause(0.01)
			# plt.waitforbuttonpress()

			#print(fn, width, height)
			im = {'id': im_id, 'width': width, 'height': height, 'file_name': fn, 'water_edges': f['water_edges'], 'roll': f['roll'], 'pitch': f['pitch'], 'calib_fn': calib_fn}
			
			images.append(im)

			ann = f['obstacles']
			#print('annotations')
			for a in ann:

				# continue

				if a['type']=='negative':
					bb = a['bbox']
					
					roi = mask[bb[0]:bb[0]+bb[2],bb[1]:bb[1]+bb[3]]
					# mask[bb[0]:bb[0]+bb[2],bb[1]:bb[1]+bb[3]] = 1
					mask[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]] = 1
					# print(roi)
					# plt.clf()
					# plt.imshow(mask)
					# plt.pause(0.01)
					# plt.waitforbuttonpress()
					continue

				ignore = 1 if in_mask(mask,a['bbox']) else 0

				if ignore:
					continue

				C = clss.index(a['type'])
				if ignore_class:
					C = 0
				annotation = {'id': ann_id, 'image_id': im_id, 'category_id': C, 'bbox': a['bbox'], 'iscrowd': 0, 'area': a['area'], 'segmentation': [], 'ignore': ignore}
				#print(annotation)
				annotations.append(annotation)

				ann_id+=1

			im_id+=1
			#break
		#break
	# print(annotations)

	# if len(annotations)==0:
	# 	annotations.append({'id': 0, 'image_id': 0, 'category_id': 0, 'bbox': [], 'iscrowd': 0, 'area': 0, 'segmentation': [], 'ignore': 0})
	# print(frames_per_seq)
	coco['annotations']=annotations
	coco['images']=images
	
	with open(out_fn, 'w') as outfile:
		json.dump(coco, outfile)

def build_coco_res(json_path, gt_json, data_path, out_fn, mode='full', ignore_class=False, indices=None):
	det = read_json(json_path)
	GT = read_json(gt_json)
	det = det['dataset']
	gt = GT['dataset']

	if indices:
		det['sequences']=[x for x in det['sequences'] if x['id'] in indices]
		gt['sequences']=[x for x in gt['sequences'] if x['id'] in indices]
	
	#res_data = read_json(res_path)
	#res_data = res_data['dataset']
	
	step = 1

	### mode: full - evaluate everything in image, edge - filter out detections above sea edge, dz - filter out detections outside of danger zone

	coco_det = []

	im_id = 0
	ann_id = 0
	#print(len(det['sequences']))
	frames_per_seq=[]
	#print(mode)

	for i,s in enumerate(det['sequences']):
		frames_per_seq.append(len(s['frames']))

		seq_nm = s['path'].split('/')[1].split('-')[0]
		seq_exhaustive = True if gt['sequences'][i]['exhaustive']==1 else False
		# print(seq_exhaustive)
		calib_fn = data_path+'/calibration/calibration-'+seq_nm+'.yaml'
		M1, M2, D1, D2, R, T = load_calibration(calib_fn)

		for j,f in enumerate(s['frames']):
			# print(f)
			# print(gt['sequences'][i]['frames'][j])
			fn = data_path+s['path']+f['image_file_name']
			I = Image.open(fn)
			(width, height)=I.size

			water_edges = gt['sequences'][i]['frames'][j]['water_edges']
			roll = gt['sequences'][i]['frames'][j]['roll']
			pitch = gt['sequences'][i]['frames'][j]['pitch']

			frame_exhaustive = seq_exhaustive
			if 'exhaustive' in f.keys():
				frame_exhaustive = True if f['exhaustive']==1 else False

			if not seq_exhaustive:
				frame_exhaustive = False

			if frame_exhaustive:
				mask = np.zeros((height, width), dtype=np.uint8)
				if mode=='edge':
					mask+= mask_from_sea_edge(water_edges,(height, width)).astype(np.uint8)
				elif mode=='dz':
					mask+= danger_zone_to_mask(roll, pitch, 0.5, 30, M1, D1, width, height)		

				if 'mask' in gt['sequences'][i].keys():
					mask+= cv2.imread(data_path+'/'+s['mask'], 0)
				mask[mask>0]=1
			else:
				mask = np.ones((height, width), dtype=np.uint8)

			# plt.clf()
			# plt.imshow(mask)
			# plt.pause(0.01)
			# plt.waitforbuttonpress()
			
			det = []
			if 'detections' in f:
				det = f['detections']

			#roll = f['roll']
			# pitch = f['pitch']
			#print(water_edges)
			gt_ann = gt['sequences'][i]['frames'][j]['obstacles']

			for a in det:

				# if any(np.array(a['bbox'])<0):
				# 	continue

				ovl = iou_overlap(a['bbox'], gt_ann)
				# print(ovl, any(ovl))
				# if not any(ovl):
				# 	continue

				# print(a)
				ignore = 1 if in_mask(mask,a['bbox']) else 0

				# check overlaps with GT, add if below water or above water and has some overlap with a GT annotation
				# if in ignore region and has no overlap, just skip
				# if not any(ovl) and ignore:
				if ignore:
					# TODO here non-overlapping annotations in ignore regions should be removed from the GT json!
					continue

				# if ignore:
				# 	continue

				# if mode=='dz' and ignore:
				# 	continue

				# exhaustive = False
				# exhaustive = True
				# if not exhaustive and not any(ovl):
				# 	continue

				# if ignore:
				# 	continue
				# else:
				# 	ignore = 0
				#print(a['type'], type(a['type']), clss.index(a['type']), str(type(a['type']))=='str')
				c = clss.index(a['type']) if isinstance(a['type'], str) else a['type']
				if ignore_class:
					c = 0
				#print(c)
				# c = 2
				
				detection = {'image_id': im_id, 'category_id': c, 'bbox': a['bbox'], 'score': 1, 'ignore': ignore} # this is for detections

				coco_det.append(detection)

				#ann_id+=1


			im_id+=1
			#break
		#break
	#print(frames_per_seq)

	if len(coco_det)==0:
		coco_det.append({'id': 0, 'image_id': 0, 'category_id': 1, 'bbox': [0,0,0,0], 'score': 0, 'ignore': 0})

	# print(coco_det)
	
	with open(out_fn, 'w') as outfile:
		json.dump(coco_det, outfile)

def build_coco_joint(data_path, results_json, out_fn_gt, out_fn_res, mode='full', ignore_class=False, indices=None):
	# jointly build COCO GT and COCO detection json files
	# in order to filter out annotations inside ignore regions
	GT = read_json(data_path+'/modd3.json')
	det = read_json(results_json)	

	det = det['dataset']
	gt = GT['dataset']

	# filter by indices

	if indices:
		det['sequences']=[x for x in det['sequences'] if x['id'] in indices]
		gt['sequences']=[x for x in gt['sequences'] if x['id'] in indices]

	# set up dataset info
	coco = {'info': {'year': 2020}}
	categories = [{'id': 0, 'name': 'ship', 'supercategory': 'obstacle'},{'id': 1, 'name': 'person', 'supercategory': 'obstacle'},{'id': 2, 'name': 'other', 'supercategory': 'obstacle'}]
	coco['categories']=categories

	images = []
	annotations = []
	detections = []
	frames_per_seq = []

	im_id = 0
	ann_id = 0
	det_id = 0

	for s_gt, s_dt in zip(gt['sequences'], det['sequences']):
		frames_per_seq.append(len(s_gt['frames']))

		# set exhaustive flag
		seq_exhaustive = True if s_gt['exhaustive']==1 else False

		# load calibration
		seq_nm = s_gt['path'].split('/')[1].split('-')[0]
		calib_fn = data_path+'/calibration/calibration-'+seq_nm+'.yaml'
		M1, M2, D1, D2, R, T = load_calibration(calib_fn)

		for f_gt, f_dt in zip(s_gt['frames'], s_dt['frames']):
			fn = data_path+s_gt['path']+f_gt['image_file_name']
			I = Image.open(fn)
			(width, height)=I.size

			# build mask
			frame_exhaustive = seq_exhaustive
			if 'exhaustive' in f_gt.keys():
				frame_exhaustive = True if f_gt['exhaustive']==1 else False

			if not seq_exhaustive:
				frame_exhaustive = False

			if not frame_exhaustive:

				if mode=='dz':
					mask = np.zeros((height, width), dtype=np.uint8)
					if mode=='edge':
						mask+= mask_from_sea_edge(f_gt['water_edges'],(height, width)).astype(np.uint8)
					elif mode=='dz':
						mask+= danger_zone_to_mask(f_gt['roll'], f_gt['pitch'], 0.5, 30, M1, D1, width, height)		

					if 'mask' in s_gt.keys():
						mask+= cv2.imread(data_path+'/'+s_gt['mask'], 0)
					mask[mask>0]=1

				else:
					mask = np.ones((height, width), dtype=np.uint8)
				# plt.clf()
				# # plt.imshow(I)
				# plt.imshow(mask, cmap='gray', alpha=0.3)
				# plt.pause(0.01)
				# plt.waitforbuttonpress()
			else:
				mask = np.zeros((height, width), dtype=np.uint8)
				if mode=='edge':
					mask+= mask_from_sea_edge(f_gt['water_edges'],(height, width)).astype(np.uint8)
				elif mode=='dz':
					mask+= danger_zone_to_mask(f_gt['roll'], f_gt['pitch'], 0.5, 30, M1, D1, width, height)		

				if 'mask' in s_gt.keys():
					mask+= cv2.imread(data_path+'/'+s_gt['mask'], 0)
				mask[mask>0]=1

			# if frame_exhaustive:
				
			# else:
			# 	mask = np.ones((height, width), dtype=np.uint8)

			# setup image data
			im = {'id': im_id, 'width': width, 'height': height, 'file_name': fn, 'water_edges': f_gt['water_edges'], 'roll': f_gt['roll'], 'pitch': f_gt['pitch'], 'calib_fn': calib_fn}

			ann = f_gt['obstacles']
			det = []

			if 'detections' in f_dt:
				det = f_dt['detections']

			plt.clf()

			for a in ann:
				bb = a['bbox']
				# add negative annotations to mask
				if a['type']=='negative':							
					roi = mask[bb[0]:bb[0]+bb[2],bb[1]:bb[1]+bb[3]]
					mask[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]] = 1
				else: # check whether in ignore region and overlaps
					ignore = 1 if in_mask(mask,a['bbox']) else 0
					# print(ignore)
					ovl = iou_overlap(a['bbox'], det)
					
					# print(ovl)
					# print(any(ovl))

					if ignore and not any(ovl):
						if mode=='dz':
							continue

					C = clss.index(a['type'])
					if ignore_class:
						C = 0
					annotation = {'id': ann_id, 'image_id': im_id, 'category_id': C, 'bbox': bb, 'iscrowd': 0, 'area': a['area'], 'segmentation': [], 'ignore': ignore}
					#print(annotation)
					annotations.append(annotation)
					ann_id+=1
					# plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=2,edgecolor='k',facecolor='none', alpha=0.5))

			# print(det)
			for a in det:
				bb = a['bbox']
				# print(a)
				ignore = 1 if in_mask(mask,a['bbox']) else 0
				ovl = iou_overlap(a['bbox'], ann)
				if ignore and not any(ovl):
					if mode=='dz':
						continue

				C = clss.index(a['type']) if isinstance(a['type'], str) else a['type']
				if ignore_class:
					C = 0

				detection = {'image_id': im_id, 'category_id': C, 'bbox': bb, 'score': 1, 'ignore': ignore} # this is for detections

				detections.append(detection)
				# plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=2,edgecolor='w',facecolor='none', alpha=0.5))

			
			# plt.imshow(I)
			# plt.imshow(mask, cmap='gray', alpha=0.3)

			# plt.pause(0.01)
			# plt.waitforbuttonpress()

			# append to final lists
			images.append(im)

			# increment counters
			im_id+= 1

		# print(s_gt, s_dt)

	coco['annotations']=annotations
	coco['images']=images

	# write gt
	with open(out_fn_gt, 'w') as outfile:
		json.dump(coco, outfile)

	# write detections
	with open(out_fn_res, 'w') as outfile:
		json.dump(detections, outfile)

def plane_from_IMU(roll, pitch, height):
	# np.set_printoptions(precision=4)
	# roll, pitch are in degrees

	# invert the signs
	pitch = -pitch
	roll = -roll

	c,s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
	Rx = np.array([[1,0,0],[0,c,-s],[0,s,c]])
	c,s = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
	Ry = np.array([[c,0,s],[0,1,0],[-s,0,c]])

	n = np.array([[0],[0],[1]])

	n = np.dot(np.dot(Rx,Ry),n);

	B = np.append(n, height)
	B = B/np.linalg.norm(B)

	return B

# class Params:

# 	def __init__(self):
# 		self.iouType = 'bbox'
# 		self.imgIds = []
# 		self.catIds = []
# 		# np.arange causes trouble.  the data point on arange is slightly larger than the true value
# 		# self.iouThrs = np.array([0.3, 0.3])
# 		self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
# 		self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
# 		self.maxDets = [1, 10, 100]
# 		self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
# 		self.areaRngLbl = ['all', 'small', 'medium', 'large']
# 		self.useCats = 1
# 		self.useSegm = None

def evaluate_results(gt_json, results_json):
	import copy
	coco = COCO(gt_json)
	coco_res = coco.loadRes(results_json)

	ev = COCOeval(coco, coco_res, iouType='bbox')
	ev.params.iouThrs = np.array([0.3, 0.3])

	ev.evaluate()	
	ev.accumulate()
	ev.summarize()

	stats = np.nan_to_num(ev.stats)
	stats[stats==-1]=0

	# print(stats)


	F = 2*(stats[0]*stats[8])/(stats[0]+stats[8]) if stats[0]!=0 and stats[8]!=0 else 0
	Fs = 2*(stats[3]*stats[9])/(stats[3]+stats[9]) if stats[3]!=0 and stats[9]!=0 else 0
	Fm = 2*(stats[4]*stats[10])/(stats[4]+stats[10]) if stats[4]!=0 and stats[10]!=0 else 0
	Fl = 2*(stats[5]*stats[11])/(stats[5]+stats[11]) if stats[5]!=0 and stats[11]!=0 else 0

	print('precision all: {},  precision small: {}, precision medium: {}, precision large: {}, recall all: {}, recall small: {}, recall medium: {}, recall large: {}, F all: {}, F small: {}, F medium: {}, F large: {}'.format(stats[0], stats[3], stats[4], stats[5], stats[8], stats[9], stats[10],stats[11], F, Fs, Fm, Fl))

	return [F, Fs, Fm, Fl]
	# return stats

def get_iou(bb1, bb2):
	"""
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	bb1 : dict
		Keys: {'x1', 'x2', 'y1', 'y2'}
		The (x1, y1) position is at the top left corner,
		the (x2, y2) position is at the bottom right corner
	bb2 : dict
		Keys: {'x1', 'x2', 'y1', 'y2'}
		The (x, y) position is at the top left corner,
		the (x2, y2) position is at the bottom right corner

	Returns
	-------
	float
		in [0, 1]
	"""

	bb1 = {'x1': bb1[0],'y1': bb1[1],'x2': bb1[0]+bb1[2],'y2': bb1[1]+bb1[3]}
	bb2 = {'x1': bb2[0],'y1': bb2[1],'x2': bb2[0]+bb2[2],'y2': bb2[1]+bb2[3]}


	# assert bb1['x1'] < bb1['x2']
	# assert bb1['y1'] < bb1['y2']
	# assert bb2['x1'] < bb2['x2']
	# assert bb2['y1'] < bb2['y2']

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1['x1'], bb2['x1'])
	y_top = max(bb1['y1'], bb2['y1'])
	x_right = min(bb1['x2'], bb2['x2'])
	y_bottom = min(bb1['y2'], bb2['y2'])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
	bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou

def display_results(coco, coco_res, ids=[], mode='edge'):
	# print(ids)
	imgIds = coco.getImgIds()
	# print(imgIds)
	ev = COCOeval(coco, coco_res, iouType='bbox')
	# ev.useCats = 1
	# return
	#prm = Params()
	# ev._prepare()
	ev.evaluate()
	#print(ev._gts)
	#for a in ev._gts.items():
	#	print(a)
	# return
	# random.shuffle(imgIds)
	# print(rd)
	step = 1
	print("MODE:", mode)

	# for i in ids:
	for i in imgIds[::step]:
	# for i in imgIds[2610::step]:

		#os.system('tput reset')
		# print(i)
		
		# print('\n', img)
		# print(coco.loadImgs(imgIds[i]))
		img = coco.loadImgs(i)[0]
		#print(img)
		

		# print(img['water_edges'])
		if mode=='edge':
			mask = mask_from_sea_edge(img['water_edges'],(img['height'],img['width']))
		elif mode=='dz':
			M1, M2, D1, D2, R, T = load_calibration(img['calib_fn'])
			#print(M1, D1)
			mask = danger_zone_to_mask(img['roll'], img['pitch'], 0.5, 30, M1, D1, img['width'], img['height'])
		


		annIds = coco.getAnnIds(imgIds=img['id'])
		anns = coco.loadAnns(annIds)

		detIds = coco_res.getAnnIds(imgIds=img['id'])
		dets = coco_res.loadAnns(detIds)

		# print('\n',"display gt:", anns)
		# print('\n',"display dt:", dets)

		# ious_0 = ev.ious[i, 0]
		# ious_1 = ev.ious[i, 1]
		# ious_2 = ev.ious[i, 2]

		# print('\n',"ious0", ious_0)
		# print('\n',"ious1", ious_1)
		# print('\n',"ious2", ious_2)

		# print(img['file_name'])

		# display FP
		thr = 0.3
		ious = np.zeros((len(anns),len(dets)))
		for i,a in enumerate(anns):
			for j,d in enumerate(dets):
				n = get_iou(a['bbox'],d['bbox'])
				ious[i,j] = n if n>thr else 0
				# ious[i,j] = n

		# print(ious)

		idx = ious.any(axis=0)
		fp = [x for x, id in zip(dets, idx) if not id]

		idx = ious.any(axis=1)
		fn = [x for x, id in zip(anns, idx) if not id]

		# print("FP:", fp)
		# print(len(fp))

		if len(fp)==1 and fp[0]['score']==0:
			continue

		if fp:
			
			I = io.imread(img['file_name'])/255.0
			plt.clf()
			plt.axis('off')
			plt.imshow(I)
			plt.imshow(mask, cmap='gray', alpha=0.3)

			# print(ious)
		# print(fn)

			# print(fp)
			for f in fp:
				bb = f['bbox']
				# print(bb, a['category_id'])
				c = cls_clr[clss[f['category_id']]]
				if f['ignore']:
					plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=2,edgecolor=c,facecolor='none', alpha=0.5))
				else:				
					plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=0,edgecolor='none',facecolor=c, alpha=0.3))
					plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=2,edgecolor=c,facecolor='none'))

		# print(fn)
		# for f in fn:
		# 	bb = f['bbox']
		# 	# print(bb, a['category_id'])
		# 	c = cls_clr[clss[f['category_id']]]
		# 	if f['ignore']:
		# 		plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=2,edgecolor=c,facecolor='none', alpha=0.5))
		# 	else:				
		# 		plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=0,edgecolor='none',facecolor=c, alpha=0.3))
		# 		plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=2,edgecolor=c,facecolor='none'))

		# for d in dets:
		# 	bb = d['bbox']
		# 	plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=1,edgecolor='k', facecolor='none', linestyle='-'))

			# print('GT')
			for a in anns:
				bb = a['bbox']
				# print(bb, a['category_id'])
				c = cls_clr[clss[a['category_id']]]
				if a['ignore']:
					plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=1,edgecolor='k',facecolor='none', alpha=0.5))
				else:				
					# plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=0,edgecolor='none',facecolor='k', alpha=0.3))
					plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=1,edgecolor='k',facecolor='none'))

		# # print('det')
		# for d in dets:
		# 	bb = d['bbox']
		# 	#print(bb, d['category_id'])
		# 	print(d)
			
		# 	c = cls_clr[clss[d['category_id']]]
		# 	if d['ignore']:
		# 		plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=2,edgecolor=c,facecolor='none', alpha=0.5, linestyle='--'))
		# 	else:				
		# 		plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=0,edgecolor='none',facecolor=c, alpha=0.3))
		# 		plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=2,edgecolor=c,facecolor='none', linestyle='--'))

		# print('\n', "ship", ev.evaluateImg(i, 0, [0 ** 2, 1e6 ** 2], 1000))
		# print('\n', "person", ev.evaluateImg(i, 1, [0 ** 2, 1e6 ** 2], 1000))
		# print('\n', "other", ev.evaluateImg(i, 2, [0 ** 2, 1e6 ** 2], 1000))

		if fp:
			plt.draw()
			# plt.pause(0.01)
			plt.waitforbuttonpress()
			# break

def sanity_check(gt, res):
	gt_data = read_json(gt)['dataset']
	res_data = read_json(res)['dataset']
	#print('GT num of sequences:', len(gt_data['sequences']))
	#print('res num of sequences:', len(res_data['sequences']))
	gt_ids = sorted([s['id'] for s in gt_data['sequences']])
	res_ids = sorted([s['id'] for s in res_data['sequences']])
	print(gt_ids)
	print(res_ids)
	return gt_ids==res_ids

def to_json(o, level=0):
	INDENT = 3
	SPACE = " "
	NEWLINE = "\n"
	ret = ""
	if isinstance(o, dict):
		ret += "{" + NEWLINE
		comma = ""
		for k, v in o.items():
			ret += comma
			comma = ",\n"
			ret += SPACE * INDENT * (level + 1)
			ret += '"' + str(k) + '":' + SPACE
			ret += to_json(v, level + 1)

		ret += NEWLINE + SPACE * INDENT * level + "}"
	elif isinstance(o, str):
		ret += '"' + o + '"'
	elif isinstance(o, list):
		ret += "[" + ",".join([to_json(e, level + 1) for e in o]) + "]"
		# Tuples are interpreted as lists
	elif isinstance(o, tuple):
		ret += "[" + ",".join(to_json(e, level + 1) for e in o) + "]"
	elif isinstance(o, bool):
		ret += "true" if o else "false"
	elif isinstance(o, int):
		ret += str(o)
	elif isinstance(o, float):
		ret += '%.7g' % o
	elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.integer):
		ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
	elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.inexact):
		ret += "[" + ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
	elif o is None:
		ret += 'null'
	else:
		raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
	return ret

def evaluate_one_sequence(data_path, gt_json, det_json):
	indices = [0, 1]

	coco_gt_out = 'gt_tmp.json'
	coco_det_out = 'det_tmp.json'

	build_coco_gt(data_path, coco_gt_out, mode='full', ignore_class=False, indices=indices)
	build_coco_res(det_json, data_path+'/'+gt_json, data_path, coco_det_out, mode='full', ignore_class=False, indices=indices)

	# with open(coco_gt_out, 'w') as outfile:
	# 	json.dump(gt_coco, outfile)
	# with open('det_tmp.json', 'w') as outfile:
	# 	json.dump(det_coco, outfile)

	coco = COCO(coco_gt_out)
	coco_res = coco.loadRes(coco_det_out)

	ev = COCOeval(coco, coco_res, iouType='bbox')
	ev.evaluate()
	# prm = Params(iouType='bbox')
	# prm.useCats = 0
	# ev.params = prm
	
	ev.accumulate()
	# print(ev.eval)
	ev.summarize()

	# evaluate

def test(data_path, gt_json):
	indices = [1]
	ignore = True
	# ignore = False
	mode = 'edge'
	gt_json_fn = 'modd3_coco_gt.json'
	build_coco_gt(data_path, gt_json_fn, mode=mode, indices=indices, ignore_class=ignore)
	coco = COCO(gt_json_fn)
	json_path = '/home/jon/Desktop/Mask_RCNN/mrcnn_res.json'
	out_json_fn = 'mrcnn_coco.json'
	build_coco_res(json_path, data_path+'/modd3.json', data_path, out_json_fn, mode=mode, indices=indices, ignore_class=ignore)
	coco_res = coco.loadRes(out_json_fn)
	stats = evaluate_results(gt_json_fn, out_json_fn)
	print(stats)
	

	display_results(coco, coco_res, indices)

def build_and_evaluate(data_path, res_json_fn, indices=None, mode='edge', ignore_class=False):
	coco_gt_out = 'gt_tmp.json'
	coco_det_out = 'det_tmp.json'
	# build_coco_gt(data_path, coco_gt_out, mode=mode, indices=indices, ignore_class=ignore_class)
	# build_coco_res(res_json_fn, data_path+'/modd3.json', data_path, coco_det_out, mode=mode, indices=indices, ignore_class=ignore_class)
	build_coco_joint(data_path, res_json_fn, coco_gt_out, coco_det_out, mode=mode, indices=indices, ignore_class=ignore_class)
	coco = COCO(coco_gt_out)
	coco_res = coco.loadRes(coco_det_out)
	stats = evaluate_results(coco_gt_out, coco_det_out)
	# display_results(coco, coco_res, mode=mode)
	# print(stats)
	return stats

def per_sequence_evaluation():
	args = get_arguments()
	detectors = [('mrcnn', '/home/jon/Desktop/Mask_RCNN/'), ('yolo','/home/jon/Desktop/darknet/'), ('fcos','/home/jon/Desktop/FCOS/'), ('ssd', '/home/jon/Desktop/ssd.pytorch/')]
	modes = ['edge', 'dz']
	gt_json_fn = 'modd3_coco_gt.json'

	res = []
	time1 = time.time()

	for i in range(96):
	# for i in range(1):
		print(i)
		ind = [i]
		results = {}
		for det, pth in detectors:
			print(det)
			json_path = pth+det+'_res.json'
			# out_json_fn = det+'_coco.json'

			cur = []

			r = build_and_evaluate(args.data_path, json_path, mode='edge', indices = ind)
			print('{:.3f} {:.3f} {:.3f} {:.3f}'.format(r[0],r[1],r[2],r[3]))
			cur.append(r)
			r = build_and_evaluate(args.data_path, json_path, mode='edge', indices = ind, ignore_class=True)
			print('{:.3f} {:.3f} {:.3f} {:.3f}'.format(r[0],r[1],r[2],r[3]))
			cur.append(r)
			r = build_and_evaluate(args.data_path, json_path, mode='dz', indices = ind)
			print('{:.3f} {:.3f} {:.3f} {:.3f}'.format(r[0],r[1],r[2],r[3]))
			cur.append(r)
			r = build_and_evaluate(args.data_path, json_path, mode='dz', indices = ind, ignore_class=True)
			print('{:.3f} {:.3f} {:.3f} {:.3f}'.format(r[0],r[1],r[2],r[3]))
			cur.append(r)
			results[det]=cur
		# print(results)
		res.append(results)
	

	print("time elapsed:", time.time()-time1)

	with open('per_sequence_results.json', 'w') as outfile:
		json.dump(res, outfile)

def main():
	# per_sequence_evaluation()
	# return
	args = get_arguments()
	# test(args.data_path, 'modd3.json')
	ind = [80]
	# ind = None
	# ignore = True
	# ignore = False
	# mode = 'dz'
	# mode='full'
	# build_and_evaluate(args.data_path, '/home/jon/Desktop/Mask_RCNN/mrcnn_res.json', indices = ind, ignore_class=ignore, mode=mode)
	# return

	# evaluate_one_sequence(args.data_path, 'modd3.json', '/home/jon/Desktop/Mask_RCNN/mrcnn_res.json')
	# return
	
	detectors = [('mrcnn', '/home/jon/Desktop/Mask_RCNN/'), ('yolo','/home/jon/Desktop/darknet/'), ('fcos','/home/jon/Desktop/FCOS/'), ('ssd', '/home/jon/Desktop/ssd.pytorch/')]
	# detectors = [('mrcnn', '/home/jon/Desktop/Mask_RCNN/')]
	# detectors = [('ssd', '/home/jon/Desktop/ssd.pytorch/')]

	modes = ['edge', 'dz']
	#detectors_paths = ['/home/jon/Desktop/Mask_RCNN/', '/home/jon/Desktop/darknet/', '/home/jon/Desktop/FCOS/', '/home/jon/Desktop/ssd.pytorch/']
	#detectors_paths = ['/home/jon/Desktop/Mask_RCNN/', '/home/jon/Desktop/darknet/', '/home/jon/Desktop/ssd.pytorch/']
	data = []
	
	gt_json_fn = 'modd3_coco_gt.json'
	
	# sanity_check(args.data_path+'/modd3.json', '/home/jon/Desktop/Mask_RCNN/mrcnn_res.json')
	# return
	
	#mode = modes[0]

	res = defaultdict(list)
	time1 = time.time()

	for det, pth in detectors:
		print(det)

		json_path = pth+det+'_res.json'
		# sanity_check(args.data_path+'/modd3.json', json_path)
		# out_json_fn = det+'_coco.json'

		r = build_and_evaluate(args.data_path, json_path, mode='edge', indices = ind)
		res[det].append((r[0],r[1],r[2],r[3]))
		r = '{:.3f} {:.3f} {:.3f} {:.3f}'.format(r[0],r[1],r[2],r[3])
		print(r)
		
		r = build_and_evaluate(args.data_path, json_path, mode='edge', indices = ind, ignore_class=True)
		res[det].append((r[0],r[1],r[2],r[3]))
		r = '{:.3f} {:.3f} {:.3f} {:.3f}'.format(r[0],r[1],r[2],r[3])
		print(r)
		
		r = build_and_evaluate(args.data_path, json_path, mode='dz', indices = ind, ignore_class=True)
		res[det].append((r[0],r[1],r[2],r[3]))
		r = '{:.3f} {:.3f} {:.3f} {:.3f}'.format(r[0],r[1],r[2],r[3])
		print(r)
		
		# break

	print("time elapsed:", time.time()-time1)
	with open('results.json', 'w') as outfile:
		outfile.write(to_json(res))
	return
	
	for m in modes:
		build_coco_gt(args.data_path, gt_json_fn, mode=m, indices=ind)
		print(m)
	
		for det, pth in detectors:
			print(det)
			json_path = pth+det+'_res.json'
			# out_json_fn = det+'_coco.json'

			r = build_and_evaluate(args.data_path, json_path, mode=m, indices = ind)
			print(r)

			# do sanity check of sizes and indices of sequences in gt and results
			# if sanity_check(args.data_path+'/modd3.json', json_path):
			# 	print(json_path)
			# 	build_coco_res(json_path, args.data_path+'/modd3.json', args.data_path, out_json_fn, mode=m)
			# 	stats = evaluate_results(gt_json_fn, out_json_fn)
			# 	print(stats)
			# 	#break
			# 	res[det+'_'+m]=stats.T.tolist()
	return

	ignore_class=True

	for m in modes:
		build_coco_gt(args.data_path, gt_json_fn, mode=m, ignore_class=ignore_class)
	
		for det, pth in detectors:
			print(det)
			json_path = pth+det+'_res.json'
			out_json_fn = det+'_coco.json'
			# do sanity check of sizes and indices of sequences in gt and results
			if sanity_check(args.data_path+'/modd3.json', json_path):
				print(json_path)
				build_coco_res(json_path, args.data_path+'/modd3.json', args.data_path, out_json_fn, mode=m, ignore_class=ignore_class)
				stats = evaluate_results(gt_json_fn, out_json_fn)
				print(stats)
				#break
				key = det+'_'+m+'_no_class' if ignore_class else det+'_'+m
				res[key]=stats.T.tolist()
			
				# coco = COCO(gt_json_fn)
				# coco_res = coco.loadRes(out_json_fn)
				# display_results(coco, coco_res)

	print("time elapsed:", time.time()-time1)

	with open('results.json', 'w') as outfile:
		outfile.write(to_json(res))
			
	# with open('results.json', 'w') as outfile:
	# 	js = json.dumps(res, indent=None,separators=(",",":"))
	# 	outfile.write(js)
		
		
	return
	
	#mrcnn_res_path = '/home/jon/Desktop/Mask_RCNN/mrcnn_res.json'
	#yolo_res_path = '/home/jon/Desktop/darknet/yolo_res.json'
	#fcos_res_path = '/home/jon/Desktop/FCOS/fcos_res.json'
	#ssd_res_path = '/home/jon/Desktop/ssd.pytorch/ssd_res.json'

	coco_json_fn = 'modd3_coco_gt.json'
	mrcnn_json_fn = 'mrcnn_coco.json'
	yolo_json_fn = 'yolo_coco.json'
	build_coco_gt(args.data_path, coco_json_fn)
	build_coco_res(yolo_res_path, args.data_path+'/modd3.json', args.data_path, yolo_json_fn)
	# build_coco_res(mrcnn_res_path, args.data_path+'/modd3.json', args.data_path, mrcnn_json_fn)

	# return

	# evaluate_results(coco_json_fn, mrcnn_json_fn)
	evaluate_results(coco_json_fn, yolo_json_fn)

	# coco = COCO(coco_json_fn)
	# coco_res = coco.loadRes(mrcnn_json_fn)
	# display_results(coco, coco_res)

	# r1 = [518, 571, 250, 67]
	# r2 = [533.1180572509766, 528.3761787414551, 233.47592163085938, 106.11275482177734]
	# print(get_iou(r1,r2))

	# 
	
	# imgIds = coco.getImgIds()
	# #print(imgIds)
	# for i in range(200,500):
	# 	img = coco.loadImgs(imgIds[i])[0]
	# 	# print(img)
	# 	I = io.imread(img['file_name'])/255.0
	# 	#print(img)
	# 	plt.axis('off')
	# 	plt.imshow(I)

	# 	plt.draw()
	# 	plt.waitforbuttonpress()

	# 	break

	return

	# anns = coco.loadAnns([0,1,2])
	# print(anns)

	# dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))

	
	#print(coco_res)
	# img_list = [a for a in range(100)]
	
	
	# annIds = coco.getAnnIds(imgIds=img_list)
	# detIds = coco.getAnnIds(imgIds=img_list)
	# anns = coco.loadAnns(annIds)
	# dets = coco_res.loadAnns(detIds)
	# #dets = coco_res.loadAnns([0,1,2])
	
	# print("anns", anns)
	# print("dets", dets)

	# #print(coco_res.loadAnns())

	# return

	imgIds = coco.getImgIds()
	#print(imgIds)
	for a in range(15):
		plt.clf()
		id = np.random.randint(0,len(imgIds))
		img = coco.loadImgs(imgIds[id])[0]
		I = io.imread(img['file_name'])/255.0
		#print(img)
		plt.axis('off')
		plt.imshow(I)
		

		annIds = coco.getAnnIds(imgIds=img['id'])
		anns = coco.loadAnns(annIds)
		coco.showAnns(anns,draw_bbox=True)

		# TODO custom detection display
		detIds = coco_res.getAnnIds(imgIds=img['id'])
		dets = coco_res.loadAnns(detIds)
		print(dets)
		#coco_res.showAnns(dets,draw_bbox=True)

		print(ev.evaluateImg(id, 0, [0, 1], 1000))
		# evaluateImg(self, imgId, catId, aRng, maxDet))

		plt.draw()
		plt.waitforbuttonpress()

	return

	
	
	

	coco.showAnns(anns,draw_bbox=True)
	print(coco.getCatIds())
	#print(a)
	
if __name__ == "__main__":
	main()
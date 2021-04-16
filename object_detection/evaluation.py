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

# from utils import read_gt_file, code_mask_to_labels, code_labels_to_colors, resize_image
from detection_utils import read_json, load_calibration, to_json, mask_from_sea_edge, in_mask, iou_overlap, danger_zone_to_mask
# from visualize_image import visualize_single_image, visualize_image_for_video
from PIL import Image
from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.cocoeval import Params

# from pycoco.coco import COCO
# from pycoco.cocoeval import COCOeval, Params

import skimage.io as io
import random

SEQUENCES = None
DATA_PATH = '/media/jon/disk1/viamaro_data/modd3_full'
clss = ['ship', 'person', 'other']

def get_arguments():
	""" Parse all the arguments provided from the CLI
	Returns: A list of parsed arguments
	"""
	parser = argparse.ArgumentParser(description='Marine Obstacle Detection Benchmark.')
	parser.add_argument("--data-path", type=str, default=DATA_PATH,
						help="Absolute path to the folder where MODB sequences are stored.")
	# parser.add_argument("--results-path", type=str, default=RESULTS_PATH,
	# 					help="Absolute path to the folder where evaluation results are stored.")
	parser.add_argument("--sequences", type=int, nargs='+', required=False, default=SEQUENCES,
						help="List of sequences on which the evaluation procedure is performed. Zero = all.")

	return parser.parse_args()

def build_coco(data_path, results_json, out_fn_gt, out_fn_res, mode='full', ignore_class=False, indices=None):
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
	categories = [{'id': 0, 'name': 'ship', 'supercategory': 'obstacle'}, {'id': 1, 'name': 'person', 'supercategory': 'obstacle'}, {'id': 2, 'name': 'other', 'supercategory': 'obstacle'}]
	coco['categories']=categories

	images = []
	annotations = []
	detections = []
	frames_per_seq = []

	im_id = 0
	ann_id = 0
	det_id = 0

	# danger zone parameters
	dz_range = 30
	dz_height  = 0.5

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
						mask+= danger_zone_to_mask(f_gt['roll'], f_gt['pitch'], dz_height, dz_range, M1, D1, width, height)		

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
					mask+= danger_zone_to_mask(f_gt['roll'], f_gt['pitch'], dz_height, dz_range, M1, D1, width, height)		

				if 'mask' in s_gt.keys():
					mask+= cv2.imread(data_path+'/'+s_gt['mask'], 0)
				mask[mask>0]=1

			# setup image data
			im = {'id': im_id, 'width': width, 'height': height, 'file_name': fn, 'water_edges': f_gt['water_edges'], 'roll': f_gt['roll'], 'pitch': f_gt['pitch'], 'calib_fn': calib_fn}

			ann = f_gt['obstacles']
			det = []

			if 'detections' in f_dt:
				det = f_dt['detections']

			# plt.clf()

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

	F = 2*(stats[0]*stats[8])/(stats[0]+stats[8]) if stats[0]!=0 and stats[8]!=0 else 0
	Fs = 2*(stats[3]*stats[9])/(stats[3]+stats[9]) if stats[3]!=0 and stats[9]!=0 else 0
	Fm = 2*(stats[4]*stats[10])/(stats[4]+stats[10]) if stats[4]!=0 and stats[10]!=0 else 0
	Fl = 2*(stats[5]*stats[11])/(stats[5]+stats[11]) if stats[5]!=0 and stats[11]!=0 else 0

	# return F score for small, medium and large obstacles
	return [F, Fs, Fm, Fl]

def build_and_evaluate(data_path, res_json_fn, indices=None, mode='edge', ignore_class=False):
	coco_gt_out = 'gt_tmp.json'
	coco_det_out = 'det_tmp.json'
	build_coco(data_path, res_json_fn, coco_gt_out, coco_det_out, mode=mode, indices=indices, ignore_class=ignore_class)
	coco = COCO(coco_gt_out)
	coco_res = coco.loadRes(coco_det_out)
	stats = evaluate_results(coco_gt_out, coco_det_out)
	return stats

def main():
	args = get_arguments()
	
	ind = args.sequences
	
	detectors = [('mrcnn', '/home/jon/Desktop/Mask_RCNN/'), ('yolo','/home/jon/Desktop/darknet/'), ('fcos','/home/jon/Desktop/FCOS/'), ('ssd', '/home/jon/Desktop/ssd.pytorch/')]
	# detectors = [('mrcnn', '/home/jon/Desktop/Mask_RCNN/')]
	# detectors = [('ssd', '/home/jon/Desktop/ssd.pytorch/')]

	res = defaultdict(list)
	time1 = time.time()

	for det, pth in detectors:
		print(det)

		json_path = pth+det+'_res.json'

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

	print("time elapsed:", time.time()-time1)
	with open('results.json', 'w') as outfile:
		outfile.write(to_json(res))
	
	
if __name__ == "__main__":
	main()
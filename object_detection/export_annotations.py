import os
import cv2
import json
import shutil
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import read_gt_file, code_mask_to_labels, code_labels_to_colors, resize_image, read_json
from visualize_image import visualize_single_image, visualize_image_for_video

DATA_PATH = '/media/jon/disk1/viamaro_data/modd3_full'

def get_arguments():
	""" Parse all the arguments provided from the CLI
	Returns: A list of parsed arguments
	"""
	parser = argparse.ArgumentParser(description='Marine Obstacle Detection Benchmark.')
	parser.add_argument("--data-path", type=str, default=DATA_PATH,
						help="Absolute path to the folder where MODB sequences are stored.")

	return parser.parse_args()

def find_seq_id(name, lst):
	seq_id = next((index for (index, d) in enumerate(lst) if d["path"] == name), None)
	
	return seq_id

def find_seq_id_in_list(id, lst):
	a = [s['id']==id for s in lst]
	b = [i for i, x in enumerate(a) if x]
	
	if b == []:
		return None
	else:
		return b[0]

def main():
	import glob
	args = get_arguments()
	res_path = '/home/jon/Desktop/Mask_RCNN/mrcnn_res.json'
	output_dir = 'annotations'
	
	#gt = read_json(args.data_path+'/modd3.json')
	gt = read_json('/media/jon/disk1/jon/Desktop/viamaro-obstacle-annotator/modd3_fix/modd3.json')
	gt = gt['dataset']   
	
	#seq = sorted(glob.glob(DATA_PATH+'*'))
	
	#cls_clr = {'ship': (0,0,1), 'person':(0,1,1), 'other':(1,0,0)}
	cls_clr = {'ship': (255,0,0), 'person':(255,255,0), 'other':(0,0,255), 'negative':(255,255,255)}
	
	#if args.sequences is None:
	#   args.sequences = np.arange(len(seq))
	#sequences = [s['path'] for s in gt['sequences']]
	#print(sequences)
	#return
		
	#seq_id = 2
	#proper_id = find_seq_id_in_list(seq_id, gt['sequences'])
	#print(proper_id)
	proper_id = 59

	try:
		os.mkdir(output_dir)
	except:
		pass

	skip = 1

	print(len(gt['sequences']))

	#if proper_id is not None:
	# for s in gt['sequences'][proper_id:proper_id+1]: 
	for s in gt['sequences']: 
		print(s['path'])			  
		
		idx = find_seq_id(s['path'],gt['sequences'])

		s_nm = s['path'].split('/')[1]
		try:
			os.mkdir(output_dir+'/'+s_nm)
		except:
			pass

		for f_gt in s['frames'][::skip]:
			# get image
			plt.clf()
			#plt.subplot(1,2,1)
			axes = plt.gca()
			axes.get_xaxis().set_visible(False)
			axes.get_yaxis().set_visible(False)
			img_name = f_gt['image_file_name']
			im_fn = args.data_path+s['path']+img_name
			img = cv2.imread(im_fn)
			
			ann = f_gt['obstacles'] 
			for a in ann:
				bb = a['bbox']  
				cv2.rectangle(img, (bb[0],bb[1]), (bb[0]+bb[2],bb[1]+bb[3]), cls_clr[a['type']], 2)

			se = f_gt['water_edges']
			for e in se:
				if not isinstance(e['x_axis'],list):
					continue
				# print(e)
				points = np.array([[x,y] for x,y in zip(e['x_axis'], e['y_axis'])], dtype=np.int32)
				cv2.polylines(img, [points], False, (0,255,102), 2)
			
			# plt.pause(0.01)
			# cv2.imshow("a", img)
			# cv2.waitKey(0)

			cv2.imwrite(output_dir+'/'+s_nm+'/'+img_name, img)
			
			# break
		#break
			
	# shutil.make_archive('modd3_annotations', 'zip', output_dir)
		
	
	
if __name__ == "__main__":
	main()

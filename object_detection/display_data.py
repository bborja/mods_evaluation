import os, math
import cv2
import json
import shutil
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# from detection_utils import read_gt_file, code_mask_to_labels, code_labels_to_colors, resize_image, read_json
from detection_utils import read_json
# from visualize_image import visualize_single_image, visualize_image_for_video
import glob

SEQUENCES = None
FRAME = None
EXPORT_VIDEO = True
OUTPUT_PATH = './results/video_output/'
RESULTS_PATH = './results'
DATA_PATH = '/media/jon/disk1/viamaro_data/modd3_full'
cls_clr = {'ship': (255,0,0), 'person':(255,255,0), 'other':(0,0,255)}
cls_clr_list = [(255,0,0),(0,128,255), (0,0,255)]

w = 1278
h = 958

def eulerAnglesToRotationMatrix(theta):
	theta = [np.radians(x) for x in theta]
	
	R_x = np.array([[1, 0, 0],
					[0, math.cos(theta[0]), -math.sin(theta[0]) ],
					[0, math.sin(theta[0]), math.cos(theta[0]) ]
					])
		
		
					
	R_y = np.array([[math.cos(theta[1]), 0,math.sin(theta[1]) ],
					[0, 1,0],
					[-math.sin(theta[1]),0,math.cos(theta[1]) ]
					])
				
	R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
					[math.sin(theta[2]), math.cos(theta[2]), 0],
					[0, 0,1]
					])
					
	R = np.dot(R_x, np.dot( R_y, R_z ))

	return R

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

def mask_from_sea_edge(sea_edge, size):
	mask = np.zeros(size[0:2])

	for e in sea_edge:
		#print(e)
		axx = e['x_axis']
		axy = e['y_axis']
		if axx and axy and isinstance(axx, list) and isinstance(axy, list):
			axx.insert(0, int(axx[0]))
			axy.insert(0, 0)
			axx.append(int(axx[-1]))
			axy.append(0)
			c = np.array([[int(x),int(y)] for x,y in zip(axx,axy)])
			cv2.fillPoly(mask, pts=[c], color=(255,255,255))
	return mask
	
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

def get_danger_zone(im, roll, pitch, height, rnge, M, D):
	# get the projection of a circle with radius range onto the sea surface (the surface being estimated from IMU)
	A,B,C,D = plane_from_IMU(roll,pitch, height)
	# print(A,B,C,D)
	# range = 10
	#print(roll)

	N = 1000
	r = np.linspace(-180,0,N)
	x = np.sin(np.radians(r))*rnge
	y = np.cos(np.radians(r))*rnge
	# z = np.zeros(N)+height
	z = np.zeros(N)
	#R = eulerAnglesToRotationMatrix([np.radians(roll),np.radians(pitch), 0])
	#R = eulerAnglesToRotationMatrix([np.radians(-roll),np.radians(pitch), 0])
	R = eulerAnglesToRotationMatrix([-roll, -pitch, 0])
	# print(R)

	pts = np.array([x,y,z])
	pts = R.dot(pts)
	pts = np.array([-pts[1,:],-pts[2,:],pts[0,:]])

	#z = np.zeros(N)+(-1/C)
	#z = z*(A*x+B*y+D)

	# points = np.transpose(np.array([-y,-z, x]))
	points = np.transpose(pts)
	pp, _ = cv2.projectPoints(points, np.identity(3), np.zeros([1,3]), M, distCoeffs=D)

	for p in pp:
		x = int(p[0,0])
		y = int(p[0,1])

		if x>0 and x<w and y>0 and y<h:
			# print(x,y)
			cv2.circle(im,(x,y),1,(255,0,0),-1)

	return im

def danger_zone_to_mask(roll, pitch, height, rnge, M, D, w, h):
	mask = np.zeros([h,w],dtype=np.uint8)

	A,B,C,D = plane_from_IMU(roll, pitch, height)

	N = 1000
	r = np.linspace(0,180,N)
	x = np.sin(np.radians(r))*rnge
	y = np.cos(np.radians(r))*rnge
	# z = np.zeros(N)-height
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
			
	cv2.fillPoly(mask, np.array([poly]), color=255)

	return mask

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

def in_mask(mask, rect, thr=0.9):
	roi = mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]/255
	#print("ROI",roi)
	s = np.sum(roi,-1)
	s = np.sum(s)
	a = rect[2]*rect[3]
	#print(s,a,s/a)
	iou = s/a
	#print("IOU", iou)

	return iou>thr

def export_all_detections():
	data_path = '/media/jon/disk1/viamaro_data/modd3_full/'
	gt_path = data_path+'modd3.json'
	detectors = [('mrcnn', '/home/jon/Desktop/Mask_RCNN/'), ('yolo','/home/jon/Desktop/darknet/'), ('fcos','/home/jon/Desktop/FCOS/'), ('ssd', '/home/jon/Desktop/ssd.pytorch/')]
	# output_dir = 'detections'
	output_dir = 'detections_mrcnn'

	try:
		os.mkdir(output_dir)
	except:
		pass

	dets = {}
	for d, p in detectors:
		data = read_json(p+d+'_res.json')
		#print(data)
		dets[d]=data

	gt=read_json(gt_path)['dataset']

	step = 1

	# for s in gt['sequences'][15:16]:
	for s in gt['sequences']:
		s_nm = s['path'].split('/')[1]
		try:
			os.mkdir(output_dir+'/'+s_nm)
		except:
			pass
		
		print(s_nm)

		for idx,f in enumerate(s['frames'][::step]):
			idx = idx*step
			img_name = f['image_file_name']
			im_fn = data_path+s['path']+img_name
			# print("im_fn", im_fn)
			img = cv2.imread(im_fn)
			# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


			# plt.clf()

			seq_id = next((index for (index, d) in enumerate(gt['sequences']) if d["id"] == s['id']), None)
			mask = mask_from_sea_edge(f['water_edges'], img.shape)

			anns = f['obstacles']
			imgs = []

			for i, d in enumerate(detectors):
				#print(i,d)

				cur = []
				I = img.copy()

				# print(dets[d[0]])

				# plt.subplot(2,2,i+1)
				# plt.imshow(img)
				# plt.imshow(mask, cmap='gray', alpha=0.3)
				# plt.title(d[0])

				textSize = cv2.getTextSize(d[0], fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, thickness=2)
				# print(textSize)
				# I = cv2.putText(I, d[0], (int(I.shape[0]/2)+int(textSize[0][0]/2),150), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,0), 2, cv2.LINE_AA)

				for a in anns:
					bb = a['bbox']
					x = int(bb[0]); y = int(bb[1]);	w = int(bb[2]);	h = int(bb[3])
	
					# plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=1,edgecolor='k',facecolor='none'))
					I = cv2.rectangle(I, (x,y), (x+w,y+h), (0,255,255), 2)

				if 'detections' in dets[d[0]]['dataset']['sequences'][seq_id]['frames'][idx]:
					cur = dets[d[0]]['dataset']['sequences'][seq_id]['frames'][idx]['detections']

				for c in cur:					
					bb = c['bbox']
					x = int(bb[0]); y = int(bb[1]);	w = int(bb[2]);	h = int(bb[3])
					clr = cls_clr[c['type']] if isinstance(c['type'],str) else cls_clr_list[c['type']]
					# plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=2,edgecolor=clr,facecolor='none'))
					I = cv2.rectangle(I, (x,y), (x+w,y+h), clr, 3)

				imgs.append(I)

				

				# cur = dets[d[0]]['dataset']['sequences'][seq_id]['frames'][idx]['detections']
				# print(cur)

			# im1 = np.concatenate((imgs[0], imgs[1]), axis=1)
			# im2 = np.concatenate((imgs[2], imgs[3]), axis=1)
			# final = np.concatenate((im1,im2),axis=0)
			final = imgs[0] # just MRCNN
			# cv2.imshow("a", final)
			# cv2.waitKey(0)
			cv2.imwrite(output_dir+'/'+s_nm+'/'+img_name, final)


			# plt.draw()
			# plt.pause(0.01)
			# plt.waitforbuttonpress()

			
			# dz = danger_zone_to_mask(roll, pitch, 0.5, rnge, M1, D1, w, h)
			# dz = np.copy(img); dz = get_danger_zone(dz, roll, pitch, imu_height, rnge, M1, D1)
			# plt.imshow(img)
			
			# plt.draw()
			# plt.pause(0.01)
			# plt.waitforbuttonpress()

def unrectify_annotations(mapx, obstacles):

	obstacles_ = []

	# undistort
	for i,bbox in enumerate(obstacles):
		p1 = (bbox[0],bbox[1])
		p2 = (bbox[0]+bbox[2],bbox[1]+bbox[3])
		# print(p1,p2)
		p1 = mapx[max(0,min(p1[1],h-1)), max(0,min(p1[0],w-1))]
		p2 = mapx[max(0,min(p2[1],h-1)), max(0,min(p2[0],w-1))]

		p1 = tuple(p1)
		p2 = tuple(p2)
		o = tuple([p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1]])
		obstacles_.append(o)

	return obstacles_

def main():

	#export_all_detections()
	#return

	w = 1278
	h = 958
	
	args = get_arguments()
	# res_path = '/home/jon/Desktop/Mask_RCNN/mrcnn_res.json'
	res_path = '/media/jon/disk1/jon/Desktop/3d-obstacle-detector/matlab/modd3_evaluation/res.json'
	#res_path = '/home/jon/Desktop/darknet/yolo_res.json'
	
	# gt = read_json(args.data_path+'/modd3.json')
	gt = read_json('/media/jon/disk1/jon/Desktop/viamaro-obstacle-annotator/modd3_fix/modd3.json')
	gt = gt['dataset']
	
	res_data = read_json(res_path)
	res_data = res_data['dataset']
	
	print(len(res_data['sequences']))
		
	#return
	
	#seq = sorted(glob.glob(DATA_PATH+'*'))
	
	cls_clr = {'ship': (0,0,1), 'person':(0,1,1), 'other':(1,0,0), 'negative':(1,1,1)}
	
	#if args.sequences is None:
	#	args.sequences = np.arange(len(seq))
	sequences = [s['path'] for s in res_data['sequences']]
	print(sequences)
	# return

	disp_ann = True
	disp_det = False
	step = 1
		
	seq_id = 5
	frame_start = 35
	# print(sequences[seq_id])
	# idx = find_seq_id('/kope104-00074750-00075530/frames/',gt['sequences'])
	# print(idx)
	# return
	for s in gt['sequences'][seq_id:seq_id+1]:
	# for s in gt['sequences']:
		#print(seq_id)
		#s = seq[seq_id]		
		print(s['path'])
		#print(s['id'])
		
		c = s['path'].split('-')
		c = c[0][1:]
		calib_file = DATA_PATH+'/calibration/calibration-'+c+'.yaml'

		M1, M2, D1, D2, R, T = load_calibration(calib_file)
		R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(M1, D1, M2, D2, (w,h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
		mapxL, mapyL = cv2.initUndistortRectifyMap(M1, D1, R1, P1, (w,h), cv2.CV_16SC2)
		
		# print(mapxL)
		
		idx = find_seq_id(s['path'],gt['sequences'])
		#print(idx)
		try:
			s_pred = res_data['sequences'][idx]
		except:
			continue

		
		for f_gt, f_pred in zip(s['frames'][frame_start::step],s_pred['frames'][frame_start::step]):
			# get image
			print(f_pred)
			
			img_name = f_gt['image_file_name']
			im_fn = args.data_path+s['path']+img_name
			print("im_fn", im_fn)
			img = cv2.imread(im_fn)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			# img = cv2.remap(img, mapxL, mapyL, cv2.INTER_LINEAR)
			#print(f_pred)
			
			roll = f_gt['roll']
			pitch = f_gt['pitch']
			yaw = f_gt['yaw']
			# pitch = -20
			
			#if pitch<5:
			#	continue
			
			print("roll: ", roll,"pitch: ", pitch,"yaw: ", yaw)

			plt.clf()
			# plt.subplot(1,3,1)

			mask = mask_from_sea_edge(f_gt['water_edges'], img.shape)
			
			if disp_ann:
				ann = f_gt['obstacles']
				#print('annotations')
				for a in ann:
					print(a)
					bb = a['bbox']	
					#print(bb)
					if not in_mask(mask, bb):
						plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=2,edgecolor='g',facecolor='none'))
					else:
						plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=2,edgecolor='r',facecolor='none'))

					# plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=2,edgecolor=cls_clr[a['type']],facecolor='none'))
			
			if disp_det:
				# dets = f_pred['detections']
				dets = f_pred['obstacles']
				# print(type(dets))
				if not isinstance(dets,list) and bool(dets):
					dets = [dets]
				if not bool(dets):
					dets = []
				# print(type(dets))
				# print(dets)
				for d in dets:
					bb = d['bbox']

					#x = int(bb[1]); y = int(bb[0]);	w = int(bb[3]);	h = int(bb[2])
					print(bb)
					x = int(bb[0]); y = int(bb[1]);	w = int(bb[2]);	h = int(bb[3])
					# bb = unrectify_annotations(mapxL, [[x,y,w,h]])[0]
					# x = int(bb[0]); y = int(bb[1]);	w = int(bb[2]);	h = int(bb[3])

					# if not in_mask(mask, bb):
					# 	plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=2,edgecolor='g',facecolor='none'))
					# else:
					# 	plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=2,edgecolor='r',facecolor='none'))
					plt.gca().add_patch(Rectangle((x,y),w,h,linewidth=2,edgecolor=cls_clr[d['type']],facecolor='none'))	

			
			#print(os.path.join('/derp', s['path'], img_name))
			w = 1278
			h = 958
			rnge = 30
			imu_height = 0.5
			
			plt.imshow(img)
			plt.imshow(mask, alpha=0.3)
			# plt.subplot(1,3,2)			
			# dz = danger_zone_to_mask(roll, pitch, 0.5, rnge, M1, D1, w, h)
			# dz = np.copy(img); dz = get_danger_zone(dz, roll, pitch, imu_height, rnge, M1, D1)
			#(im, roll, pitch, height, rnge, M, D):
			# plt.imshow(dz)

			# rnge = 30
			# plt.subplot(1,3,3)			
			# dz = danger_zone_to_mask(roll, pitch, 0.5, rnge, M1, D1, w, h)
			# dz1 = np.copy(img); dz1 = get_danger_zone(dz1, roll, pitch, imu_height, rnge, M1, D1)
			# plt.imshow(dz1)
			# plt.imshow(dz, cmap='gray', alpha=0.3)
			
			plt.draw()
			plt.pause(0.01)
			plt.waitforbuttonpress()
			#break
			
			# get gt annotations
			# get detections
		
			#break
		
		#img = cv2.imread(os.path.join(args.data_path, results['sequences'][args.sequences-1]['frames'][args.frame-1]['img_name']))
		
		#print(os.path.join(args.data_path, 'seq%s' % s, 'annotations.json'))
		#gt = read_gt_file(os.path.join(args.data_path, '%s' % s, 'annotations.json'))
		continue
		
		for f in gt['sequence']['frames']:
			plt.clf()
			#print(f)
			img_name = f['image_file_name']
			
			im_fn = os.path.join(args.data_path, s, 'frames', img_name)
			#print(im_fn)
			img = cv2.imread(os.path.join(args.data_path, s, 'frames', img_name))
			#print(img.shape)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			
			for ob in f['obstacles']:
				#print(ob)
				bb = ob['bbox']			
				plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2]-bb[0],bb[3]-bb[1],linewidth=2,edgecolor=cls_clr[ob['type']],facecolor='none'))
			
			
			plt.imshow(img)
			plt.draw()
			plt.pause(0.01)
		
		#print(gt)
	
	
if __name__ == "__main__":
	main()
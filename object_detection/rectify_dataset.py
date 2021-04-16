import os, cv2, glob, json, shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import read_json

w = 1278
h = 958
DATA_PATH = '/media/jon/disk1/viamaro_data/modd3_full'
cls_clr = {'ship': (0,0,1), 'person':(0,1,1), 'other':(1,0,0)}

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

def rectify_annotations(ann, M, D, R, P):
	res = []
	for a in ann:
		bbox = [float(x) for x in a['bbox']]
		p = []
		p.append([bbox[0], bbox[1]])
		p.append([bbox[0]+bbox[2], bbox[1]+bbox[3]])
		p = np.array(p)
		p = np.expand_dims(p, 1)
		b = cv2.undistortPoints(p, M, D, R=R, P=P)
		x = b[0,0,0]
		y = b[0,0,1]
		w_ = b[1,0,0]-x
		h_ = b[1,0,1]-y
		bbox = [x,y,w_,h_]
		
		# clip boxes beyond image edge
		if bbox[0] < 0:
			bbox[2]=bbox[2]+bbox[0]
			bbox[0]=0
		if bbox[1] < 0:		
			bbox[3]=bbox[3]+bbox[1]
			bbox[1]=0
		if bbox[0]+bbox[2] > w:
			# adjust width
			bbox[2]=w-bbox[0]
		if bbox[1]+bbox[3] > h:
			# adjust height
			bbox[3]=h-bbox[1]
		
		bbox = tuple([int(x) for x in bbox])
		a['bbox']=bbox
		#if x < w and y < h and x >= 0 and y >= 0:
		if x < w and y < h and bbox[2]>10 and bbox[3]>10:
			res.append(a)
	return res
	
def rectify_water_edge(water_edge, M, D, R, P):
	res = []

	for se in water_edge:
		x = se['x_axis']
		y = se['y_axis']
		pts = np.transpose(np.array((x,y))).astype(np.float32)
		pts = np.expand_dims(pts, 1)
		b = cv2.undistortPoints(pts, M, D, R=R, P=P)

		se['x_axis'] = b[:,0,0].tolist()
		se['y_axis'] = b[:,0,1].tolist()
		#se['x_axis'] = np.clip(se['x_axis'],0,w).tolist()
		#se['y_axis'] = np.clip(se['y_axis'],0,h).tolist()
		res.append(se)		
		
	return res

def mask_from_sea_edge(sea_edge, size):
	mask = np.zeros(size[0:2])

	for e in sea_edge:
		axx = e['x_axis']
		axy = e['y_axis']
		if axx and axy:
			axx.insert(0, int(axx[0]))
			axy.insert(0, 0)
			axx.append(int(axx[-1]))
			axy.append(0)
			c = np.array([[int(x),int(y)] for x,y in zip(axx,axy)])
			cv2.fillPoly(mask, pts =[c], color=(255,255,255))
	return mask

def main():

	res_path = '/home/jon/Desktop/Mask_RCNN/mrcnn_res.json'
	#res_path = '/home/jon/Desktop/darknet/yolo_res.json'
	
	gt = read_json(DATA_PATH+'/modd3.json')
	gt = gt['dataset']

	step = 10
		
	for s in gt['sequences'][1:]:	
		print(s['path'])
		
		c = s['path'].split('-')
		c = c[0][1:]
		calib_file = DATA_PATH+'/calibration/calibration-'+c+'.yaml'

		M1, M2, D1, D2, R, T = load_calibration(calib_file)
		R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(M1, D1, M2, D2, (w,h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
		mapxL, mapyL = cv2.initUndistortRectifyMap(M1, D1, R1, P1, (w,h), cv2.CV_16SC2)
		
		for f_gt in s['frames'][::step]:

			plt.clf()
			img_name = f_gt['image_file_name']
			im_fn = DATA_PATH+s['path']+img_name

			img = cv2.imread(im_fn)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			
			img_rect = cv2.remap(img, mapxL, mapyL, cv2.INTER_LINEAR)			

			plt.subplot(1,3,1)
			
			ann = f_gt['obstacles']
			for a in ann:
				bb = a['bbox']
				plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=1,edgecolor='r',facecolor='none'))
				
			se = f_gt['water_edges']			
			for e in se:
				plt.plot(e['x_axis'],e['y_axis'],'g')

			plt.imshow(img)
			plt.subplot(1,3,2)
			plt.imshow(img_rect)
			
			ann_rect = rectify_annotations(ann, M1, D1, R1, P1)
			se_rect = rectify_water_edge(se, M1, D1, R1, P1)
			
			for e in se_rect:
				plt.plot(e['x_axis'],e['y_axis'],'g')
			
			for a in ann_rect:
				bb = a['bbox']
				plt.gca().add_patch(Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=1,edgecolor='r',facecolor='none'))
				
			plt.subplot(1,3,3)
			mask = mask_from_sea_edge(se_rect, img.shape)
			plt.imshow(mask)
			
			plt.draw()
			plt.pause(0.01)
			plt.waitforbuttonpress()		
	
	
if __name__ == "__main__":
	main()
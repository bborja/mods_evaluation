

import json, os, cv2
import numpy as np

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

def in_mask(mask, rect, thr=0.5):
	# check if IoU of mask and rect is at least thr
	rect = [int(round(x)) for x in rect]
	roi = mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
	s = np.sum(roi,-1)
	s = np.sum(s)
	a = rect[2]*rect[3]
	iou = s/a
	return iou>thr

def iou_overlap(rect, gt, thr=0.3):
	# check overlaps of current rectangle with all rectangles in gt
	res = []

	for g in gt:
		gt_bb = g['bbox']
		iou = get_iou(gt_bb, rect)
		res.append(iou)

	return [x if x>thr else 0 for x in res]

def danger_zone_to_mask(roll, pitch, height, rnge, M, D, w, h):
	mask = np.ones([h,w],dtype=np.uint8)*255

	A,B,C,D = plane_from_IMU(roll, pitch, height)

	N = 1000
	r = np.linspace(0,180,N)
	x = np.sin(np.radians(r))*rnge
	y = np.cos(np.radians(r))*rnge
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

def mask_from_sea_edge(sea_edge, size):
	mask = np.zeros(size[0:2])

	for e in sea_edge:
		axx = e['x_axis']
		axy = e['y_axis']

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

def read_json(file_name):
	with open(file_name) as f:
		data = json.load(f)

	return data

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
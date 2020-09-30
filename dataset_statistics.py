import numpy as np
import argparse
import os
import cv2
import sys
import json
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from colorama import Fore, Back, Style
from colorama import init
from datetime import datetime
from danger_zone import plane_from_IMU, danger_zone_to_mask

from scipy.stats import norm
from sklearn.neighbors import KernelDensity

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Path to the MODB dataset
DATA_PATH = "E:/MODB/raw"
OBSTACLE_SIZE_CLASSES = [5*5, 15*15, 30*30, 50*50, 100*100, 200*200]


def get_arguments():
    """ Parse all the arguments provided from the CLI
    Returns: A list of parsed arguments
    """
    parser = argparse.ArgumentParser(description='Marine Obstacle Detection Benchmark.')
    parser.add_argument("--data-path", type=str, default=DATA_PATH,
                        help="Absolute path to the folder where MODB sequences are stored.")

    return parser.parse_args()


def get_dataset_statistics():
    init()  # initialize colorama

    # Get input arguments
    args = get_arguments()
    args.data_path = os.path.normpath(args.data_path)

    # Load danger zone masks
    # danger_zone_15 = cv2.imread('E:/MODB/danger_zone_15.png', cv2.IMREAD_GRAYSCALE)
    # danger_zone_30 = cv2.imread('E:/MODB/danger_zone_30.png', cv2.IMREAD_GRAYSCALE)

    # Read ground truth file
    with open(os.path.normpath(os.path.join(args.data_path, 'modb.json'))) as f:
        gt = json.load(f)

    # PieChart data
    num_gt_object_annotations = np.zeros(3)  # numbers of all and each class annotations
    # BarPlot data
    sizes_v = []  # sizes of "vessel" annotations
    sizes_o = []  # sizes of "other" annotations
    sizes_p = []  # sizes of "person" annotations
    # Size classes for each type of obstacles
    size_classes_obstacles = np.zeros((3, 7)) # Vesel, Other, Person
    num_images_wateredge = np.zeros(2)  # number of images where water edge is/is not annotated
    num_images_obstacles = np.zeros(3)  # number of images where obstacles are/are not annotated
    # Heat mask of obstacles
    obstacles_heat_mask = np.zeros((958, 1278))
    heat_mask_v = np.zeros((958, 1278))
    heat_mask_o = np.zeros((958, 1278))
    heat_mask_p = np.zeros((958, 1278))
    # Number of wateredge pieces in total
    num_wateredges = 0
    # Number of obstacles within each danger zone section
    num_obstacles_danger = np.zeros((3, 3))

    num_sequences = gt['dataset']['num_seq']

    # Loop through all sequences
    for seq_num in range(num_sequences):
        # Get number of annotated frames inside the sequence
        tmp_num_frames = gt['dataset']['sequences'][seq_num]['num_frames']
        # Loop through the frames inside the sequence

        # Get sequence path
        seq_path = gt['dataset']['sequences'][seq_num - 1]['path']
        # Strip path and get sequence name
        seq_name = seq_path.rstrip().split('/')[1]
        calib_id = seq_name.split('-')[0]
        # Get calibration file name
        calib_file = 'E:/MODB/calibration-%s.yaml' % calib_id

        """ Load camera calibration file """
        # Read calibration file
        fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
        M = fs.getNode("M1").mat()  # Extract calibration matrix (M)
        D = fs.getNode("D1").mat()  # Extract distortion coefficients (D)

        for fr_num in range(tmp_num_frames):
            """ Generate danger zone... """
            roll = gt['dataset']['sequences'][seq_num]['frames'][fr_num]['roll']  # Get IMU roll
            pitch = gt['dataset']['sequences'][seq_num]['frames'][fr_num]['pitch']  # Get IMU pitch
            danger_zone_15 = danger_zone_to_mask(roll, pitch, 0.7, 15, M, D, 1278, 958)  # 15m
            danger_zone_30 = danger_zone_to_mask(roll, pitch, 0.7, 30, M, D, 1278, 958)  # 30m

            tmp_obstacles = gt['dataset']['sequences'][seq_num]['frames'][fr_num]['obstacles']
            tmp_wateredge = gt['dataset']['sequences'][seq_num]['frames'][fr_num]['water_edges']

            tmp_num_obstacles = len(tmp_obstacles)  # number of annotated obstacles in frame
            tmp_num_wateredge = len(tmp_wateredge)  # number of annotated water edges in frame
            
            num_wateredges += tmp_num_wateredge
            
            #if gt['dataset']['sequences'][seq_num]['frames'][fr_num]['all_annotations'] == 'true':
            if 'exhaustive' in gt['dataset']['sequences'][seq_num]['frames'][fr_num].keys():
                if gt['dataset']['sequences'][seq_num]['frames'][fr_num]['exhaustive']:
                    tmp_all_annotations = True
                else:
                    tmp_all_annotations = False
            else:
                tmp_all_annotations = False

            if tmp_num_wateredge > 0:
                num_images_wateredge[0] += 1
            else:
                num_images_wateredge[1] += 1

            if tmp_num_obstacles > 0 and tmp_all_annotations == True:
                num_images_obstacles[0] += 1
            elif tmp_num_obstacles > 0 and tmp_all_annotations == False:
                num_images_obstacles[1] += 1
            elif tmp_num_obstacles == 0:
                num_images_obstacles[2] += 1

            for obs_num in range(tmp_num_obstacles):
                tmp_area = gt['dataset']['sequences'][seq_num]['frames'][fr_num]['obstacles'][obs_num]['area']
                tmp_bbox = gt['dataset']['sequences'][seq_num]['frames'][fr_num]['obstacles'][obs_num]['bbox']
                tmp_type = gt['dataset']['sequences'][seq_num]['frames'][fr_num]['obstacles'][obs_num]['type']

                if tmp_type == 'ship':
                    sizes_v.append(tmp_area)
                    num_gt_object_annotations[0] += 1
                    heat_mask_v[tmp_bbox[1]:tmp_bbox[1]+tmp_bbox[3], tmp_bbox[0]:tmp_bbox[0]+tmp_bbox[2]] += 1
                    size_classes_obstacles = update_detection_by_sizes(size_classes_obstacles, 0, tmp_area)
                    tmp_type_number = 0
                elif tmp_type == 'other':
                    sizes_o.append(tmp_area)
                    num_gt_object_annotations[1] += 1
                    heat_mask_o[tmp_bbox[1]:tmp_bbox[1] + tmp_bbox[3], tmp_bbox[0]:tmp_bbox[0] + tmp_bbox[2]] += 1
                    size_classes_obstacles = update_detection_by_sizes(size_classes_obstacles, 1, tmp_area)
                    tmp_type_number = 1
                elif tmp_type == 'person':
                    sizes_p.append(tmp_area)
                    num_gt_object_annotations[2] += 1
                    heat_mask_p[tmp_bbox[1]:tmp_bbox[1] + tmp_bbox[3], tmp_bbox[0]:tmp_bbox[0] + tmp_bbox[2]] += 1
                    size_classes_obstacles = update_detection_by_sizes(size_classes_obstacles, 2, tmp_area)
                    tmp_type_number = 2

                obstacles_heat_mask[tmp_bbox[1]:tmp_bbox[1]+tmp_bbox[3], tmp_bbox[0]:tmp_bbox[0]+tmp_bbox[2]] += 1
                
                tmp_bbox[0] -= 1
                tmp_bbox[1] -= 1
                tmp_bbox[2] -= 1
                tmp_bbox[3] -= 1
                
                tmp_bbox[0] = max(tmp_bbox[0], 0)
                tmp_bbox[1] = max(tmp_bbox[1], 0)
                tmp_bbox[2] = min(tmp_bbox[0]+tmp_bbox[2], 1277) - tmp_bbox[0]
                tmp_bbox[3] = min(tmp_bbox[1]+tmp_bbox[3], 957) - tmp_bbox[1]
                
                if(danger_zone_15[tmp_bbox[1]+tmp_bbox[3], tmp_bbox[0]+tmp_bbox[2]] == 255 and
                   danger_zone_15[tmp_bbox[1]+tmp_bbox[3], tmp_bbox[0]] == 255):
                    num_obstacles_danger[tmp_type_number][0] += 1
                elif(danger_zone_30[tmp_bbox[1]+tmp_bbox[3], tmp_bbox[0]+tmp_bbox[2]] == 255 and
                     danger_zone_30[tmp_bbox[1]+tmp_bbox[3], tmp_bbox[0]] == 255 or
                     danger_zone_30[tmp_bbox[1]+tmp_bbox[3], tmp_bbox[0]+tmp_bbox[2]] == 255 and
                     danger_zone_15[tmp_bbox[1]+tmp_bbox[3], tmp_bbox[0]] == 255 or
                     danger_zone_15[tmp_bbox[1]+tmp_bbox[3], tmp_bbox[0]+tmp_bbox[2]] == 255 and
                     danger_zone_30[tmp_bbox[1]+tmp_bbox[3], tmp_bbox[0]] == 255):
                    num_obstacles_danger[tmp_type_number][1] += 1
                else:
                    num_obstacles_danger[tmp_type_number][2] += 1

        print('Sequence %02d / %02d finished.' % (seq_num, num_sequences))

    explode_2 = (0.1, 0.1)  # Only explode slice belonging to the TPs
    explode_3_1 = (0.1, 0.1, 0.1)
    explode_3_2 = (0.1, 0.1, 0)
    pie_chart_obstacles1_labels = ['Images with obstacles', 'Images without all obstacles annotated', 'Images without obstacles']
    pie_chart_obstacles2_labels = ['Vessel', 'Other', 'Person'] 
    pie_chart_wateredge_labels = ['Water edge annotated', 'Water edge not annotated']
    
    wateredge_percentages = [num_images_wateredge[0]/(num_images_wateredge[0]+num_images_wateredge[1]),
                             num_images_wateredge[1]/(num_images_wateredge[0]+num_images_wateredge[1])]
    obstacles_percentages = [num_images_obstacles[0]/(num_images_obstacles[0]+num_images_obstacles[1]+num_images_obstacles[2]),
                             num_images_obstacles[1]/(num_images_obstacles[0]+num_images_obstacles[1]+num_images_obstacles[2]),
                             num_images_obstacles[2]/(num_images_obstacles[0]+num_images_obstacles[1]+num_images_obstacles[2])]
    
    types_percentages = [num_gt_object_annotations[0]/(num_gt_object_annotations[0]+num_gt_object_annotations[1]+num_gt_object_annotations[2]),
                         num_gt_object_annotations[1]/(num_gt_object_annotations[0]+num_gt_object_annotations[1]+num_gt_object_annotations[2]),
                         num_gt_object_annotations[2]/(num_gt_object_annotations[0]+num_gt_object_annotations[1]+num_gt_object_annotations[2])]
    
    print('Total number of obstacles: %d' % (num_gt_object_annotations[0]+num_gt_object_annotations[1]+num_gt_object_annotations[2]))
    print('Total number of vessels: %d' % num_gt_object_annotations[0])
    print('Total number of other obstacles: %d' % num_gt_object_annotations[1])
    print('Total number of person: %d' % num_gt_object_annotations[2])
    print('Total number of wateredge annotations: %d' % num_wateredges)
    print('Total number of obstacles closer than 15m: %d' % (num_obstacles_danger[0][0] + num_obstacles_danger[1][0] + num_obstacles_danger[2][0]))
    print('Total number of obstacles further than 15m and closer than 30m: %d' % (num_obstacles_danger[0][1] + num_obstacles_danger[1][1] + num_obstacles_danger[2][1]))
    print('Total number of obstacles further than 30m: %d' % (num_obstacles_danger[0][2] + num_obstacles_danger[1][2] + num_obstacles_danger[2][2]))
    print(num_obstacles_danger)

    plt.figure(1)
    plt.clf()
    # Plot PieChart of number images where water edge is annotated
    plt.subplot(3, 3, 1)
    plt.pie(wateredge_percentages, explode=explode_2, labels=pie_chart_wateredge_labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Frames with/without water-edge visible')
    # Plot PieChart of number images where obstacles are annotated
    plt.subplot(3, 3, 2)
    plt.pie(obstacles_percentages, explode=explode_3_1, labels=pie_chart_obstacles1_labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Frames with/without obstacles visible')
    # Plot PieChart of number of obstacles
    plt.subplot(3, 3, 3)
    plt.pie(types_percentages, explode=explode_3_2, labels=pie_chart_obstacles2_labels, autopct='%1.1f%%', shadow=True, startangle=90)
    # Plot bar graphs of obstacle sizes
    plt.subplot(3, 3, 4)
    plt.hist(sizes_v, bins=10)
    plt.title('Histogram of vessel sizes')
    plt.subplot(3, 3, 5)
    plt.hist(sizes_o, bins=10)
    plt.title('Histogram of other sizes')
    plt.subplot(3, 3, 6)
    plt.hist(sizes_p, bins=10)
    plt.title('Histogram of person sizes')
    #
    x_axis = np.arange(7)
    plt.subplot(3, 3, 7)
    plt.bar(x_axis, size_classes_obstacles[0, :])
    plt.title('Histogram of vessel sizes')
    plt.subplot(3, 3, 8)
    plt.bar(x_axis, size_classes_obstacles[1, :])
    plt.title('Histogram of other sizes')
    plt.subplot(3, 3, 9)
    plt.bar(x_axis, size_classes_obstacles[2, :])
    plt.title('Histogram of person sizes')
    
    
    plt.figure(2)
    plt.clf()
    plt.imshow(obstacles_heat_mask, cmap='Blues')
    plt.title('Heat mask for all obstacles')
    
    plt.figure(3)
    plt.clf()
    plt.subplot(131)
    plt.imshow(heat_mask_v, cmap='Oranges')
    plt.title('Heat mask for vessels')
    plt.subplot(132)
    plt.imshow(heat_mask_o, cmap='Oranges')
    plt.title('Heat mask for other')
    plt.subplot(133)
    plt.imshow(heat_mask_p, cmap='Oranges')
    plt.title('Heat mask for person')
    
    plt.show()
    
    
# Function parses through the list of detections and checks in which size class does the detection fall into
def update_detection_by_sizes(det_list, type_index, bbox_area):
    # type_index: 0 = Vessel, 1 = Other, 2 = Person
    # sizes_list: list that we are updating
    # bbox_area: obstacle size
    for j in range(len(OBSTACLE_SIZE_CLASSES) + 1):
        if j == 0 and bbox_area <= OBSTACLE_SIZE_CLASSES[j]:
            det_list[type_index, j] += 1
        elif j == len(OBSTACLE_SIZE_CLASSES) and bbox_area > OBSTACLE_SIZE_CLASSES[j-1]:
            det_list[type_index, j] += 1
        elif OBSTACLE_SIZE_CLASSES[j - 1] < bbox_area <= OBSTACLE_SIZE_CLASSES[j]:
            det_list[type_index, j] += 1

    return det_list


if __name__ == '__main__':
    get_dataset_statistics()

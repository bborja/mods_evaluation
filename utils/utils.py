import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage import draw

from tqdm.auto import tqdm
from multiprocessing import Queue, Pool, RLock
from contextlib import contextmanager
import utils.context as ctx


# Function codes segmentation mask to labels
# 0 denotes obstacles
# 1 denotes water component
# 2 denotes sky component
def code_mask_to_labels(segmentation_mask, segmentation_colors):
    # Convert BGR to RGB
    segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2RGB)

    new_segmentation_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1]))

    # Add water labels
    new_segmentation_mask[(segmentation_mask[:, :, 0] == segmentation_colors[1][0]) &
                          (segmentation_mask[:, :, 1] == segmentation_colors[1][1]) &
                          (segmentation_mask[:, :, 2] == segmentation_colors[1][2])] = 1
    # Add sky labels
    new_segmentation_mask[(segmentation_mask[:, :, 0] == segmentation_colors[2][0]) &
                          (segmentation_mask[:, :, 1] == segmentation_colors[2][1]) &
                          (segmentation_mask[:, :, 2] == segmentation_colors[2][2])] = 2

    return new_segmentation_mask


# Function converts segmentation mask labels to our default color-coded output
def code_labels_to_colors(segmentation_mask, cfg):
    # Initialize new color-coded segmentation mask
    segmentation_mask_new = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), np.uint8)

    segmentation_mask_new_r = segmentation_mask_new[:, :, 0]
    segmentation_mask_new_g = segmentation_mask_new[:, :, 1]
    segmentation_mask_new_b = segmentation_mask_new[:, :, 2]

    # Fill new segmentation mask with colors
    segmentation_mask_new_r[segmentation_mask == 0] = cfg.SEGMENTATIONS.OBS_LABEL[0]
    segmentation_mask_new_g[segmentation_mask == 0] = cfg.SEGMENTATIONS.OBS_LABEL[1]
    segmentation_mask_new_b[segmentation_mask == 0] = cfg.SEGMENTATIONS.OBS_LABEL[2]

    segmentation_mask_new_r[segmentation_mask == 1] = cfg.SEGMENTATIONS.WAT_LABEL[0]
    segmentation_mask_new_g[segmentation_mask == 1] = cfg.SEGMENTATIONS.WAT_LABEL[1]
    segmentation_mask_new_b[segmentation_mask == 1] = cfg.SEGMENTATIONS.WAT_LABEL[2]

    segmentation_mask_new_r[segmentation_mask == 2] = cfg.SEGMENTATIONS.SKY_LABEL[0]
    segmentation_mask_new_g[segmentation_mask == 2] = cfg.SEGMENTATIONS.SKY_LABEL[1]
    segmentation_mask_new_b[segmentation_mask == 2] = cfg.SEGMENTATIONS.SKY_LABEL[2]

    segmentation_mask_new_r[segmentation_mask == 3] = cfg.SEGMENTATIONS.OVS_LABEL[0]
    segmentation_mask_new_g[segmentation_mask == 3] = cfg.SEGMENTATIONS.OVS_LABEL[1]
    segmentation_mask_new_b[segmentation_mask == 3] = cfg.SEGMENTATIONS.OVS_LABEL[2]

    segmentation_mask_new_r[segmentation_mask == 4] = cfg.SEGMENTATIONS.UNS_LABEL[0]
    segmentation_mask_new_g[segmentation_mask == 4] = cfg.SEGMENTATIONS.UNS_LABEL[1]
    segmentation_mask_new_b[segmentation_mask == 4] = cfg.SEGMENTATIONS.UNS_LABEL[2]

    segmentation_mask_new[:, :, 0] = segmentation_mask_new_r
    segmentation_mask_new[:, :, 1] = segmentation_mask_new_g
    segmentation_mask_new[:, :, 2] = segmentation_mask_new_b

    # Convert RGB to BGR
    # segmentation_mask_new = cv2.cvtColor(segmentation_mask_new, cv2.COLOR_RGB2BGR)

    return segmentation_mask_new


# Function expands regions above the ground-truth water-edge
def expand_land(gt_mask, eval_params):
    # how many pixels this is based on the width of the image
    amount = np.ceil(eval_params['expand_land'] * gt_mask.shape[1]).astype(int)
    # construct kernel

    # kernel for only horizontal expansion
    # tmp_kernel = np.zeros((amount * 2 + 1, amount * 2 + 1), dtype=np.uint8)
    # tmp_kernel[amount, :] = 1

    # kernel for both horizontal and vertical expansion
    # tmp_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (amount * 2 + 1, amount * 2 + 1))  # Cross Kernel
    tmp_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (amount * 2 + 1, amount * 2 + 1))  # Ellipse Kernel

    gt_mask_new = cv2.dilate(gt_mask, kernel=tmp_kernel)

    return gt_mask_new


# Function computes binary obstacle mask, where obstacles in the mask are marked with ones, while sky and sea are
#   marked with zeros
def generate_obstacle_mask(segmentation_mask, gt_obstacles):
    # Initialize the obstacle mask
    obstacle_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1]))
    # Add the obstacle component
    obstacle_mask[segmentation_mask == 0] = 1

    # Get number of obstacles
    num_obstacles = len(gt_obstacles)
    # Loop through obstacles and check if some of them is "negative" aka ignore region
    for i in range(num_obstacles):
        if gt_obstacles[i]['type'] == 'negative':
            tmp_bbox = gt_obstacles[i]['bbox']
            # Paint over this area as a non-obstacle label
            # Extra: Expand box a bit, otherwise some FPs still go through
            obstacle_mask[tmp_bbox[1]-10:np.min([segmentation_mask.shape[0], tmp_bbox[3]+10]), tmp_bbox[0]-10:tmp_bbox[2]+10] = 0

    return obstacle_mask


# Function computes binary water mask from the segmentation output
#   The water mask is used for water-edge estimation
#   Pixels, belonging to the water component are denoted with ones
def generate_water_mask(segmentation_mask):
    # Initialize the water mask
    water_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1]))
    # Fill the water mask with ones where the water component is located
    water_mask[segmentation_mask == 1] = 1

    return water_mask


# Function filters obstacle mask (removes obstacles that area smaller than the surface area (eval['area_threshold'])
#   Remove such obstacles by simply painting over them...
def filter_obstacle_mask(obstacle_mask, eval_params):
    # Extract connected components from the obstacle mask.
    # The extracted blobs represent potential detections
    tmp_labels = measure.label(obstacle_mask)
    tmp_region_list = measure.regionprops(tmp_labels)

    num_regions = len(tmp_region_list)
    # Loop through the extracted blobs
    for i in range(num_regions):
        # Check if the surface area is sufficiently large enough
        # If not, then paint over the blob with zero values (aka non-obstacle)
        if tmp_region_list[i].area < eval_params['area_threshold']:
            obstacle_mask[tmp_region_list[i].bbox[0]:tmp_region_list[i].bbox[2],
                          tmp_region_list[i].bbox[1]:tmp_region_list[i].bbox[3]] = 0

    return obstacle_mask


# Function computes surface of the bounding box
def compute_surface_area(obstacle_bb):
    # Get height
    obstacle_bb_h = obstacle_bb[3] - obstacle_bb[1] + 1
    # Get width
    obstacle_bb_w = obstacle_bb[2] - obstacle_bb[0] + 1

    # Compute surface area
    obstacle_bb_area = obstacle_bb_h * obstacle_bb_w

    return obstacle_bb_area


# Function resizes image to a given size using nearest-neighbour interpolation
def resize_image(img, size):
    # Resize
    img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

    return img


# Function generates a binary mask from the provided polygon coordinates
def poly2mask(row_coordinates, col_coordinates, shape):
    # Generate list of coordinates, that need to be filled
    fill_row_coords, fill_col_coords = draw.polygon(row_coordinates, col_coordinates, shape)
    # Initialize a binary mask
    mask = np.zeros(shape)
    # Fill the binary mask with ones inside the provided polygon
    mask[fill_row_coords, fill_col_coords] = 1

    return mask


# Function reads ground truth txt file and parses it into appropriate format for further use
def read_gt_file(txt_path, is_coverage_file=False):
    # Read ground-truth JSON file
    with open(txt_path) as f:
        data = json.load(f)

    if is_coverage_file:
        return data
    else:
        return prepare_gt_obs_annotations(data)


# Function writes results to JSON file
def write_json_file(output_path, method_name, json_content):
    output_file_name = 'results_%s.json' % method_name
    with open(os.path.join(output_path, output_file_name), 'w') as json_file:
        json.dump(json_content, json_file)


# Function prepares ground truth annotation of obstacles
def prepare_gt_obs_annotations(gt):
    num_sequences = gt['dataset']['num_seq']
    for cur_seq in range(num_sequences):
        num_frames = gt['dataset']['sequences'][cur_seq]['num_frames']
        for fr in range(num_frames):
            num_obstacles = len(gt['dataset']['sequences'][cur_seq]['frames'][fr]['obstacles'])
            for i in range(num_obstacles):
                tmp_bb = np.array(np.round(gt['dataset']['sequences'][cur_seq]['frames'][fr]['obstacles'][i]['bbox'])).astype(int)
                tmp_bb[2] += tmp_bb[0]  # Change width to right-most point of a bounding-box
                tmp_bb[3] += tmp_bb[1]  # Change height to bottom-most point of a bounding-box
                gt['dataset']['sequences'][cur_seq]['frames'][fr]['obstacles'][i]['bbox'] = tmp_bb  # Update annotations

            num_water_edges = len(gt['dataset']['sequences'][cur_seq]['frames'][fr]['water_edges'])
            for i in range(num_water_edges):
                tmp_x_values = np.array(np.round(gt['dataset']['sequences'][cur_seq]['frames'][fr]['water_edges'][i]['x_axis']))
                tmp_y_values = np.array(np.round(gt['dataset']['sequences'][cur_seq]['frames'][fr]['water_edges'][i]['y_axis']))
                gt['dataset']['sequences'][cur_seq]['frames'][fr]['water_edges'][i]['x_axis'] = tmp_x_values
                gt['dataset']['sequences'][cur_seq]['frames'][fr]['water_edges'][i]['y_axis'] = tmp_y_values

    return gt


# Get usv-camera-imu calibration offsets
def get_calibration_offsets(cfg, seq_name):
    seq_id = int(seq_name[-2::])

    if seq_id <= 3:
        offset_pitch = cfg.OFFSETS.CAM_IMU_CALIB[0][0]
        offset_roll  = cfg.OFFSETS.CAM_IMU_CALIB[0][1]
    elif 4 <= seq_id <= 8:
        offset_pitch = cfg.OFFSETS.CAM_IMU_CALIB[1][0]
        offset_roll  = cfg.OFFSETS.CAM_IMU_CALIB[1][1]
    elif 9 <= seq_id <= 22:
        offset_pitch = cfg.OFFSETS.CAM_IMU_CALIB[2][0]
        offset_roll  = cfg.OFFSETS.CAM_IMU_CALIB[2][1]
    elif 23 <= seq_id <= 47:
        offset_pitch = cfg.OFFSETS.CAM_IMU_CALIB[3][0]
        offset_roll  = cfg.OFFSETS.CAM_IMU_CALIB[3][1]
    elif 48 <= seq_id <= 65:
        offset_pitch = cfg.OFFSETS.CAM_IMU_CALIB[4][0]
        offset_roll  = cfg.OFFSETS.CAM_IMU_CALIB[4][1]
    elif 66 <= seq_id <= 67:
        offset_pitch = cfg.OFFSETS.CAM_IMU_CALIB[5][0]
        offset_roll  = cfg.OFFSETS.CAM_IMU_CALIB[5][1]
    else:
        offset_pitch = cfg.OFFSETS.CAM_IMU_CALIB[6][0]
        offset_roll  = cfg.OFFSETS.CAM_IMU_CALIB[6][1]

    return np.rad2deg(offset_pitch), np.rad2deg(offset_roll)


# Calculate root mean of squared errors provided in the list
def calculate_root_mean(elements_list):
    if elements_list.size != 0:
        return np.sqrt(np.mean(elements_list))
    else:
        return 0


# Generate list of all sequences in the dataset
def build_sequences_list(num_total_sequences_in_dataset):
    sequences_list = []

    for i in range(1, num_total_sequences_in_dataset+1):
        sequences_list.append({"id": i,
                               "evaluated": False,
                               "frames": []})

    return sequences_list


# Get accurate number of obstacles
def get_obstacle_count(np_list):
    if len(np_list) == 0:
        num_obstacles = 0

    elif np.ndim(np_list) == 1:
        num_obstacles = 1

    else:
        num_obstacles = np_list.shape[0]

    return num_obstacles


# Count the number of actual FP detections
def count_number_fps(fp_list):
    num_fps = 0

    num_entries = len(fp_list)
    for i in range(num_entries):
        num_fps += fp_list[i]['num_triggers']

    return num_fps


# Build dict for sequence image names mapping
def build_mapping_dict(data_path):
    mapping_dict = {}
    # Read data
    f = open(os.path.join(data_path, 'sequence_mapping.txt'), "r")
    for x in f:
        y = x.rstrip().split(' ')
        mapping_dict.update({y[0]: y[1]})

    return mapping_dict


def tqdm_pool_initializer(q,lock,initializer,args):
    # Set process id, tqdm lock
    ctx.set_pid(q.get())
    tqdm.set_lock(lock)

    if initializer is not None:
        initializer(*args)

@contextmanager
def TqdmPool(processes, initializer=None, initargs=None, *args, **kwargs):
    """Wrapper of multiprocessing.Pool, suitable for use with tqdm. Workers are numbered in a global variable `PROC_I`."""

    q = Queue()
    for i in range(processes):
        q.put(i)

    with Pool(processes, initializer=tqdm_pool_initializer, initargs=(q,RLock(),initializer,initargs), *args, **kwargs) as p:
        yield p

import numpy as np
import argparse
import os
import sys
import cv2
import time
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from colorama import Fore, Back, Style
from colorama import init
from datetime import datetime
from utils import generate_water_mask, generate_obstacle_mask, write_json_file, read_gt_file, resize_image, \
                  code_mask_to_labels, build_sequences_list, poly2mask, expand_land, build_mapping_dict
from danger_zone import plane_from_IMU, danger_zone_to_mask
from detect_wateredge import evaluate_water_edge
from detect_obstacles import detect_obstacles_modb
from scipy.stats import norm
from skimage import measure

# Default paths
# Path to the MODB dataset
DATA_PATH = "E:/MODB/raw"
# Path to the output segmentations
SEGMENTATION_PATH = "E:/MODB_output"
# Path to the output evaluation results folder
OUTPUT_PATH = "./results"
# Default segmentation colors (RGB format, where first row corresponds to obstacles, second row corresponds to water and
#   third row corresponds to the sky component
SEGMENTATION_COLORS = np.array([[  0,   0,   0],
                                [255,   0,   0],
                                [  0, 255,   0]])
# Default minimal overlap between two bounding boxes
MIN_OVERLAP = 0.5
# Default area threshold for obstacle detection
AREA_THRESHOLD = 5 * 5
# Percentage to expand all regions above the water-edge
EXPAND_LAND = 0.01
# Percentage to expand the obstacle
EXPAND_OBJS = 0.01


def get_arguments():
    """ Parse all the arguments provided from the CLI
    Returns: A list of parsed arguments
    """
    parser = argparse.ArgumentParser(description='Marine Obstacle Detection Benchmark.')
    parser.add_argument("--data-path", type=str, default=DATA_PATH,
                        help="Absolute path to the folder where MODB sequences are stored.")
    parser.add_argument("--segmentation-path", type=str, default=SEGMENTATION_PATH,
                        help="Absolute path to the output folder where segmentation masks are stored.")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH,
                        help="Output path where the results and statistics of the evaluation will be stored.")
    parser.add_argument("--method-name", type=str, required=True,
                        help="<required> Method name. This should be equal to the folder name in which the "
                             "segmentation masks are located")
    parser.add_argument("--sequences", type=int, nargs='+', required=False,
                        help="List of sequences on which the evaluation procedure is performed. Zero = all.")
    parser.add_argument("--segmentation-colors", type=int, default=SEGMENTATION_COLORS,
                        help="Segmentation mask colors corresponding to the three semantic labels. Given as 3x3 array,"
                             "where the first row corresponds to the obstacles color, the second row corresponds to the"
                             "water component color and the third row correspond to the sky component color, all"
                             "written in the RGB format.")
    parser.add_argument("--min-overlap", type=int, default=MIN_OVERLAP,
                        help="Minimal overlap between two bounding boxes")
    parser.add_argument("--area-threshold", type=int, default=AREA_THRESHOLD,
                        help="Area threshold for obstacle detection and consideration in evaluation.")
    parser.add_argument("--expand-land", type=int, default=EXPAND_LAND,
                        help="Percentage to expand all regions above the annotated water-edge.")
    parser.add_argument("--expand-objs", type=int, default=EXPAND_OBJS,
                        help="Percentage to expand all object regions when checking for FPs.")

    return parser.parse_args()


# Run the evaluation procedure on all sequences of the MODB sequences
# The results are stored in a single JSON file
def run_evaluation():
    init()  # initialize colorama
    args = get_arguments()

    # Norm paths...
    args.data_path = os.path.normpath(args.data_path)
    args.segmentation_path = os.path.normpath(args.segmentation_path)
    args.output_path = os.path.normpath(args.output_path)
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    eval_params = {
                   "min_overlap": args.min_overlap,
                   "area_threshold": args.area_threshold,
                   "expand_land": args.expand_land,
                   "expand_objs": args.expand_objs
                  }

    # Read ground truth annotations JSON file
    gt = read_gt_file(os.path.join(args.data_path, 'modb.json'))

    # List of sequences on which we will evaluate the method
    if args.sequences is None:
        args.sequences = np.arange(1, gt['dataset']['num_seq'] + 1)

    # Get current date and time
    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")

    # Initialize evaluation results dict
    evaluation_results = {
                          "method_name": args.method_name,
                          "date_time": date_time,
                          "parameters": {
                                          "min_overlap": "%0.2f" % args.min_overlap,
                                          "area_threshold": "%d" % args.area_threshold
                                        },
                          "sequences": build_sequences_list(gt['dataset']['num_seq']),
                         }
    
    # Initialize overlap results dict
    overlap_results = {
                       "method_name": args.method_name,
                       "date_time": date_time,
                       "overlap_perc_all": [],
                       "overlap_perc_dng": []
    }

    # Quick statistics
    # Displaying detection statistics (TP, FP, FN in and out of danger zone) and water-edge statistics
    total_detections = np.zeros((6, 1), np.int)  # TP, FP, FN, TP_D, FP_D, FN_D
    total_edge_approx = []  # RMSE
    total_over_under = np.zeros(2)  # Overshot, undershot water-edge (number of times when it was not exact)
    total_land_detections = np.zeros(2)  # Total number of TP/FN land detections

    # Quick statistics that show all overlaps with ground truth
    # This is useful information for selecting the thresholds...
    total_overlap_percentages = []
    total_overlap_percentages_d = []

    # Build name mapping dict for the current sequence
    mapping_dict_seq = build_mapping_dict(args.data_path)

    # Loop through sequences
    for seq_index_counter in range(len(args.sequences)):
        # Read txt file with a list of images with their corresponding ground truth annotation files
        seq_id = args.sequences[seq_index_counter]

        # Read number of frames for current sequence
        num_frames = gt['dataset']['sequences'][seq_id - 1]['num_frames']
        print_progress_bar(0, num_frames, prefix='Processing sequence %02d / %02d:' % (seq_index_counter + 1,
                                                                                       len(args.sequences)),
                           suffix='Complete', length=50)

        # Get sequence path
        seq_path = gt['dataset']['sequences'][seq_id - 1]['path']
        # Strip path and get sequence name
        seq_name = seq_path.rstrip().split('/')[1]
        # Get calibration ID
        calib_id = seq_name.split('-')[0]
        # Get calibration file name
        calib_file = 'E:/MODB/calibration-%s.yaml' % calib_id

        """ Load camera calibration file """
        # Read calibration file
        fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
        M = fs.getNode("M1").mat()  # Extract calibration matrix (M)
        D = fs.getNode("D1").mat()  # Extract distortion coefficients (D)

        """ Ignore mask for camera vignette (sequences 1, 2 and 3) """
        # Check if ignore mask is needed
        if seq_id == 1 or seq_id == 2 or seq_id == 3:
            # Read the ignore mask
            ignore_mask = cv2.imread(os.path.join(args.data_path, 'mask_seq123.png'), cv2.IMREAD_GRAYSCALE)
        else:
            ignore_mask = None

        # Loop through frames in the sequence
        for frame_number in range(num_frames):               
            # Update progress bar
            print_progress_bar(frame_number + 1, num_frames,
                               prefix='Processing sequence %02d / %02d:' % (seq_index_counter + 1, len(args.sequences)),
                               suffix='Complete', length=50)

            img_name = gt['dataset']['sequences'][seq_id - 1]['frames'][frame_number]['image_file_name']
            img_name_split = img_name.split('.')
            hor_name = '%s.png' % img_name_split[0]

            # Perform evaluation on the current image
            rmse, num_land_detections, ou_mask, tp_list, fp_list, fn_list, num_fps, \
             tp_list_d, fp_list_d, fn_list_d, num_fps_d, \
             overlap_percentages, overlap_percentages_d = run_evaluation_image(args.data_path,
                                                                               args.segmentation_path,
                                                                               args.segmentation_colors,
                                                                               args.method_name,
                                                                               gt['dataset']['sequences'][seq_id - 1],
                                                                               frame_number, eval_params,
                                                                               mapping_dict_seq, M, D,
                                                                               ignore_mask)

            total_overlap_percentages = total_overlap_percentages + overlap_percentages
            total_overlap_percentages_d = total_overlap_percentages_d + overlap_percentages_d
            
            num_oshoot_wateredge = np.sum(ou_mask == 1)
            num_ushoot_wateredge = np.sum(ou_mask == 2)

            """
            if num_oshoot_wateredge + num_ushoot_wateredge > 0:
                we_o = float(num_oshoot_wateredge / (num_oshoot_wateredge + num_ushoot_wateredge))
                we_u = float(num_ushoot_wateredge / (num_oshoot_wateredge + num_ushoot_wateredge))
            else:
                we_o = 0
                we_u = 0
            """

            # Add to the evaluation results
            evaluation_results['sequences'][seq_id - 1]['frames'].append({"we_rmse": rmse,
                                                                          "we_o": int(num_oshoot_wateredge),
                                                                          "we_u": int(num_ushoot_wateredge),
                                                                          "we_detections": num_land_detections,
                                                                          "obstacles": {"tp_list": tp_list,
                                                                                        "fp_list": fp_list,
                                                                                        "fn_list": fn_list},
                                                                          "obstacles_danger": {"tp_list": tp_list_d,
                                                                                               "fp_list": fp_list_d,
                                                                                               "fn_list": fn_list_d},
                                                                          "img_name": img_name,
                                                                          "hor_name": hor_name
                                                                          })

            # Update quick statistics
            total_detections[0] += len(tp_list)
            total_detections[1] += num_fps
            total_detections[2] += len(fn_list)
            total_detections[3] += len(tp_list_d)
            total_detections[4] += num_fps_d
            total_detections[5] += len(fn_list_d)

            total_edge_approx.append(rmse)
            total_over_under[0] += num_oshoot_wateredge
            total_over_under[1] += num_ushoot_wateredge
            total_land_detections += num_land_detections

        evaluation_results['sequences'][seq_id - 1]['evaluated'] = True

    overlap_results['overlap_perc_all'] = total_overlap_percentages
    overlap_results['overlap_perc_dng'] = total_overlap_percentages_d

    # Print quick statistics
    table = PrettyTable()

    tmp_edge = np.ceil(np.mean(total_edge_approx))
    if np.sum(total_over_under) > 0:
        tmp_oshot = (total_over_under[0] / (total_over_under[0] + total_over_under[1])) * 100
        tmp_ushot = (total_over_under[1] / (total_over_under[0] + total_over_under[1])) * 100
    else:
        tmp_oshot = 0
        tmp_ushot = 0

    wedge_line = '%d px ' + Fore.LIGHTRED_EX + '(+%.01f%%, ' + Fore.LIGHTYELLOW_EX + '-%.01f%%)' + Fore.WHITE
    wedge_line = wedge_line % (tmp_edge, tmp_oshot, tmp_ushot)
    
    if total_land_detections[0] + total_land_detections[1] > 0:
        water_land_detections = '%.01f%%' % ((total_land_detections[0] / (total_land_detections[0] + 
                                                                          total_land_detections[1])) * 100)
    else:
        water_land_detections = 100

    tp_line = Fore.LIGHTGREEN_EX + '%d (%d)' + Fore.WHITE
    tp_line = tp_line % (total_detections[0], total_detections[3])

    fp_line = Fore.LIGHTYELLOW_EX + '%d (%d)' + Fore.WHITE
    fp_line = fp_line % (total_detections[1], total_detections[4])

    fn_line = Fore.LIGHTRED_EX + '%d (%d)' + Fore.WHITE
    fn_line = fn_line % (total_detections[2], total_detections[5])

    f1_line = '%.01f%% (%.01f%%)' % ((((2 * total_detections[0]) / (2 * total_detections[0] + total_detections[1] +
                                                                    total_detections[2])) * 100),
                                     (((2 * total_detections[3]) / (2 * total_detections[3] + total_detections[4] +
                                                                    total_detections[5])) * 100))

    table.field_names = ['Water-edge RMSE', 'Water-Land detections', 'TPs', 'FPs', 'FNs', 'F1']
    table.add_row([wedge_line, water_land_detections, tp_line, fp_line, fn_line, f1_line])

    print(table.get_string(title="Results for method %s on %d sequence/s" % (args.method_name, len(args.sequences))))

    # Write the evaluation results to JSON file
    write_json_file(args.output_path, args.method_name, evaluation_results)
    # Write the overlap results to JSON file
    write_json_file(args.output_path, '%s_overlap' % args.method_name, overlap_results)


# Run evaluation on a single image
def run_evaluation_image(data_path, segmentation_path, seg_colors, method_name, gt, frame_number, eval_params,
                         mapping_dict_seq, M, D, ignore_mask=None):
    """ Function performs evaluation process on a single image
        * In the evaluation process we evaluate the water-edge accuracy and obstacle detection accuracy

    Input params: data_path - path to the MODB raw folder
                  segmentation_path - path to the folder where segmentation masks for all sequences are stored
                  seg_colors - segmentation mask colors (so we can decode them)
                  method_name - name of the method we want to evaluate. The segmentation masks for each sequence should
                                be stored in the parent folder of the same name as method
                  gt - ground truth dictonary
                  frame_number - number of frame which we evaluate
                  eval_param - evaluation parameter values
                  mapping_dict_seq - dictonary with mapping of sequences
                  M - camera calibration matrix
                  D - camera distortion coefficients vector

    Output: - RMSE of the water edge
            - vector of the number of [correctly and incorrectly] assigned water-edge pixels
            - mask where o-shot and u-shot water-edge regions are marked
            - list of true-positives
            - list of false-positives
            - list of false-negatives
            - total number of false-positives (some detections generate more than one false-positive)
            - list of true-positives inside the danger-zone
            - list of false-positives inside the danger zone
            - list of false-negatives inside the danger zone
            - number of false-positives inside the danger zone (some detections generate more than one false-positive)
            - list of overlapping percentages of detections and ground-truth
            - list of overlapping percetanges of detections and ground-truth inside the danger zone
    """

    """ Reading all necessary data... """
    # Read image name:
    img_name = gt['frames'][frame_number]['image_file_name']
    img_name_split = img_name.split('.')
    seq_path = gt['path']
    seq_path_split = seq_path.split('/')

    # Look-up name in dict:
    seq_name = mapping_dict_seq[seq_path_split[1]]
    
    # Read image
    img = cv2.imread(os.path.join(data_path + seq_path, img_name))

    # Read segmentation mask
    seg = cv2.imread(os.path.join(segmentation_path, seq_name,
                                  method_name, "%04d.png" % (frame_number*10)))

    # Read horizon mask generated with imu
    horizon_mask = cv2.imread(os.path.join(data_path, seq_path_split[1], 'imus', '%s.png' % img_name_split[0]),
                              cv2.IMREAD_GRAYSCALE)

    """ Generate danger zone... """
    roll = gt['frames'][frame_number]['roll']  # Get IMU roll
    pitch = gt['frames'][frame_number]['pitch']  # Get IMU pitch
    danger_zone_mask = danger_zone_to_mask(roll, pitch, 0.7, 15, M, D, 1278, 958)  # Generate binary danger-zone mask

    # Code mask to labels
    seg = code_mask_to_labels(seg, seg_colors)
    # Resize segmentation mask to match the image
    seg = resize_image(seg, (img.shape[1], img.shape[0]))

    """ Perform the evaluation """
    # Generate obstacle mask
    obstacle_mask = generate_obstacle_mask(seg, gt['frames'][frame_number]['obstacles'])
    # Check if ignore_mask is set (this should be set only for sequences 1, 2 and 3)
    if ignore_mask is not None:
        # Remove all detections that occur in the ignore area
        obstacle_mask = np.logical_and(obstacle_mask, np.logical_not(ignore_mask))

    """ Get connected components of obstacles """
    # Extract connected components from the obstacle mask.
    # The extracted blobs represent potential detections
    obstacle_mask_labels = measure.label(obstacle_mask)
    obstacle_region_list = measure.regionprops(obstacle_mask_labels)
    filtered_region_list = []
    # Delete obstacles that are smaller than threshold
    num_regions = len(obstacle_region_list)
    # Loop through the extracted blobs
    for i in range(num_regions):
        # Check if the surface area is sufficiently large enough
        # If not, then paint over the blob with zero values (aka non-obstacle)
        if obstacle_region_list[i].area < eval_params['area_threshold']:
            # BBOX given in Y_TL, X_TL, Y_BR, X_BR
            obstacle_mask_labels[obstacle_region_list[i].bbox[0]:obstacle_region_list[i].bbox[2],
                                 obstacle_region_list[i].bbox[1]:obstacle_region_list[i].bbox[3]] = 0
            obstacle_mask[obstacle_region_list[i].bbox[0]:obstacle_region_list[i].bbox[2],
                          obstacle_region_list[i].bbox[1]:obstacle_region_list[i].bbox[3]] = 0
        else:
            filtered_region_list.append(obstacle_region_list[i])

    # Perform the evaluation of the water-edge
    rmse, num_land_detections, ou_mask, land_mask, ignore_abv_strad = evaluate_water_edge(gt['frames'][frame_number],
                                                                                          obstacle_mask_labels,
                                                                                          horizon_mask,
                                                                                          eval_params)

    # Calculate GT mask (This is land mask with an extension of the undershot regions)
    gt_mask = (np.logical_or(land_mask, ou_mask == 2)).astype(np.uint8)
    # Expand the mask left and right by 1% of an image width
    gt_mask = expand_land(gt_mask, eval_params)
    
    # plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(ou_mask)
    # plt.subplot(122)
    # plt.imshow(gt_mask)
    # plt.show()

    """
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(obstacle_mask)
    plt.subplot(222)
    plt.imshow(obstacle_mask_danger)
    plt.subplot(223)
    plt.imshow(gt_mask)
    plt.subplot(224)
    plt.imshow(gt_mask_danger)
    """

    # Read flag if the sequence is exhaustively annotated (aka if all obstacles in the sequence are annotated)
    if gt['exhaustive'] == 1 and gt['frames'][frame_number]['exhaustive'] == 1:
        exhaustive_annotations = True
    else:
        # If not all obstacles in the sequence are annotated, then we shall ignore all false-positive detections,
        # since they may belong to a non-annotated obstacle
        exhaustive_annotations = False

    # Perform the evaluation of the obstacle detection
    tp_list, fp_list, fn_list, num_fps, overlap_percentages,\
    tp_list_d, fp_list_d, fn_list_d, num_fps_d, overlap_perc_d = detect_obstacles_modb(gt['frames'][frame_number],
                                                                                       obstacle_mask, gt_mask,
                                                                                       ignore_abv_strad,
                                                                                       horizon_mask, eval_params,
                                                                                       exhaustive_annotations,
                                                                                       danger_zone_mask)

    plt.show()

    return rmse, num_land_detections, ou_mask, tp_list, fp_list, fn_list, num_fps, tp_list_d, fp_list_d, fn_list_d,\
           num_fps_d, overlap_percentages, overlap_perc_d


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == '__main__':
    run_evaluation()

import numpy as np
import argparse
import os
import sys
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from utils import generate_water_mask, generate_obstacle_mask, write_json_file, read_gt_file, resize_image, \
                  code_mask_to_labels, build_sequences_list, poly2mask
from detect_wateredge import evaluate_water_edge
from detect_obstacles import detect_obstacles_modb

# Default paths
# Path to the MODB dataset
DATA_PATH = "F:/Projects/matlab/RoBoat/dataset_public"
# Path to the output segmentations
SEGMENTATION_PATH = "F:/Projects/matlab/RoBoat/dataset_public"
# Path to the output evaluation results folder
OUTPUT_PATH = "./results"
# Default segmentation colors (RGB format, where first row corresponds to obstacles, second row corresponds to water and
#   third row corresponds to the sky component
SEGMENTATION_COLORS = np.array([[  0,   0,   0],
                                [255,   0,   0],
                                [  0, 255,   0]])
# Default minimal overlap between two bounding boxes
MIN_OVERLAP = 0.15
# Default area threshold for obstacle detection
AREA_THRESHOLD = 5 * 5


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

    return parser.parse_args()


# Run the evaluation procedure on all sequences of the MODB sequences
# The results are stored in a single JSON file
def run_evaluation():
    args = get_arguments()

    # Norm paths...
    args.data_path = os.path.normpath(args.data_path)
    args.segmentation_path = os.path.normpath(args.segmentation_path)
    args.output_path = os.path.normpath(args.output_path)
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    eval_params = {
                   "min_overlap": args.min_overlap,
                   "area_threshold": args.area_threshold
                  }

    # List of sequences on which we will evaluate the method
    if args.sequences is None:
        args.sequences = np.arange(28)  # Array of all sequences (presume that we have 28 sequences

    # Get current date and time
    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")

    # Initialize evaluation results dict
    evaluation_results = {
                          "method-name": args.method_name,
                          "date-time": date_time,
                          "parameters": {
                                          "min_overlap": "%0.2f" % args.min_overlap,
                                          "area_threshold": "%d" % args.area_threshold
                                        },
                          "sequences": build_sequences_list(args.data_path)
                         }

    # Quick statistics
    total_detections = np.zeros((6, 1), np.int)  # TP, FP, FN, TP_D, FP_D, FN_D
    total_edge_aprox = np.zeros((3, 1), np.float)  # RMSE_t, RMSE_o, RMSE_u

    # Loop through sequences
    for seq_index_counter in range(len(args.sequences)):
        print("Evaluating sequence %02d / %02d...\n" % (seq_index_counter+1, len(args.sequences)))
        # Read txt file with a list of images with their corresponding ground truth annotation files
        seq_id = args.sequences[seq_index_counter]

        # Read ground truth annotations JSON file
        gt = read_gt_file(os.path.join(args.data_path, 'seq%02d' % seq_id, 'annotations.json'))
        num_frames = len(gt['sequence'])
        for frame_number in range(num_frames):
            # Print progress
            sys.stdout.write("\rProcessing image number %03d / %03d" % ((frame_number + 1), num_frames))
            # feed, so it erases the previous line.
            sys.stdout.flush()

            img_name = gt['sequence'][frame_number]['image_file_name']
            hor_name = gt['sequence'][frame_number]['horizon_file_name']

            # Perform evaluation on current image
            rmse_t, rmse_o, rmse_u, ou_mask, tp_list, fp_list, fn_list, \
            tp_list_d, fp_list_d, fn_list_d,  = run_evaluation_image(args.data_path,
                                                                     args.segmentation_path,
                                                                     args.segmentation_colors,
                                                                     args.method_name,
                                                                     gt, seq_id,
                                                                     frame_number, eval_params)

            # Add to the evaluation results

            evaluation_results['sequences'][seq_id - 1]['frames'].append({"rmse_t": int(rmse_t),
                                                                          "rmse_o": int(rmse_o),
                                                                          "rmse_u": int(rmse_u), #"over_under_mask": ou_mask,
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
            total_detections[1] += len(fp_list)
            total_detections[2] += len(fn_list)
            total_detections[3] += len(tp_list_d)
            total_detections[4] += len(fp_list_d)
            total_detections[5] += len(fn_list_d)

            total_edge_aprox = np.column_stack((total_edge_aprox, [rmse_t, rmse_o, rmse_u]))

        evaluation_results['sequences'][seq_id - 1]['evaluated'] = True

    # Print quick statistics
    print('\n')
    print('***********************')
    print('Evaluated %s on %02d sequences' % (args.method_name, len(args.sequences)))
    print('Total RMSE: %03d pixels' % np.mean(total_edge_aprox[0, :]))
    print('RMSE over:  %.01f percent' % (np.mean(total_edge_aprox[1, :]) / (np.mean(total_edge_aprox[1, :]) +
                                                                            np.mean(total_edge_aprox[2, :])) * 100))
    print('RMSE under: %.01f percent' % (np.mean(total_edge_aprox[2, :]) / (np.mean(total_edge_aprox[1, :]) +
                                                                            np.mean(total_edge_aprox[2, :])) * 100))
    print('Total TP:   %d  (%d)' % (total_detections[0], total_detections[3]))
    print('Total FP:   %d  (%d)' % (total_detections[1], total_detections[4]))
    print('Total FN:   %d  (%d)' % (total_detections[2], total_detections[5]))
    print('Total F1:   %.01f percent' % (((2 * total_detections[0]) / (2 * total_detections[0] + total_detections[1] +
                                                                       total_detections[2])) * 100))
    print('***********************')

    # Write the evaluation results to JSON file
    write_json_file(args.output_path, args.method_name, evaluation_results)


# Run evaluation on a single image
def run_evaluation_image(data_path, segmentation_path, seg_colors, method_name, gt, seq_id, frame_number,
                         eval_params):
    """ Reading data... """
    # Read image
    #print(os.path.join(data_path, 'seq%02d' % seq_id, 'frames',
    #                              gt['sequence'][frame_number]['image_file_name']))
    img = cv2.imread(os.path.join(data_path, 'seq%02d' % seq_id, 'frames',
                                  gt['sequence'][frame_number]['image_file_name']))
    img_name_split = gt['sequence'][frame_number]['image_file_name'].split(".")

    # Read horizon mask
    horizon_mask = cv2.imread(os.path.join(data_path, 'seq%02d' % seq_id, 'horizons',
                                           gt['sequence'][frame_number]['horizon_file_name']),
                              cv2.IMREAD_GRAYSCALE)

    # Read segmentation mask
    # seg = cv2.imread(os.path.join(segmentation_path, 'seq%02d' % seq_id, method_name, 'mask_%03d.png' % frame_number))
    #print(os.path.join(segmentation_path, 'seq%02d' % seq_id, method_name, "%s.png" % img_name_split[0]))
    seg = cv2.imread(os.path.join(segmentation_path, 'seq%02d' % seq_id, method_name, "%s.png" % img_name_split[0]))

    # Read danger zone
    danger_zone_x = gt['sequence'][frame_number]['danger_zone']['x-axis']
    danger_zone_y = gt['sequence'][frame_number]['danger_zone']['y-axis']
    # Build danger zone mask
    danger_zone_mask = poly2mask(danger_zone_y, danger_zone_x, (img.shape[0], img.shape[1]))


    # Code mask to labels
    seg = code_mask_to_labels(seg, seg_colors)
    # Resize segmentation mask to match the image
    seg = resize_image(seg, (img.shape[1], img.shape[0]))

    """ Perform the evaluation """
    # Generate obstacle mask
    obstacle_mask = generate_obstacle_mask(seg)
    # Generate water mask
    water_mask = generate_water_mask(seg)

    # Modify obstacle mask according to the danger-zone
    obstacle_mask_danger = (np.logical_and(obstacle_mask, danger_zone_mask)).astype(np.uint8)
    water_mask_danger = (np.logical_and(water_mask, danger_zone_mask)).astype(np.uint8)

    # Perform the evaluation of the water-edge
    rmse_t, rmse_o, rmse_u, ou_mask, land_mask = evaluate_water_edge(gt['sequence'][frame_number], water_mask,
                                                                     eval_params)

    # Calculate GT mask (This is land mask with an extension of the undershot regions)
    gt_mask = (np.logical_or(land_mask, ou_mask == 2)).astype(np.uint8)
    gt_mask_danger = (np.logical_or(np.logical_not(danger_zone_mask), gt_mask)).astype(np.uint8)

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

    # Perform the evaluation of the obstacle detection
    tp_list, fp_list, fn_list = detect_obstacles_modb(gt['sequence'][frame_number], obstacle_mask, gt_mask,
                                                      horizon_mask, eval_params)

    # Perform the evaluation of the obstacle detection inside the danger zone only
    tp_list_d, fp_list_d, fn_list_d = detect_obstacles_modb(gt['sequence'][frame_number], obstacle_mask_danger,
                                                            gt_mask_danger, horizon_mask, eval_params,
                                                            danger_zone=danger_zone_mask)

    # print('%d - %d' % (len(tp_list), len(tp_list_d)))
    # print('%d - %d' % (len(fp_list), len(fp_list_d)))
    # print('%d - %d' % (len(fn_list), len(fn_list_d)))
    plt.show()

    return rmse_t, rmse_o, rmse_u, ou_mask, tp_list, fp_list, fn_list, tp_list_d, fp_list_d, fn_list_d


if __name__ == '__main__':
    run_evaluation()

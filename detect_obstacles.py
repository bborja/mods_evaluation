import numpy as np
from skimage import measure
from utils import compute_surface_area, filter_obstacle_mask, get_obstacle_count
import cv2
import matplotlib.pyplot as plt


# Function performs obstacle detection evaluation and return a list of TP, FP and FN detections with their BBoxes
def detect_obstacles_modb(gt, obstacle_mask, gt_mask, horizon_mask, eval_params, danger_zone=None):

    # Filter GT obstacle annotations - keep only those that are inside the danger zone mask
    if danger_zone is not None:
        gt = filter_gt_danger_zone(gt, danger_zone, eval_params)

    # Filter the obstacles mask and extract obstacles from it
    _, obstacle_mask = filter_obstacle_mask(obstacle_mask, eval_params)

    # Perform the detections
    # - Detect TPs and FNs
    tp_list, fn_list = check_tp_detections(gt, obstacle_mask, eval_params)
    # - Extract FPs ( but only if there are all obstacles in the image annotated, otherwise do not report FPs)
    if gt['all-annotations']:
        fp_list = check_fp_detections(gt, obstacle_mask, gt_mask, horizon_mask, eval_params)
        #fp_list, num_fp_dets = check_fp_detections_2(gt, obstacle_mask, gt_mask, horizon_mask, eval_params)
    else:
        fp_list = []

    #print('******\n')
    #print(num_fp_dets)
    #print(fp_list)

    return tp_list, fp_list, fn_list  #, num_fp_dets


# Function checks TP detections (and consequently FN detections as well)
def check_tp_detections(gt, obstacle_mask_filtered, eval_params):
    # Initialize arrays
    # (Arrays will be of size n x 4, where each row is in format: x_TL,y_TL,x_BR,y_BR)
    # TL = top-left, BR = bottom-right
    tp_detections = []
    fn_detections = []

    # Get number of filtered obstacles
    num_gt_obstacles = len(gt['obstacles'])

    # Check for overlap between filtered ground truth annotations and filtered detections
    for i in range(num_gt_obstacles):
        # Check if obstacle is sufficiently large
        if gt['obstacles'][i]['area'] >= eval_params['area_threshold']:
            # Get current GT obstacle bounding-box
            tmp_obs = gt['obstacles'][i]['bbox']

            # Get surface area of the GT obstacle
            tmp_area_surf = gt['obstacles'][i]['area']

            # Extract bounding-box region from the filtered obstacle mask
            tmp_area = obstacle_mask_filtered[tmp_obs[1]:tmp_obs[3], tmp_obs[0]:tmp_obs[2]]

            # Check the coverage of the area (pixels labeled with ones belong to obstacles, while pixels labeled with
            #   zero belong to the navigable surface for the USV
            tmp_area_obstacles = np.sum(tmp_area)

            # Check if enough area is covered by an obstacle to be considered a TP detection
            if tmp_area_obstacles / tmp_area_surf >= eval_params['min_overlap']:
                # Add obstacle to the list of TP detections
                tp_detections.append({"bbox": tmp_obs.tolist(),
                                      "type": gt['obstacles'][i]['type'],
                                      "area": int(tmp_area_surf),
                                      "coverage": int(tmp_area_obstacles / tmp_area_surf * 100)})

            else:
                # Add obstacle to the list of FN detections
                fn_detections.append({"bbox": tmp_obs.tolist(),
                                      "type": gt['obstacles'][i]['type'],
                                      "area": int(tmp_area_surf),
                                      "coverage": int(tmp_area_obstacles / tmp_area_surf * 100)})

    # Return lists of TP and FN detections
    return tp_detections, fn_detections


# Function checks FP detections by searching for blobs that do not overlap with any ground truth annotation
def check_fp_detections_2(gt, obstacle_mask_filtered, gt_mask_filtered, horizon_mask, eval_params):
    # Initialize false positives mask with all detections
    fp_mask = obstacle_mask_filtered
    # Filter out detections that correspond with an expanded ground truth obstacles and land component
    fp_mask[gt_mask_filtered > 0] = 0

    # Extract connected component from the filtered FP_mask. These blobs represent potential false-positive detections
    tmp_labels = measure.label(fp_mask)
    tmp_region_list = measure.regionprops(tmp_labels)

    # Initialize list of false-positive detections
    # (Arrays will be of size n x 4, where each row is in format x_TL, y_TL, x_BR, y_BR (TL = top-L, BR = bottom-R)
    tmp_fp_list = np.array([])
    fp_list = []

    num_regions = len(tmp_region_list)
    for i in range(num_regions):
        # Check if obstacle is large enough
        tmp_bb = np.zeros(4)
        tmp_bb[0] = tmp_region_list[i].bbox[1]
        tmp_bb[1] = tmp_region_list[i].bbox[0]
        tmp_bb[2] = tmp_region_list[i].bbox[3]
        tmp_bb[3] = tmp_region_list[i].bbox[2]
        if tmp_region_list[i].area >= eval_params['area_threshold']:
            # Append the list
            if len(tmp_fp_list) == 0:
                tmp_fp_list = tmp_bb.astype(np.int)
            else:
                tmp_fp_list = np.row_stack((tmp_fp_list, tmp_bb.astype(np.int)))

    # Remove detections far above horizon
    tmp_fp_list = remove_above_horizon(tmp_fp_list, horizon_mask)

    num_fp_dets = get_obstacle_count(tmp_fp_list)

    num_obstacles = len(gt['obstacles'])

    num_all_fps = 0

    # Loop through the remaining detections and check their overlap with ground truth annotations
    # If multiple ground truth annotations are covered by a single detection, then check how big this detection is and
    #   how many FPs does it produce...
    for i in range(num_fp_dets):
        num_gts = 0
        largest_size_gts = 0
        if num_fp_dets > 1:
            bb_det = tmp_fp_list[i, :]
        else:
            bb_det = tmp_fp_list
        detection_mask = np.zeros(gt_mask_filtered.shape)
        detection_mask[bb_det[1]:bb_det[3],
                       bb_det[0]:bb_det[2]] = fp_mask[bb_det[1]:bb_det[3],
                                                      bb_det[0]:bb_det[2]]

        for i_gt in range(num_obstacles):
            bb_gt = gt['obstacles'][i_gt]['bbox']
            bb_gt_area = gt['obstacles'][i_gt]['area']

            intersection_bb = [np.max([bb_det[0], bb_gt[0]]),
                               np.max([bb_det[1], bb_gt[1]]),
                               np.min([bb_det[2], bb_gt[2]]),
                               np.min([bb_det[3], bb_gt[3]])]

            # Compute the surface area of the intersection BB
            intersection_bb_area = compute_surface_area(intersection_bb)
            bb_det_area = compute_surface_area(bb_det)

            # Compute the surface area of the union
            union_area = bb_det_area + bb_gt_area - intersection_bb_area

            # Calculate the intersection over union (aka overlap) score
            overlap_score = intersection_bb_area / union_area

            if overlap_score >= eval_params['min_overlap']:
                num_gts += 1
                if bb_gt_area > largest_size_gts:
                    largest_size_gts = bb_gt_area
                # Cut-out
                detection_mask[bb_gt[1]:bb_gt[3], bb_gt[0]:bb_gt[2]] = 0

        if num_gts > 0:
            fp_area_sum = np.sum(detection_mask)
            num_fps = np.floor(fp_area_sum / (num_gts * largest_size_gts))
            if num_fps >= 1:
                num_all_fps += num_fps
            else:
                num_all_fps += 1

            fp_list.append({"bbox": bb_det,
                            "area": int((bb_det[2] - bb_det[0]) * (bb_det[3] - bb_det[1]))})

    return fp_list, num_all_fps


# Function checks FP detections by searching for blobs that do not overlap with any ground truth annotation
def check_fp_detections(gt, obstacle_mask_filtered, gt_mask_filtered, horizon_mask, eval_params):
    # Initialize false positives mask with all detections
    fp_mask = obstacle_mask_filtered
    # Filter out detections that correspond with an expanded ground truth obstacles and land component
    fp_mask[gt_mask_filtered > 0] = 0

    # Extract connected components from the filtered FP_mask. These blobs represent potential false-positive detections
    tmp_labels = measure.label(fp_mask)
    tmp_region_list = measure.regionprops(tmp_labels)

    """
    plt.figure(22)
    plt.imshow(tmp_labels)
    plt.show()
    """

    # Initialize list of false-positive detections
    # (Arrays will be of size n x 4, where each row is in format: x_TL,y_TL,x_BR,y_BR)
    # TL = top-left, BR = bottom-right
    tmp_fp_list = np.array([])
    fp_list = []

    num_regions = len(tmp_region_list)
    for i in range(num_regions):
        # Check if obstacle is large enough
        tmp_bb = np.zeros(4)
        tmp_bb[0] = tmp_region_list[i].bbox[1]
        tmp_bb[1] = tmp_region_list[i].bbox[0]
        tmp_bb[2] = tmp_region_list[i].bbox[3]
        tmp_bb[3] = tmp_region_list[i].bbox[2]
        if tmp_region_list[i].area >= eval_params['area_threshold']:
            # Append the list
            if len(tmp_fp_list) == 0:
                tmp_fp_list = tmp_bb.astype(np.int)
            else:
                tmp_fp_list = np.row_stack((tmp_fp_list, tmp_bb.astype(np.int)))

    # Remove detections far above horizon
    tmp_fp_list = remove_above_horizon(tmp_fp_list, horizon_mask)

    # Additionally suppress detections that are close-by
    # TODO!!!!

    # Check overlapping with ground truth annotations
    num_fp_dets = get_obstacle_count(tmp_fp_list)

    assigned_fps = np.zeros((num_fp_dets, 1))
    num_obstacles = len(gt['obstacles'])
    for i_gt in range(num_obstacles):
        # Get current ground truth bounding-box annotation
        if gt['obstacles'][i_gt]['area'] >= eval_params['area_threshold']:
            bb_gt = gt['obstacles'][i_gt]['bbox']
            bb_gt_area = gt['obstacles'][i_gt]['area']

            for i_det in range(num_fp_dets):
                # Get current detection bounding box annotation
                if num_fp_dets == 1:
                    bb_det = tmp_fp_list
                else:
                    bb_det = tmp_fp_list[i_det, :]

                # First check if detection is fully contained within the bounding box of a ground truth annotation.
                # In such case, ignore the detection - it should not contribute towards FP count due to its proximity
                if bb_det[0] >= bb_gt[1] and bb_det[1] >= bb_gt[1] and bb_det[2] <= bb_gt[2] and bb_det[3] <= bb_gt[3]:
                    assigned_fps[i_det] = 1

                # Otherwise check if there is an intersection between the detected obstacle and the GT obstacle
                # If such overlap exists and if it is sufficiently large, then ignore such detection,
                # otherwise add it to a filtered list of false positive detections
                else:
                    # Compute intersection BB
                    intersection_bb = [
                                       np.max([bb_det[0], bb_gt[0]]),
                                       np.max([bb_det[1], bb_gt[1]]),
                                       np.min([bb_det[2], bb_gt[2]]),
                                       np.min([bb_det[3], bb_gt[3]])
                                      ]

                    # Compute the surface area of the intersection BB
                    intersection_bb_area = compute_surface_area(intersection_bb)

                    # Check if there is an overlap (aka both width and height must be positive)
                    if intersection_bb_area > 0:
                        # Compute the surface area of the detected obstacle
                        bb_det_area = compute_surface_area(bb_det)

                        # Compute the surface area of the union
                        union_area = bb_det_area + bb_gt_area - intersection_bb_area

                        # Calculate the intersection over union (aka overlap) score
                        overlap_score = intersection_bb_area / union_area

                        # Check if the overlap is sufficiently large
                        if overlap_score >= eval_params['min_overlap']:
                            # Its a match! Assign the detected obstacle
                            assigned_fps[i_det] = 1

    # Update the list of FP detections. (Remove all assigned detections - they belong to TPs)
    num_fp_dets = get_obstacle_count(tmp_fp_list)
    for i in range(num_fp_dets):
        if assigned_fps[i] == 0:
            if num_fp_dets == 1:
                fp_list.append({"bbox": tmp_fp_list.tolist(),
                                "area": int((tmp_fp_list[2] - tmp_fp_list[0]) * (tmp_fp_list[3] - tmp_fp_list[1]))
                                })
            else:
                fp_list.append({"bbox": tmp_fp_list[i, :].tolist(),
                                "area": int((tmp_fp_list[i, 2] - tmp_fp_list[i, 0]) *
                                            (tmp_fp_list[i, 3] - tmp_fp_list[i, 1]))
                                })

    return fp_list


# Filter GT based on danger zone mask
def filter_gt_danger_zone(gt, danger_zone_mask, eval_params):
    new_gt_list = []
    num_obs = len(gt['obstacles'])
    # print(num_obs)
    for i in range(num_obs):
        tmp_obs_bb = gt['obstacles'][i]['bbox']
        tmp_obs_mask = np.zeros((danger_zone_mask.shape[0], danger_zone_mask.shape[1]), dtype=np.uint8)
        tmp_obs_mask[tmp_obs_bb[1]:tmp_obs_bb[3], tmp_obs_bb[0]:tmp_obs_bb[2]] = 1
        tmp_obs_mask = (np.logical_and(tmp_obs_mask, danger_zone_mask)).astype(np.uint8)

        """
        plt.figure(1)
        plt.imshow(tmp_obs_mask + danger_zone_mask)
        plt.show()
        """

        if np.sum(tmp_obs_mask) is not 0:
            tmp_labels = measure.label(tmp_obs_mask)
            tmp_region_list = measure.regionprops(tmp_labels)

            if len(tmp_region_list) > 0 and tmp_region_list[0].area >= eval_params['area_threshold']:
                tmp_obs = gt['obstacles'][i]

                tmp_obs_bb_new = np.zeros(4, dtype=int)
                tmp_obs_bb_new[0] = tmp_region_list[0].bbox[1]
                tmp_obs_bb_new[1] = tmp_region_list[0].bbox[0]
                tmp_obs_bb_new[2] = tmp_region_list[0].bbox[3]
                tmp_obs_bb_new[3] = tmp_region_list[0].bbox[2]

                tmp_obs['bbox'] = tmp_obs_bb_new
                tmp_obs['area'] = tmp_region_list[0].area
                new_gt_list.append(tmp_obs)

    gt['obstacles'] = new_gt_list

    return gt


# Remove annotations way above the horizon. This detections should not count towards FPs as they do not affect the
#   navigation of the USV.
def remove_above_horizon(fp_list, horizon_mask):
    dilatation_type = cv2.MORPH_ELLIPSE
    dilatation_size = 150
    element = cv2.getStructuringElement(dilatation_type, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))

    above_horizon_mask = 1 - cv2.dilate(horizon_mask, element)

    filtered_fp_list = np.array([])

    num_fp_obstacles = get_obstacle_count(fp_list)

    if num_fp_obstacles == 1:
        tmp_det_surf = (fp_list[2] - fp_list[0] + 1) * (fp_list[3] - fp_list[1] + 1)
        tmp_det_covr = np.sum(above_horizon_mask[fp_list[1]:fp_list[3], fp_list[0]:fp_list[2]])
        if tmp_det_covr < tmp_det_surf:
            filtered_fp_list = fp_list
    else:
        for i in range(num_fp_obstacles):
            tmp_det = fp_list[i, :]
            tmp_det_surf = (tmp_det[2] - tmp_det[0] + 1) * (tmp_det[3] - tmp_det[1] + 1)
            tmp_det_covr = np.sum(above_horizon_mask[tmp_det[1]:tmp_det[3], tmp_det[0]:tmp_det[2]])

            if tmp_det_covr < tmp_det_surf:
                if len(filtered_fp_list) == 0:
                    filtered_fp_list = tmp_det
                else:
                    filtered_fp_list = np.row_stack((filtered_fp_list, tmp_det))

    return filtered_fp_list











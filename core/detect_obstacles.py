import numpy as np
from skimage import measure, transform, segmentation, draw

from utils import compute_surface_area, filter_obstacle_mask, get_obstacle_count
import cv2
import math
import matplotlib.pyplot as plt


# Function performs obstacle detection evaluation and return a list of TP, FP and FN detections with their BBoxes
def detect_obstacles_modb(gt, gt_coverage, obstacle_mask, gt_mask, ignore_abv_strad, horizon_mask, eval_params,
                          exhaustive, danger_zone):

    # Filter GT obstacle annotations - keep only those that are inside the danger zone mask
    # if danger_zone is not None:
    #     gt = filter_gt_danger_zone(gt, danger_zone, eval_params)

    # Perform the detections
    # - Detect TPs and FNs (both inside and outside the danger zone)
    tp_list, fn_list, overlap_percentages_list, \
        tp_list_d, fn_list_d, overlap_percentages_list_d = check_tp_detections(gt, gt_coverage, obstacle_mask,
                                                                               danger_zone, eval_params)

    gt_obstacles_list = np.append(tp_list, fn_list)

    # - Extract FPs ( but only if there are all obstacles in the image annotated, otherwise do not report FPs)
    if exhaustive:
        obstacle_mask, gt_ah, horizon_mask_lower = remove_above_horizon_2(obstacles_mask=obstacle_mask,
                                                                          groundtruth_list=gt_obstacles_list,
                                                                          horizon_mask=horizon_mask)
        obstacle_mask_f = np.logical_and(np.logical_not(ignore_abv_strad), obstacle_mask)
        fp_list, num_fps,\
            fp_list_d, num_fps_d = check_fp_detections(obstacle_mask_f, gt_obstacles_list, gt_mask, gt_ah,
                                                       horizon_mask_lower, danger_zone, eval_params)
    else:
        # The frame is not exhaustively annotated. We do not report False-Positive detections.
        fp_list = []
        fp_list_d = []

        num_fps = 0
        num_fps_d = 0

    return tp_list, fp_list, fn_list, num_fps, overlap_percentages_list, tp_list_d, fp_list_d, fn_list_d, num_fps_d, \
        overlap_percentages_list_d


# Function checks TP detections (and consequently FN detections as well)
def check_tp_detections(gt, gt_coverage, obstacle_mask_filtered, danger_zone, eval_params):
    # Initialize arrays
    # (Arrays will be of size n x 4, where each row is in format: x_TL,y_TL,x_BR,y_BR)
    # TL = top-left, BR = bottom-right
    tp_detections = []
    fn_detections = []
    # Detections inside the danger zone
    tp_detections_d = []
    fn_detections_d = []

    # We store here all overlaps with ground truth that are above 10%
    # This information is later used for setting the optimal TP detection threshold and other analysis
    overlap_percentages = []
    overlap_percentages_d = []

    # Get number of filtered obstacles
    num_gt_obstacles = len(gt['obstacles'])
    num_gt_obstacles_cover = len(gt_coverage['obstacles'])

    # Check for overlap between filtered ground truth annotations and filtered detections
    for i in range(num_gt_obstacles):
        # Check if obstacle is sufficiently large
        # gt_area_surface = compute_surface_area(gt['obstacles'][i]['bbox'])
        gt_area_surface = gt['obstacles'][i]['area']

        # Sanity check (dextr coverage):
        #if gt['obstacles'][i]['id'] == gt_coverage['obstacles'][i]['id']:
        #    print('all good in the hood')
        #else:
        #    print('there are many soci-economic problems in the hood')

        if gt_area_surface >= eval_params['area_threshold'] and gt['obstacles'][i]['type'] != 'negative':
            # Get current GT obstacle bounding-box
            if isinstance(gt['obstacles'][i]['bbox'], list):
                tmp_gt_obs = (np.array(gt['obstacles'][i]['bbox']).astype(np.int))
            else:
                tmp_gt_obs = (gt['obstacles'][i]['bbox']).astype(np.int)

            # Extract bounding-box region from the filtered obstacle mask
            obstacle_mask_area = obstacle_mask_filtered[tmp_gt_obs[1]:tmp_gt_obs[3], tmp_gt_obs[0]:tmp_gt_obs[2]]
            danger_zone_mask_area = danger_zone[tmp_gt_obs[1]:tmp_gt_obs[3], tmp_gt_obs[0]:tmp_gt_obs[2]]

            # Check the coverage of the area (pixels labeled with ones belong to obstacles, while pixels labeled with
            #   zero belong to the navigable surface for the USV
            num_correctly_detected_pixels = np.sum(obstacle_mask_area)

            # Check the coverage of the danger zone area (if the sum is non-zero, then the obstacle is located in d-z)
            num_dangerzone_pixels = np.sum(danger_zone_mask_area)

            # Dextr coverage...
            is_coverage_computed = False
            for i_c in range(num_gt_obstacles_cover):
                if gt['obstacles'][i]['id'] == gt_coverage['obstacles'][i_c]['id']:
                    is_coverage_computed = True
                    break

            if is_coverage_computed:
                # Get percentage of obstacle above water edge
                p_above_we = gt_coverage['obstacles'][i_c]['p_above_water_edge']
                # Get dextr obstacle coverage
                expected_coverage = gt_coverage['obstacles'][i_c]['expected_coverage']

                # Check if enough area is covered by an obstacle to be considered a TP detection
                if p_above_we < 0.8 and gt_area_surface > 250:
                    # The GT semantic mask of the obstacle may not align with the GT bounding-box of the obstacle
                    # (i.e., obstacle is non-convex and there are a lot of pixels inside the bounding-box annotation
                    #  that belong to the background).
                    # In such cases, we treat obstacle as correctly detected, if it covers at least:
                    #  min_overlap * gt_area_surface * expected_coverage of pixels. However, since segmentation mask
                    # is not always precise, there might be detected more pixels inside the GT bounding-box than
                    # gt_area_surface * expected_coverage. In that case, cap the overlap to 1.
                    correctly_covered_percentage = np.min([num_correctly_detected_pixels / (gt_area_surface *
                                                                                            expected_coverage + 1e-9), 1])
                else:
                    correctly_covered_percentage = np.min([num_correctly_detected_pixels / (gt_area_surface *
                                                                                            expected_coverage + 1e-9), 1])
            else:
                correctly_covered_percentage = num_correctly_detected_pixels / gt_area_surface

            # append the overlap percentage with information of the current obstacle...
            if correctly_covered_percentage >= 0.1:
                overlap_percentages.append(round(correctly_covered_percentage, 2))
                if num_dangerzone_pixels > 0:
                    overlap_percentages_d.append(round(correctly_covered_percentage, 2))

            if correctly_covered_percentage >= eval_params['min_overlap']:
                # Add obstacle to the list of TP detections
                tp_detections.append({"bbox": tmp_gt_obs.tolist(),
                                      "type": gt['obstacles'][i]['type'],
                                      "area": int(gt_area_surface),
                                      "coverage": int(correctly_covered_percentage * 100)})

                if num_dangerzone_pixels > 0:
                    tp_detections_d.append({"bbox": tmp_gt_obs.tolist(),
                                            "type": gt['obstacles'][i]['type'],
                                            "area": int(gt_area_surface),
                                            "coverage": int(correctly_covered_percentage * 100)})

            else:
                # Add obstacle to the list of FN detections
                fn_detections.append({"bbox": tmp_gt_obs.tolist(),
                                      "type": gt['obstacles'][i]['type'],
                                      "area": int(gt_area_surface),
                                      "coverage": int(correctly_covered_percentage * 100)})

                if num_dangerzone_pixels > 0:
                    fn_detections_d.append({"bbox": tmp_gt_obs.tolist(),
                                            "type": gt['obstacles'][i]['type'],
                                            "area": int(gt_area_surface),
                                            "coverage": int(correctly_covered_percentage * 100)})

    # Return lists of TP and FN detections
    return tp_detections, fn_detections, overlap_percentages, tp_detections_d, fn_detections_d, overlap_percentages_d


# Function checks FP detections by searching for blobs that do not overlap with any ground truth annotation
def check_fp_detections(obstacle_mask_filtered, gt_obstacle_list, gt_mask_filtered, gt_ah, horizon_mask, danger_zone,
                        eval_params):
    # Initialize false positives mask with all detections
    detections_mask = np.copy(obstacle_mask_filtered)
    detections_mask[gt_mask_filtered > 0] = 0

    detections_mask_labels = measure.label(detections_mask)
    detection_regions_list = measure.regionprops(detections_mask_labels)

    # detection_regions_list = group_near_by_detections(detection_regions_list, detection_labels)
    num_detections = len(detection_regions_list)

    # Get number of TP detections
    num_gt_obs = len(gt_obstacle_list)

    # Initialize FP list
    fp_list = []
    num_fps = 0
    # Inside Dangerzone
    fp_list_d = []
    num_fps_d = 0

    # Invert horizon mask
    horizon_mask = 1 - horizon_mask

    # Loop through all detections on screen
    for i in range(num_detections):
        # Get BBOX of current detection in format (Y_TL, X_TL, Y_BR, X_BR)
        tmp_detection_bbox = detection_regions_list[i].bbox

        # Calculate danger-zone overlap
        dangerzone_overlap = danger_zone[tmp_detection_bbox[0]:tmp_detection_bbox[2],
                                         tmp_detection_bbox[1]:tmp_detection_bbox[3]]
        # Coverage of danger-zone
        dangerzone_coverage = np.sum(dangerzone_overlap)

        # Get surface area of current detection
        #   Note: THIS IS NOT A BBOX SURFACE AREA
        tmp_detection_area = detection_regions_list[i].area

        # Get surface area of current detection
        #   Note: THIS IS A BBOX SURFACE AREA
        # tmp_detection_surf = np.round((tmp_detection_bbox[2] - tmp_detection_bbox[0]) *
        #                               (tmp_detection_bbox[3] - tmp_detection_bbox[1]))

        # Get detection label
        tmp_detection_label = detection_regions_list[i].label

        # Extract binary mask where only pixels belonging to the current detection label are set to one
        tmp_detection_mask = (detections_mask_labels == tmp_detection_label) * 1

        # Check if detection is large enough
        if tmp_detection_area > eval_params['area_threshold'] and \
                ((np.sum(horizon_mask[tmp_detection_bbox[1]:tmp_detection_bbox[3], tmp_detection_bbox[0]:tmp_detection_bbox[3]]) > 0 and np.sum(gt_ah[tmp_detection_bbox[3]:, tmp_detection_bbox[0]:tmp_detection_bbox[2]]) == 0) or
                 (np.sum(horizon_mask[tmp_detection_bbox[1]:tmp_detection_bbox[3], tmp_detection_bbox[0]:tmp_detection_bbox[3]]) == 0)):

            # Get width of the detection
            tmp_detection_width = tmp_detection_bbox[3] - tmp_detection_bbox[1]

            # Initialize counter of assigned GT obstacles
            assigned_gt_obstacles = 0
            assigned_gt_obstacles_partially = 0
            tmp_largest_gt_width = 0  # largest width of the assigned obstacle

            # Loop through all GT detections and check if we can assign any to current detection
            for j in range(num_gt_obs):
                # Get BBOX of current GT obstacle in format (X_TL, Y_TL, X_BR, Y_BR)
                tmp_gt_obs_bbox = gt_obstacle_list[j]['bbox']
                tmp_gt_width = tmp_gt_obs_bbox[2] - tmp_gt_obs_bbox[0]

                # Get surface area of current GT obstacle
                #   Note: THIS IS A BBOX SURFACE AREA
                tmp_gt_obs_area = gt_obstacle_list[j]['area']

                # Extract part where the GT obstacle is located
                tmp_gt_coverage = tmp_detection_mask[tmp_gt_obs_bbox[1]:tmp_gt_obs_bbox[3],
                                                     tmp_gt_obs_bbox[0]:tmp_gt_obs_bbox[2]]

                # Check if the current ground_truth annotation is covered by the current detection label

                # If GT obstacle is sufficiently covered, then assign this GT obstacle to current detection
                if np.sum(tmp_gt_coverage) / tmp_gt_obs_area > eval_params['min_overlap']:
                    # Subtract from the detection width the current width of the assigned GT obstacle
                    tmp_detection_width -= tmp_gt_width
                    # Update the largest GT width
                    if tmp_largest_gt_width < tmp_gt_width:
                        tmp_largest_gt_width = tmp_gt_width
                    assigned_gt_obstacles += 1

                else:
                    if np.sum(tmp_gt_coverage) / tmp_detection_area > eval_params['min_overlap']:
                        # This is a partially assigned GT (that means, that the detection is probably (almost) fully
                        # encapsulated inside this GT, therefore the detection should not be counted as FP
                        assigned_gt_obstacles_partially += 1

            # Calculate number of FP detection based on assigned TP to the blob
            if assigned_gt_obstacles > 0:
                # How many largest widths * 1.1 can be fitted inside the remaining width of the detection blob
                tmp_factor = tmp_detection_width / (1.1 * tmp_largest_gt_width)

                if tmp_factor >= 2:
                    fp_list.append({"area": int(tmp_detection_area),
                                    "bbox": np.array([tmp_detection_bbox[1], tmp_detection_bbox[0],
                                                      tmp_detection_bbox[3], tmp_detection_bbox[2]]).tolist(),
                                    "num_triggers": int(np.floor(tmp_factor) - 1)})

                    num_fps += int(np.floor(tmp_factor) - 1)

                    if dangerzone_coverage > 0:
                        fp_list_d.append({"area": int(tmp_detection_area),
                                          "bbox": np.array([tmp_detection_bbox[1], tmp_detection_bbox[0],
                                                            tmp_detection_bbox[3], tmp_detection_bbox[2]]).tolist(),
                                          "num_triggers": int(np.floor(tmp_factor) - 1)})

                        num_fps_d += int(np.floor(tmp_factor) - 1)

            else:
                if assigned_gt_obstacles_partially == 0:
                    num_fps += 1
                    fp_list.append({"area": int(tmp_detection_area),
                                    "bbox": np.array([tmp_detection_bbox[1], tmp_detection_bbox[0],
                                                      tmp_detection_bbox[3], tmp_detection_bbox[2]]).tolist(),
                                    "num_triggers": int(1)})

                    if dangerzone_coverage > 0:
                        fp_list_d.append({"area": int(tmp_detection_area),
                                          "bbox": np.array([tmp_detection_bbox[1], tmp_detection_bbox[0],
                                                            tmp_detection_bbox[3], tmp_detection_bbox[2]]).tolist(),
                                          "num_triggers": int(1)})

                        num_fps_d += 1

    return fp_list, num_fps, fp_list_d, num_fps_d


def remove_above_horizon_2(obstacles_mask, groundtruth_list, horizon_mask):
    # Copy first column of horizon mask to zeroth column. (To address bug where there is empty column)
    horizon_mask[:, 0] = horizon_mask[:, 1]

    # kernel layer type
    dilatation_type = cv2.MORPH_ELLIPSE
    # kernel size
    dilatation_size = 50
    # Create kernel
    element = cv2.getStructuringElement(dilatation_type, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))

    # Dilate horizon mask (this expands surface below the horizon where ones are)
    above_horizon_mask = cv2.dilate(horizon_mask, element)

    # Create mask of ground truth obstacles
    gt_mask = np.zeros(obstacles_mask.shape)
    num_gts = len(groundtruth_list)
    for i in range(num_gts):
        tmp_gt_bb = groundtruth_list[i]['bbox']
        gt_mask[tmp_gt_bb[1]:tmp_gt_bb[3], tmp_gt_bb[0]:tmp_gt_bb[2]] = 1

    # Logical and with horizon mask
    lower_horizon_mask = cv2.erode(horizon_mask, element)
    gt_mask = gt_mask * (1 - lower_horizon_mask)

    # Filter first those obstacles that are significantly above the horizon
    filtered_obstacles_1 = obstacles_mask * above_horizon_mask

    return filtered_obstacles_1, gt_mask, horizon_mask  #lower_horizon_mask


# Remove annotations way above the horizon. This detections should not count towards FPs as they do not affect the
#   navigation of the USV.
def remove_above_horizon(fp_list, horizon_mask):
    dilatation_type = cv2.MORPH_ELLIPSE
    dilatation_size = 150
    element = cv2.getStructuringElement(dilatation_type, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))

    above_horizon_mask = 1 - cv2.dilate(horizon_mask, element)

    filtered_fp_list = []
    num_filtered_fp = 0

    num_fp_obstacles = len(fp_list)

    if num_fp_obstacles == 1:
        tmp_fp_bbox = fp_list[0]['bbox']
        tmp_fp_surf = fp_list[0]['area']
        tmp_fp_covr = np.sum(above_horizon_mask[tmp_fp_bbox[1]:tmp_fp_bbox[3], tmp_fp_bbox[0]:tmp_fp_bbox[2]])
        if tmp_fp_covr < tmp_fp_surf:
            filtered_fp_list = filtered_fp_list.append({"bbox": tmp_fp_bbox,
                                                        "area": tmp_fp_surf
                                                        })
    else:
        for i in range(num_fp_obstacles):
            tmp_fp_bbox = fp_list[i]['bbox']
            tmp_fp_surf = fp_list[i]['area']
            tmp_fp_covr = np.sum(above_horizon_mask[tmp_fp_bbox[1]:tmp_fp_bbox[3], tmp_fp_bbox[0]:tmp_fp_bbox[2]])

            if tmp_fp_covr < tmp_fp_surf:
                if len(filtered_fp_list) == 0:
                    filtered_fp_list.append({"bbox": tmp_fp_bbox,
                                             "area": tmp_fp_surf
                                             })
                else:
                    filtered_fp_list.append({"bbox": tmp_fp_bbox,
                                             "area": tmp_fp_surf
                                             })

                num_filtered_fp += 1

    return filtered_fp_list, num_filtered_fp


# Group near-by FP detections
def group_near_by_detections(regions, label_img):
    pixel_distance = 10

    # Check distances for each label
    for props in regions:

        # Get boundaries coordinates for that label
        label_boundaries = segmentation.find_boundaries(label_img == props.label)
        bound_cord = np.column_stack(np.where(label_boundaries))

        # We will compare each boundaries coordinates with the coordinates of all other label boundaries
        for j_props in regions:
            if j_props.label > props.label:
                regrouped = False
                # Get boundaries coordinates for that label
                j_label_boundaries = segmentation.find_boundaries(label_img == j_props.label)
                j_bound_cord = np.column_stack(np.where(j_label_boundaries))

                # Coordinates comparisons
                i = 0
                while not regrouped and i < len(bound_cord):
                    j = 0
                    while not regrouped and j < len(j_bound_cord):
                        # Apply distance condition
                        if math.hypot(j_bound_cord[j][1] - bound_cord[i][1],
                                      j_bound_cord[j][0] - bound_cord[i][0]) <= pixel_distance:
                            # Assign the less label value
                            label_img[label_img == j_props.label] = min(props.label, j_props.label)
                            j_props.label = min(props.label, j_props.label)
                            regrouped = True
                        j += 1
                    i += 1

    # Second time we use regionprobs to get new labels informations
    regions_2 = measure.regionprops(label_img)

    return regions_2











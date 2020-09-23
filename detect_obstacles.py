import numpy as np
from skimage import measure, transform, segmentation, draw

from utils import compute_surface_area, filter_obstacle_mask, get_obstacle_count
import cv2
import math
import matplotlib.pyplot as plt


# Function performs obstacle detection evaluation and return a list of TP, FP and FN detections with their BBoxes
def detect_obstacles_modb(gt, obstacle_mask, gt_mask, ignore_abv_strad, horizon_mask, eval_params, danger_zone=None):

    # Filter GT obstacle annotations - keep only those that are inside the danger zone mask
    if danger_zone is not None:
        gt = filter_gt_danger_zone(gt, danger_zone, eval_params)

    # Perform the detections
    # - Detect TPs and FNs
    tp_list, fn_list, overlap_percentages_list = check_tp_detections(gt, obstacle_mask, eval_params)

    gt_obstacles_list = np.append(tp_list, fn_list)
    
    # Debug
    # plt.figure(999)
    # plt.clf()
    # plt.subplot(131)
    # plt.imshow(obstacle_mask)
    # plt.subplot(132)
    # plt.imshow(gt_mask)
    # plt.subplot(133)
    # plt.imshow(horizon_mask)
    # plt.show()

    # - Extract FPs ( but only if there are all obstacles in the image annotated, otherwise do not report FPs)
    if gt['all_annotations'] == 'true':
        obstacle_mask, gt_ah, horizon_mask_lower = remove_above_horizon_2(obstacles_mask=obstacle_mask, 
                                                                          groundtruth_list=gt_obstacles_list,
                                                                          horizon_mask=horizon_mask)
        obstacle_mask_f = np.logical_and(np.logical_not(ignore_abv_strad), obstacle_mask)
        fp_list, num_fps = check_fp_detections_2(obstacle_mask_f, gt_obstacles_list, gt_mask, gt_ah, horizon_mask_lower,
                                                 eval_params)
    else:
        fp_list = []
        num_fps = 0
    
    """
    plt.figure(888)
    plt.clf()
    plt.subplot(221)
    plt.imshow(obstacle_mask)
    plt.subplot(222)
    plt.imshow(ignore_abv_strad)
    plt.subplot(223)
    plt.imshow(obstacle_mask_f)
    plt.subplot(224)
    plt.imshow(horizon_mask)
    plt.show()
    """
    
    return tp_list, fp_list, fn_list, num_fps, overlap_percentages_list


# Function checks TP detections (and consequently FN detections as well)
def check_tp_detections(gt, obstacle_mask_filtered, eval_params):
    # Initialize arrays
    # (Arrays will be of size n x 4, where each row is in format: x_TL,y_TL,x_BR,y_BR)
    # TL = top-left, BR = bottom-right
    tp_detections = []
    fn_detections = []
    
    # We store here all overlaps with ground truth that are above 10%
    # This information is later used for setting the optimal TP detection threshold and other analysis
    overlap_percentages = []

    # Get number of filtered obstacles
    num_gt_obstacles = len(gt['obstacles'])

    # Check for overlap between filtered ground truth annotations and filtered detections
    for i in range(num_gt_obstacles):
        # Check if obstacle is sufficiently large
        # gt_area_surface = compute_surface_area(gt['obstacles'][i]['bbox'])
        gt_area_surface = gt['obstacles'][i]['area']

        if gt_area_surface >= eval_params['area_threshold']:
            # Get current GT obstacle bounding-box
            if isinstance(gt['obstacles'][i]['bbox'], list):
                tmp_gt_obs = (np.array(gt['obstacles'][i]['bbox']).astype(np.int))
            else:
                tmp_gt_obs = (gt['obstacles'][i]['bbox']).astype(np.int)

            # Extract bounding-box region from the filtered obstacle mask
            obstacle_mask_area = obstacle_mask_filtered[tmp_gt_obs[1]:tmp_gt_obs[3], tmp_gt_obs[0]:tmp_gt_obs[2]]

            # Check the coverage of the area (pixels labeled with ones belong to obstacles, while pixels labeled with
            #   zero belong to the navigable surface for the USV
            num_correctly_detected_pixels = np.sum(obstacle_mask_area)

            # Check if enough area is covered by an obstacle to be considered a TP detection
            correctly_covered_percentage = num_correctly_detected_pixels / gt_area_surface
            
            # append the overlap percentage with information of the current obstacle...
            if correctly_covered_percentage >= 0.1:
                overlap_percentages.append(round(correctly_covered_percentage, 2))
            
            if correctly_covered_percentage >= eval_params['min_overlap']:
                # Add obstacle to the list of TP detections
                tp_detections.append({"bbox": tmp_gt_obs.tolist(),
                                      "type": gt['obstacles'][i]['type'],
                                      "area": int(gt_area_surface),
                                      "coverage": int(correctly_covered_percentage * 100)})

            else:
                # Add obstacle to the list of FN detections
                fn_detections.append({"bbox": tmp_gt_obs.tolist(),
                                      "type": gt['obstacles'][i]['type'],
                                      "area": int(gt_area_surface),
                                      "coverage": int(correctly_covered_percentage * 100)})

    # Return lists of TP and FN detections
    return tp_detections, fn_detections, overlap_percentages


# Function checks FP detections by searching for blobs that do not overlap with any ground truth annotation
def check_fp_detections_2(obstacle_mask_filtered, gt_obstacle_list, gt_mask_filtered, gt_ah, horizon_mask, eval_params):
    # Initialize false positives mask with all detections
    detections_mask = np.copy(obstacle_mask_filtered)
    detections_mask[gt_mask_filtered > 0] = 0

    detections_mask_labels = measure.label(detections_mask)
    detection_regions_list = measure.regionprops(detections_mask_labels)
    
    # detection_regions_list = group_near_by_detections(detection_regions_list, detection_labels)
    num_detections = len(detection_regions_list)

    #plt.figure(55)
    #plt.clf()
    #plt.subplot(221)
    #plt.imshow(detections_mask_labels)
    #plt.subplot(222)
    #plt.imshow(detections_mask)
    #plt.subplot(223)
    #plt.imshow(horizon_mask)
    #plt.subplot(224)
    #plt.imshow(gt_ah)
    #plt.show()

    # Get number of TP detections
    num_gt_obs = len(gt_obstacle_list)

    # Initialize FP list
    fp_list = []
    num_fps = 0

    # Loop through all detections on screen
    for i in range(num_detections):
        # Get BBOX of current detection in format (Y_TL, X_TL, Y_BR, X_BR)
        tmp_detection_bbox = detection_regions_list[i].bbox

        # Get surface area of current detection
        #   Note: THIS IS NOT A BBOX SURFACE AREA
        tmp_detection_area = detection_regions_list[i].area
        
        # Get surface area of current detection
        #   Note: THIS IS A BBOX SURFACE AREA
        tmp_detection_surf = np.round((tmp_detection_bbox[2] - tmp_detection_bbox[0]) * 
                                      (tmp_detection_bbox[3] - tmp_detection_bbox[1]))

        # Get detection label
        tmp_detection_label = detection_regions_list[i].label

        # Extract binary mask where only pixels belonging to the current detection label are set to one
        tmp_detection_mask = (detections_mask_labels == tmp_detection_label) * 1
        
        horizon_mask = 1 - horizon_mask
        # Check if detection is large enough
        if tmp_detection_area > eval_params['area_threshold'] and \
                ((np.sum(horizon_mask[tmp_detection_bbox[1]:tmp_detection_bbox[3], tmp_detection_bbox[0]:tmp_detection_bbox[3]]) > 0 and np.sum(gt_ah[:, tmp_detection_bbox[0]:tmp_detection_bbox[2]]) == 0) or
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
                
            else:
                if assigned_gt_obstacles_partially == 0:
                    num_fps += 1
                    fp_list.append({"area": int(tmp_detection_area),
                                    "bbox": np.array([tmp_detection_bbox[1], tmp_detection_bbox[0],
                                                      tmp_detection_bbox[3], tmp_detection_bbox[2]]).tolist(),
                                    "num_triggers": int(1)})

    return fp_list, num_fps


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


def remove_above_horizon_2(obstacles_mask, groundtruth_list, horizon_mask):
    horizon_mask[:, 0] = horizon_mask[:, 1]
    dilatation_type = cv2.MORPH_ELLIPSE
    dilatation_size = 50
    element = cv2.getStructuringElement(dilatation_type, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))

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

    return filtered_obstacles_1, gt_mask, lower_horizon_mask


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











import numpy as np
import matplotlib.pyplot as plt
from utils import poly2mask, calculate_root_mean
from skimage import measure
import scipy
import cv2


# Function performs water-edge evaluation
def evaluate_water_edge(gt, obstacle_mask_labels, horizon_mask, eval_params):
    # Get height and width of the image
    h, w = obstacle_mask_labels.shape

    # Get number of danger lines
    num_danger_lines = len(gt['water_edges'])

    # Initialize land mask
    land_mask = np.zeros(obstacle_mask_labels.shape)
    filtered_areas = np.ones(obstacle_mask_labels.shape)
    # Build land mask (ones above the danger lines)
    for i in range(num_danger_lines):
        tmp_danger_line_x = np.array(gt['water_edges'][i]['x_axis']) - 1
        tmp_danger_line_y = np.array(gt['water_edges'][i]['y_axis']) - 1

        if 2 <= tmp_danger_line_x.size == tmp_danger_line_y.size and tmp_danger_line_y.size >= 2:
            # Generate a land mask above the current danger line
            # The mask should be the same size as water-mask
            tmp_mask = poly2mask(np.concatenate(([0], tmp_danger_line_y, [0]), axis=0),
                                 np.concatenate(([tmp_danger_line_x[0]], tmp_danger_line_x, [tmp_danger_line_x[-1]]),
                                                axis=0), (h, w))

            # Add generated land mask of the current danger line to the total land mask...
            land_mask = (np.logical_or(land_mask, tmp_mask)).astype(np.uint8)
            
    land_mask_orig = np.copy(land_mask)
    
    # Mask for obstacles under the water-edge
    obstacles_under_edge_mask = np.zeros(land_mask.shape, dtype=np.uint8)

    # Remove large GT obstacle annotations from the land mask (this is where the boats, etc straddle the water edge. On
    # such sections the water edge should not be evaluated since we cannot pinpoint the actual location)
    for i in range(len(gt['obstacles'])):
        # Get current obstacle annotation
        tmp_obstacle = gt['obstacles'][i]['bbox']
        tmp_obstacle_patch = land_mask[tmp_obstacle[1]:tmp_obstacle[3], tmp_obstacle[0]:tmp_obstacle[2]]
        
        # Build ground truth obstacle mask
        obstacles_under_edge_mask[tmp_obstacle[1]:tmp_obstacle[3], tmp_obstacle[0]:tmp_obstacle[2]] = 1

        # if obstacle protrudes through the water edge
        # (the obstacle should be partially located bellow the water edge and above the water edge)
        if gt['obstacles'][i]['area'] != np.sum(tmp_obstacle_patch) > 0:
            # Remove such part from the evaluation of the water-edge
            land_mask[0:h, tmp_obstacle[0]:tmp_obstacle[2]] = 0
            filtered_areas[0:h, tmp_obstacle[0]:tmp_obstacle[2]] = 0
            
            # !!!!!!!!!!!!!!!!!!!!!!
            # THIS IS TO ADDRESS PARTS WHERE THE WATER-EDGE WAS NOT EVALUATED
            # Add such part to the land_mask_orig, to not generate false-positives mid annotations...
            land_mask_orig[0:tmp_obstacle[3], tmp_obstacle[0]:tmp_obstacle[2]] = 1

    # Build an IgnoreMask for detections above obstacles that straddle the horizon. Such detections should not affect
    # the final score since they do not affect the navigation in any way. Mostly, these detections consists of FP, which
    # are triggered by the un-annotated masts of the ships, etc. Since they are above the water-edge, they do not affect
    # the path planning....
    obstacles_above_horizon = np.logical_and(np.logical_not(cv2.erode(horizon_mask,
                                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))),
                                             obstacles_under_edge_mask)
    #obstacles_above_horizon = np.logical_and(np.logical_not(horizon_mask), obstacles_under_edge_mask)
    obstacles_above_horizon_top = obstacles_above_horizon.argmax(axis=0)
    ignore_above_straddled_obstacles = np.zeros(obstacles_above_horizon.shape)
    for x in range(len(obstacles_above_horizon_top)):
        if obstacles_above_horizon_top[x] > 0:
            ignore_above_straddled_obstacles[0:obstacles_above_horizon_top[x], x] = 1

    # Initializations
    rmse = []  # Total RMSE of the water-edge

    overunder_mask = np.zeros(land_mask.shape, dtype=np.uint8)

    # Get map of obstacles under the water-edge
    obstacles_under_edge_mask = np.logical_and(obstacles_under_edge_mask, np.logical_not(land_mask))

    # Create obstacle map
    obstacle_mask_labels = obstacle_mask_labels * filtered_areas
    lower_bounds_obstacle_mask = np.zeros((h, w))
    
    # Process obstacle mask
    for x in range(w):
        seen_labels = []
        for y in range(h-1):
            current_label = obstacle_mask_labels[y, x]
            next_label = obstacle_mask_labels[y+1, x]
            
            # If we are entering new label
            if next_label != current_label:
                # Check if this is first entry
                if next_label not in seen_labels:
                    seen_labels.append(next_label)
                # When we are exiting the entered label
                else:
                    lower_bounds_obstacle_mask[y+1, x] = 1
                # If we are exiting obstacle    
                if next_label == 0:
                    lower_bounds_obstacle_mask[y, x] = 1
                    
        # Find first non-zero element in the water mask. This is where the water-edge is located.
        idx_gt = np.argmax(1 - land_mask[:, x])

        if idx_gt != 0:
            # Find first non-zero element in the obstacle-mask
            # idx_det = np.argmax(1 - obstacle_mask_labels[:, x])

            # IF we undershoot the water-edge (below the actual water-edge annotation)
            if obstacle_mask_labels[idx_gt, x] > 0:
                gt_hit = False
                lowest_index = idx_gt
                for idx_det in range(idx_gt, h):
                    if obstacles_under_edge_mask[idx_det, x] > 0:  # Make sure we dont hit GT under the annotated w-e
                        gt_hit = True
                        break
                    if obstacle_mask_labels[idx_det, x] == 0 or idx_det == h-1:
                        lowest_index = idx_det
                        # overunder_mask[idx_gt:idx_det, x] = 2
                        break
                if not gt_hit:
                    overunder_mask[idx_gt:lowest_index, x] = 2
            # IF we overshoot the water-edge (above the actual water-edge annotation)
            else:
                for idx_det in range(idx_gt, 0, -1):
                    if obstacle_mask_labels[idx_det, x] == 1 or idx_det == 1:
                        overunder_mask[idx_det:idx_gt, x] = 1
                        break
            
    #plt.figure(2)
    #plt.clf()
    #plt.subplot(121)
    #plt.imshow(land_mask)
    #plt.subplot(122)
    #plt.imshow(obstacles_under_edge_mask)
    #plt.show()

    dtf_detected_water_edge = scipy.ndimage.morphology.distance_transform_edt(np.logical_not(lower_bounds_obstacle_mask))

    num_land_detections = [0] * 2
    # Loop through the width and check for each point of the ground-truth water edge where is the closest detected
    #  water edge point
    for we in range(num_danger_lines):
        # Loop through danger lines
        if (isinstance(gt['water_edges'][we]['x_axis'], list) and len(gt['water_edges'][we]['x_axis']) > 1) or \
           (isinstance(gt['water_edges'][we]['x_axis'], np.ndarray) and gt['water_edges'][we]['x_axis'].size > 1):
            cur_danger_line_x = np.array(gt['water_edges'][we]['x_axis'])
            # Get the beginning of the danger line (x-axis)
            cur_danger_line_x_min = np.max([np.min(np.round(cur_danger_line_x)), 0]).astype(np.int)
            # Get the end of the danger line (x-axis)
            cur_danger_line_x_max = np.min([np.max(np.round(cur_danger_line_x)), w]).astype(np.int)
            
            # Initialize the counter of correctly/incorrectly detected pixels of the danger line
            tmp_land_detections = [0] * 2
            # Loop through all the pixels between the start and end of the danger line
            for pix in range(cur_danger_line_x_min, cur_danger_line_x_max):
                
                # Get y value of the ground truth annotation at the current position 'pix'
                idx_gt = np.argmax(1 - land_mask[:, pix])
        
                # If y value exists
                #   Note: such y value might not exist in land mask since we slice out the parts where boats and other
                #         larger obstacles protrude through the water-edge annotation
                if idx_gt != 0:
                    # x-coordinate of the ground truth water edge
                    tmp_x_coord = pix
                    # y-coordinate of the ground truth water edge
                    tmp_y_coord = idx_gt
        
                    # Check in the DT mask how far the nearest lower-bound of obstacle annotation is located
                    tmp_error = dtf_detected_water_edge[tmp_y_coord, tmp_x_coord]
                    # If such location is further is than 40 pixels, then we treat this as incorrect detection
                    if tmp_error > 20: #40: #20: (changed 23.1.2020) #40:  40 - for mu_r and 20 for mu_a
                        # The root-mean-squared-error (aka the distance to the detection) is set to the highest possible
                        # (half of the image height) in order to punish missed detection, while the pixel located at
                        # current x position is considered as FN
                        rmse.append(idx_gt)  # or h/2
                        # tmp_land_detections[1] += 1
                        num_land_detections[1] += 1
                    # If such location is less than 40 pixels away, then we treat this as correct detection
                    else:
                        # The RMSE is set to the actual distance, read from the DTF mask
                        rmse.append(tmp_error)
                        # The pixel located at the current x position is considered as TP
                        # tmp_land_detections[0] += 1
                        num_land_detections[0] += 1

            """
            # We compute whether the current water-edge was correctly detected or not based on the ratio of the
            # correctly/incorrectly detected pixels from its edge and prescribed threshold 
            if tmp_land_detections[0] + tmp_land_detections[1] > 0 and \
                tmp_land_detections[0] / (tmp_land_detections[0] + tmp_land_detections[1]) > eval_params['min_overlap']:
                # If such ratio excedes the threshold, then this water-edge was correctly detected
                num_land_detections[0] += 1
            else:
                # Otherwise this water-edge was incorrectly detected
                num_land_detections[1] += 1
            """
            
    # Compute the average of RMSE errors for the current frame
    if len(rmse) == 0:
        rmse = 0
    else:
        rmse = np.mean(rmse)
    
    # plt.figure(2)
    # plt.clf()
    # plt.subplot(131)
    # plt.imshow(obstacle_mask_labels)
    # plt.subplot(132)
    # plt.imshow(lower_bounds_obstacle_mask)
    # plt.subplot(133)
    # plt.imshow(dtf_detected_water_edge)
    # plt.show()

    return rmse, num_land_detections, overunder_mask, land_mask_orig, ignore_above_straddled_obstacles


# Function performs water edge evaluation
def evaluate_water_edge_old(gt, water_mask, eval_params):
    # Get the number of danger lines
    num_danger_lines = len(gt['water_edges'])

    # Initialize land mask
    land_mask = np.zeros(water_mask.shape)

    # Build land mask (ones above the danger lines)
    for i in range(num_danger_lines):
        tmp_danger_line_x = np.array(gt['water_edges'][i]['x_axis'])
        tmp_danger_line_y = np.array(gt['water_edges'][i]['y_axis'])

        if 2 <= tmp_danger_line_x.size == tmp_danger_line_y.size and tmp_danger_line_y.size >= 2:
            # Generate a land mask above the current danger line
            # The mask should be the same size as water-mask
            tmp_mask = poly2mask(np.concatenate(([0], tmp_danger_line_y, [0]), axis=0),
                                 np.concatenate(([tmp_danger_line_x[0]], tmp_danger_line_x, [tmp_danger_line_x[-1]]), axis=0),
                                 (water_mask.shape[0], water_mask.shape[1]))
    
            # Add generated land mask of the current danger line to the total land mask...
            land_mask = (np.logical_or(land_mask, tmp_mask)).astype(np.uint8)

    # Remove large GT obstacle annotations from the land mask
    for i in range(len(gt['obstacles'])):
        # Get current obstacle annotation
        tmp_obstacle = gt['obstacles'][i]['bbox']

        tmp_obstacle_patch = land_mask[tmp_obstacle[1]:tmp_obstacle[3], tmp_obstacle[0]:tmp_obstacle[2]]

        # If obstacle protrudes through the water edge
        if np.sum(tmp_obstacle_patch) > 0:
            # Remove such part from the evaluation of the water-edge
            land_mask[0:tmp_obstacle[3], tmp_obstacle[0]:tmp_obstacle[2]] = 0

    # Initializations
    rmse_w = np.array([])  # Total RMSE of the water-edge
    rmse_o = np.array([])  # RMSE of the area where the algorithm has overshoot the edge
    rmse_u = np.array([])  # RMSE of the area where the algorithm has undershoot the edge

    overunder_mask = np.zeros(land_mask.shape)

    # Loop through the width of the image and perform water-edge evaluation where needed
    for i in range(water_mask.shape[1]):
        # Find first zero element in land mask at the current column.
        # This is where the water-edge is located. If such index is 0, then ignore, since the water edge is not
        #   annotated in that column
        idx_gt = np.argmax(1 - land_mask[:, i])

        if idx_gt != 0:
            # Find first non-zero element in the water mask. This is where the water-edge is located.
            idx_det = np.argmax(water_mask[:, i])

            # Compute squared error between the idx_det and idx_gt
            if len(rmse_w) == 0:
                rmse_w = np.array([(idx_gt - idx_det)**2])
            else:
                rmse_w = np.row_stack((rmse_w, (idx_gt - idx_det)**2))

            # Check whether we have undershoot or overshoot the water-edge and update overunder_mask accordingly
            if idx_gt > idx_det:  # we have overshoot
                overunder_mask[idx_det:idx_gt, i] = 1
                if len(rmse_o) == 0:
                    rmse_o = np.array([(idx_gt - idx_det)**2])
                else:
                    rmse_o = np.row_stack((rmse_o, (idx_gt - idx_det)**2))

            else:  # we have undershoot
                overunder_mask[idx_gt:idx_det, i] = 2
                if len(rmse_u) == 0:
                    rmse_u = np.array([(idx_gt - idx_det)**2])
                else:
                    rmse_u = np.row_stack((rmse_u, (idx_gt - idx_det)**2))
    """
    plt.figure()
    plt.subplot(131)
    plt.imshow(land_mask)
    plt.subplot(132)
    plt.imshow(water_mask)
    plt.subplot(133)
    plt.imshow(overunder_mask)
    plt.show()
    """

    # Calculate root mean of square errors for the total water-edge
    return calculate_root_mean(rmse_w), calculate_root_mean(rmse_o), calculate_root_mean(rmse_u), overunder_mask,\
           land_mask







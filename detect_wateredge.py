import numpy as np
import matplotlib.pyplot as plt
from utils import poly2mask, calculate_root_mean


# Function performs water edge evaluation
def evaluate_water_edge(gt, water_mask, eval_params):
    # Get the number of danger lines
    num_danger_lines = len(gt['water-edges'])

    # Initialize land mask
    land_mask = np.zeros(water_mask.shape)

    # Build land mask (ones above the danger lines)
    for i in range(num_danger_lines):
        tmp_danger_line_x = gt['water-edges'][i]['x-axis']
        tmp_danger_line_y = gt['water-edges'][i]['y-axis']

        # Generate a land mask above the current danger line
        # The mask should be the same size as water-mask
        tmp_mask = poly2mask(np.concatenate([[0], tmp_danger_line_y, [0]]),
                             np.concatenate([[tmp_danger_line_x[0]], tmp_danger_line_x, [tmp_danger_line_x[-1]]]),
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







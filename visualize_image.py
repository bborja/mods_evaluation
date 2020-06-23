import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def visualize_single_image(img, segm_mask, results_detection, gt, original_segm_mask=None):
    # Overlay segmentation mask over the actual image
    added_image = cv2.addWeighted(img, 0.7, segm_mask, 0.3, 0.2)
    # For video frame generating
    if original_segm_mask is not None:
        img_scalled = cv2.resize(img, (426, 320))
        msk_scalled = cv2.resize(original_segm_mask, (426, 320), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((11, 11), np.float32) / 121
        added_image[0:320, 0:426, :] = img_scalled
        added_image[0:320, 426:426+426, :] = msk_scalled
        added_image[0:320, -426:-1, :] = (0.8 *
                                          cv2.filter2D(added_image[0:320, -426:-1, :], -1, kernel)).astype(np.uint8)

    plt.figure(1)
    # Plot image
    plt.imshow(added_image)
    plt.axis('equal')
    # Plot water-edge danger lines
    num_danger_lines = len(gt['water-edges'])
    for i in range(num_danger_lines):
        tmp_danger_line_x = gt['water-edges'][i]['x-axis']
        tmp_danger_line_y = gt['water-edges'][i]['y-axis']
        plt.plot(tmp_danger_line_x, tmp_danger_line_y, marker='', color='black', linewidth=2, linestyle='solid')
        plt.plot(tmp_danger_line_x, tmp_danger_line_y, marker='', color='pink', linewidth=1, linestyle='dashed')
    # Plot detection rectangles
    ax = plt.gca()
    # Plot TPs
    ax = plot_detection_rectangles(results_detection, 'tp_list', ax)
    ax = plot_detection_rectangles(results_detection, 'fp_list', ax)
    ax = plot_detection_rectangles(results_detection, 'fn_list', ax)

    if original_segm_mask is None:
        plt.show()


# Plot detection rectangles
def plot_detection_rectangles(results_detection, list_name, ax):
    if list_name == 'tp_list':
        edge_color = 'green'
    elif list_name == 'fn_list':
        edge_color = 'red'
    else:
        edge_color = 'yellow'

    num_dets = len(results_detection['obstacles'][list_name])
    for i in range(num_dets):
        tmp_bbox = results_detection['obstacles'][list_name][i]['bbox']
        rect_bg = patches.Rectangle((tmp_bbox[0], tmp_bbox[1]), tmp_bbox[2]-tmp_bbox[0], tmp_bbox[3]-tmp_bbox[1],
                                    linewidth=2, edgecolor='black', facecolor='none')
        rect_fg = patches.Rectangle((tmp_bbox[0], tmp_bbox[1]), tmp_bbox[2]-tmp_bbox[0], tmp_bbox[3]-tmp_bbox[1],
                                    linewidth=1, edgecolor=edge_color, facecolor='none')
        ax.add_patch(rect_bg)
        ax.add_patch(rect_fg)

        if edge_color == 'yellow':
            print(tmp_bbox)

    return ax


# Visualize image for video
def visualize_image_for_video(img, segm_mask, segm_mask_overlay, results_detection, gt):
    # Overlay segmentation mask over the actual image
    visualize_single_image(img, segm_mask_overlay, results_detection, gt, segm_mask)

    rmse_t = results_detection['rmse_t']
    rmse_o = results_detection['rmse_o']
    rmse_u = results_detection['rmse_u']
    num_tps = len(results_detection['obstacles']['tp_list'])
    num_fps = len(results_detection['obstacles']['fp_list'])
    num_fns = len(results_detection['obstacles']['fn_list'])
    f1_score = ((2 * num_tps) / (2 * num_tps + num_fps + num_fns)) * 100

    # Overlay text statistics...
    plt.text(120, 50, "Input image", fontsize=8)
    plt.text(490, 50, "Segmentation mask", fontsize=8)
    plt.text(880, 50, "RMSE: %d px" % rmse_t, fontsize=8)
    plt.text(880, 90, "RMSE over: %.01f" % rmse_o, fontsize=8)
    plt.text(880, 130, "RMSE under: %.01f" % rmse_u, fontsize=8)
    plt.text(880, 170, "TP: %d" % num_tps, fontsize=8)
    plt.text(880, 210, "FP: %d" % num_fps, fontsize=8)
    plt.text(880, 250, "FN: %d" % num_fns, fontsize=8)
    plt.text(880, 290, "F1: %.01f" % f1_score, fontsize=8)
    plt.show()

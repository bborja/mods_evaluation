import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def visualize_single_image(img, segm_mask, results_detection, gt, original_segm_mask=None):
    # Overlay segmentation mask over the actual image
    added_image = cv2.addWeighted(img, 0.7, segm_mask, 0.3, 0.2)
    # For video frame generating
    if original_segm_mask is not None:
        # Read QR code
        img_qr = cv2.resize(((cv2.imread('images/qr-code_2.png')) > 180).astype(np.uint8), (200, 200))
        # Resize image and mask to fit on top of the screen
        img_scalled = cv2.resize(img, (426, 320))
        msk_scalled = cv2.resize(original_segm_mask, (426, 320), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((11, 11), np.float32) / 121
        # image
        added_image[0:320, 0:426, :] = img_scalled
        # segmentation mask
        added_image[0:320, 426:426+426, :] = msk_scalled
        # information
        added_image[0:320, -426:-1, :] = (0.8 *
                                          cv2.filter2D(added_image[0:320, -426:-1, :], -1, kernel)).astype(np.uint8)
        added_image[110:310, -210:-10, :] *= img_qr
        # black separating line
        added_image[320:322, :] = 0

    #mpl.rcParams['text.color'] = 'white'
    dpi = mpl.rcParams['figure.dpi']
    height, width, depth = img.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Plot image
    ax.imshow(added_image)

    # Plot danger zone
    ax.plot(gt['danger_zone']['x-axis'], gt['danger_zone']['y-axis'], marker='', color='orange', linewidth=1,
            linestyle='dashed')

    # Plot water-edge danger lines
    num_danger_lines = len(gt['water-edges'])
    for i in range(num_danger_lines):
        tmp_danger_line_x = gt['water-edges'][i]['x-axis']
        tmp_danger_line_y = gt['water-edges'][i]['y-axis']
        ax.plot(tmp_danger_line_x, tmp_danger_line_y, marker='', color='black', linewidth=3, linestyle='solid')
        ax.plot(tmp_danger_line_x, tmp_danger_line_y, marker='', color='pink', linewidth=1, linestyle='dashed')
        ax.text(tmp_danger_line_x[0], tmp_danger_line_y[0] - 2, 'water-edge-%d' % i, fontsize=6)

    # Plot detection rectangles
    #ax = plt.gca()
    ax = plot_detection_rectangles(results_detection, 'tp_list', ax)  # Plot TPs
    ax = plot_detection_rectangles(results_detection, 'fp_list', ax)  # Plot FPs
    ax = plot_detection_rectangles(results_detection, 'fn_list', ax)  # Plot FNs

    ax = plot_detection_rectangles(results_detection, 'tp_list', ax, True)  # Plot TPs in danger zone
    ax = plot_detection_rectangles(results_detection, 'fp_list', ax, True)  # Plot FPs in danger zone
    ax = plot_detection_rectangles(results_detection, 'fn_list', ax, True)  # Plot FNs in danger zone

    # if original_segm_mask is None:
    #     plt.show()

    return fig, ax


# Plot detection rectangles
def plot_detection_rectangles(results_detection, list_name, ax, in_danger_zone=False):
    if list_name == 'tp_list':
        edge_color = 'green'
    elif list_name == 'fn_list':
        edge_color = 'red'
    else:
        edge_color = 'yellow'

    if in_danger_zone:
        detection_type = 'obstacles_danger'
    else:
        detection_type = 'obstacles'

    num_dets = len(results_detection[detection_type][list_name])
    for i in range(num_dets):
        tmp_bbox = results_detection[detection_type][list_name][i]['bbox']
        if in_danger_zone:
            rect_fg = patches.Rectangle((tmp_bbox[0], tmp_bbox[1]), tmp_bbox[2]-tmp_bbox[0], tmp_bbox[3]-tmp_bbox[1],
                                        linewidth=2, edgecolor=edge_color, facecolor='none', linestyle=':')
            ax.add_patch(rect_fg)
        else:
            if edge_color is not 'yellow':
                rect_fg = patches.Rectangle((tmp_bbox[0], tmp_bbox[1]), tmp_bbox[2] - tmp_bbox[0],
                                            tmp_bbox[3] - tmp_bbox[1],
                                            linewidth=1, edgecolor='black', facecolor=edge_color, alpha=0.35)

                ax.text(tmp_bbox[0], tmp_bbox[1], results_detection[detection_type][list_name][i]['type'] +
                        '-%d%%' % results_detection[detection_type][list_name][i]['coverage'], fontsize=6)
            else:
                rect_fg = patches.Rectangle((tmp_bbox[0], tmp_bbox[1]), tmp_bbox[2] - tmp_bbox[0],
                                            tmp_bbox[3] - tmp_bbox[1],
                                            linewidth=1, edgecolor='black', facecolor=edge_color, alpha=0.25)

                ax.text(tmp_bbox[0], tmp_bbox[1],
                        'FP (%d)' % results_detection[detection_type][list_name][i]['num_triggers'], fontsize=6)

            ax.add_patch(rect_fg)

    return ax


# Visualize image for video
def visualize_image_for_video(img, segm_mask, segm_mask_overlay, results_detection, gt):
    # Overlay segmentation mask over the actual image
    fig, ax = visualize_single_image(img, segm_mask_overlay, results_detection, gt, segm_mask)

    rmse_t = results_detection['rmse_t']
    rmse_o = results_detection['rmse_o']
    rmse_u = results_detection['rmse_u']
    num_tps = len(results_detection['obstacles']['tp_list'])
    num_tps_d = len(results_detection['obstacles_danger']['tp_list'])
    num_fps = count_number_fps(results_detection['obstacles']['fp_list'])
    num_fps_d = count_number_fps(results_detection['obstacles_danger']['fp_list'])
    num_fns = len(results_detection['obstacles']['fn_list'])
    num_fns_d = len(results_detection['obstacles_danger']['fn_list'])
    f1_score = ((2 * num_tps) / (2 * num_tps + num_fps + num_fns)) * 100

    #ax = fig.add_axes([0, 0, 1, 1])

    if rmse_o + rmse_u > 0:
        rmse_o_percent = rmse_o / (rmse_o + rmse_u) * 100
        rmse_u_percent = rmse_u / (rmse_o + rmse_u) * 100
    else:
        rmse_o_percent = 0
        rmse_u_percent = 0

    # Overlay text statistics...
    ax.text(150, 50, "Input image", fontsize=12)
    ax.text(560, 50, "Segmentation mask", fontsize=12)
    ax.text(880, 50, "RMSE: %d px (above: %.01f%%, under: %.01f%%)" % (rmse_t, rmse_o_percent, rmse_u_percent),
            fontsize=12)
    ax.text(880, 90,  "TPs: %d" % num_tps, fontsize=12)
    ax.text(890, 110, "in danger zone: %d" % num_tps_d, fontsize=12)
    ax.text(880, 150, "FPs: %d" % num_fps, fontsize=12)
    ax.text(890, 170, "in danger zone: %d" % num_fps_d, fontsize=12)
    ax.text(880, 210, "FNs: %d" % num_fns, fontsize=12)
    ax.text(890, 230, "in danger zone: %d" % num_fns_d, fontsize=12)
    ax.text(880, 290, "F1: %.01f%%" % f1_score, fontsize=15)

    ax.text(1100, 110, "MODB Dataset", fontsize=12)

    # fig.savefig('./results/bla.png')
    # plt.show()

    return fig


def count_number_fps(fp_list):
    num_fps = 0

    num_entries = len(fp_list)
    for i in range(num_entries):
        num_fps += fp_list[i]['num_triggers']

    return num_fps

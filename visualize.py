import os
import cv2
import json
import argparse
import numpy as np
import matplotlib.animation as animation
from utils import read_gt_file, code_mask_to_labels, code_labels_to_colors, resize_image
from visualize_image import visualize_single_image, visualize_image_for_video

SEQUENCES = None
FRAME = None
EXPORT_VIDEO = True
OUTPUT_PATH = './results/video_output/'
RESULTS_PATH = './results'
DATA_PATH = "F:/Projects/matlab/RoBoat/dataset_public"
SEGMENTATION_PATH = "F:/Projects/matlab/RoBoat/dataset_public"
SEGMENTATION_COLORS = np.array([[  0,   0,   0],
                                [255,   0,   0],
                                [  0, 255,   0]])


def get_arguments():
    """ Parse all the arguments provided from the CLI
    Returns: A list of parsed arguments
    """
    parser = argparse.ArgumentParser(description='Marine Obstacle Detection Benchmark.')
    parser.add_argument("--data-path", type=str, default=DATA_PATH,
                        help="Absolute path to the folder where MODB sequences are stored.")
    parser.add_argument("--segmentation-path", type=str, default=SEGMENTATION_PATH,
                        help="Absolute path to the output folder where segmentation masks are stored.")
    parser.add_argument("--output-path", type=str, default=OUTPUT_PATH,
                        help="Output path where the results and statistics of the evaluation will be stored.")
    parser.add_argument("--results-path", type=str, default=RESULTS_PATH,
                        help="Absolute path to the folder where evaluation results are stored.")
    parser.add_argument("--method-name", type=str, required=True,
                        help="<Required> Method name. This should be equal to the folder name in which the "
                             "segmentation masks are located.")
    parser.add_argument("--sequences", type=int, nargs='+', required=False, default=SEQUENCES,
                        help="List of sequences on which the evaluation procedure is performed. Zero = all.")
    parser.add_argument("--frame", type=int, required=False, default=FRAME,
                        help="Frame number, for a single image visualization.")
    parser.add_argument("--segmentation-colors", type=int, default=SEGMENTATION_COLORS,
                        help="Segmentation mask colors corresponding to the three semantic labels. Given as 3x3 array,"
                             "where the first row corresponds to the obstacles color, the second row corresponds to the"
                             "water component color and the third row correspond to the sky component color, all"
                             "written in the RGB format.")
    parser.add_argument("--export-video", type=bool, default=EXPORT_VIDEO,
                        help="Switch for exporting a video of sequence/s.")

    return parser.parse_args()


def main():
    args = get_arguments()

    # Norm paths...
    args.data_path = os.path.normpath(args.data_path)
    args.segmentation_path = os.path.normpath(args.segmentation_path)
    args.output_path = os.path.normpath(args.output_path)

    # Read results JSON file
    with open(os.path.join(args.results_path, 'results_%s.json' % args.method_name)) as f:
        results = json.load(f)

    print(results)
    print(len(results['sequences']))
    print(results['sequences'][0]['evaluated'])
    print(results['sequences'][1]['evaluated'])

    if args.export_video:
        if not os.path.exists(os.path.join(args.output_path, args.method_name)):
            if not os.path.exists(os.path.join(args.output_path)):
                os.mkdir(os.path.join(args.output_path))
            os.mkdir(os.path.join(args.output_path, args.method_name))

    if args.frame is not None and len(args.frame) == 1:
        if args.sequences is not None and len(args.sequences) == 1:
            # Load image
            img = cv2.imread(os.path.join(args.data_path,
                                          results['sequences'][args.sequences-1]['frames'][args.frame-1]['img_name']))
            # Load ground truth
            gt = read_gt_file(os.path.join(args.data_path,
                                           results['sequences'][args.sequences-1]['frames'][args.frame]['ant_name']))
            # Load segmentation output
            seg = cv2.imread(os.path.join(args.segmentation_path, 'seq%02d' % args.sequences, args.method_name,
                                          'mask_%03d.png' % args.frame))

            # Over/Under mask
            ou_mask = results['sequences'][args.sequences-1]['frames'][args.frame]['over_under_mask']

            # Code mask to labels
            seg = code_mask_to_labels(seg, args.segmentation_colors)
            # Update segmentation mask with the over/under mask
            seg[ou_mask == 1] = 3
            seg[ou_mask == 2] = 4
            # Code labels to colors
            seg = code_labels_to_colors(seg)

            # Visualize image
            visualize_single_image(img, seg, results['sequences'][args.sequences-1]['frames'][args.frame],
                                   gt[args.frame])

        else:
            print('<Error>: Sequence not specified or more than one sequence given!')
            return 0

    else:
        if args.sequences is None:
            args.sequences = np.arange(len(results['sequences']))

        # Set up formatting for the movie files
        #Writer = animation.writers['ffmpeg']
        #writer = Writer(fps=30, metadata=dict(artist='Borja Bovcon, Jon Muhovic, Janez Pers, Matej Kristan'), bitrate=1800)

        for seq_id in args.sequences:
            if results['sequences'][seq_id - 1]['evaluated']:
                num_frames_in_sequence = len(results['sequences'][seq_id - 1]['frames'])
                print(num_frames_in_sequence)

                # Load ground truth of sequence
                gt = read_gt_file(os.path.join(args.data_path, 'seq%02d' % seq_id, 'annotations.json'))

                for fr_id in range(num_frames_in_sequence):
                    # Load image
                    img_name = results['sequences'][seq_id - 1]['frames'][fr_id]['img_name']
                    img_name_split = img_name.split(".")
                    img = cv2.imread(os.path.join(args.data_path, 'seq%02d' % seq_id, 'frames', img_name))
                    # Convert BGR TO RGB for visualization
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Load segmentation output
                    seg = cv2.imread(os.path.join(args.segmentation_path, 'seq%02d' % seq_id,
                                                  args.method_name, '%s.png' % img_name_split[0]))
                    seg = resize_image(seg, (img.shape[1], img.shape[0]))

                    #ou_mask = results['sequences'][seq_id]['frames'][fr_id]['over_under_mask']

                    # Code mask to labels
                    seg = code_mask_to_labels(seg, args.segmentation_colors)
                    # Update segmentation mask with the over/under mask.
                    seg_ou = np.copy(seg)
                    #seg_ou[ou_mask == 1] = 3
                    #seg_ou[ou_mask == 2] = 4

                    # Code labels to colors
                    seg = code_labels_to_colors(seg)
                    seg_ou = code_labels_to_colors(seg_ou)

                    # Visualize image
                    # seg = output segmentation mask
                    # seg_overlay = output segmentation mask with an additional visual information of over/under w.edge
                    fig1 = visualize_image_for_video(img, seg, seg_ou, results['sequences'][seq_id-1]['frames'][fr_id],
                                                     gt['sequence'][fr_id])

                    # Save plot to video
                    #line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                    #                                   interval=50, blit=True)
                    #line_ani.save('lines.mp4', writer=writer)

            else:
                print('<Error>: Sequence %d was not evaluated' % seq_id)


if __name__ == "__main__":
    main()


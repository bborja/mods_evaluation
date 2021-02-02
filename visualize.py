import os
import cv2
import sys
import json
import shutil
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from utils import read_gt_file, code_mask_to_labels, code_labels_to_colors, resize_image
from visualize_image import visualize_single_image, visualize_image_for_video

SEQUENCES = None
FRAME = None
EXPORT_VIDEO = True
OUTPUT_PATH = 'E:/MODB_results/video_output/'
RESULTS_PATH = './results'
DATA_PATH = "E:/MODB/raw"
SEGMENTATION_PATH = "E:/MODB_output"
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

    if args.export_video:
        if not os.path.exists(os.path.join(args.output_path, args.method_name)):
            if not os.path.exists(os.path.join(args.output_path)):
                os.mkdir(os.path.join(args.output_path))
            os.mkdir(os.path.join(args.output_path, args.method_name))

    gt = read_gt_file(os.path.join(args.data_path, 'modb.json'))

    if args.frame is not None and isinstance(args.frame, int):
        if args.sequences is not None and len(args.sequences) == 1:
            seq_id = args.sequences[0]
            if not os.path.exists(os.path.join(args.output_path, args.method_name, 'seq%02d' % seq_id)):
                os.mkdir(os.path.join(args.output_path, args.method_name, 'seq%02d' % seq_id))
                os.mkdir(os.path.join(args.output_path, args.method_name, 'seq%02d' % seq_id, 'tmp_frames'))

            # Load image
            seq_path = gt['dataset']['sequences'][seq_id - 1]['path']
            img = cv2.imread(os.path.join(args.data_path, seq_path,
                                          results['sequences'][seq_id-1]['frames'][args.frame-1]['img_name']))
            
            # Load segmentation output
            seg = cv2.imread(os.path.join(args.segmentation_path, 'seq%02d' % seq_id, args.method_name, '%04d.png' % (args.frame * 10)))
            print(os.path.join(args.segmentation_path, 'seq%02d' % seq_id, args.method_name, '%04d.png' % args.frame))
                                          #'mask_%03d.png' % args.frame))

            # Over/Under mask
            #ou_mask = results['sequences'][seq_id-1]['frames'][args.frame]['over_under_mask']

            # Code mask to labels
            print(seg.shape)
            seg = code_mask_to_labels(seg, args.segmentation_colors)
            # Update segmentation mask with the over/under mask
            #seg[ou_mask == 1] = 3
            #seg[ou_mask == 2] = 4
            # Code labels to colors
            seg = code_labels_to_colors(seg)

            # Visualize image
            visualize_single_image(img, seg, results['sequences'][seq_id-1]['frames'][args.frame],
                                   gt['dataset']['sequences'][seq_id-1]['frames'][args.frame])

        else:
            print('<Error>: Sequence not specified or more than one sequence given!')
            return 0

    else:
        if args.sequences is None:
            args.sequences = np.arange(1, len(results['sequences']))

        for seq_id in args.sequences:
            # Create folders if they dont exist yet
            if not os.path.exists(os.path.join(args.output_path, args.method_name, 'seq%02d' % seq_id)):
                os.mkdir(os.path.join(args.output_path, args.method_name, 'seq%02d' % seq_id))
                os.mkdir(os.path.join(args.output_path, args.method_name, 'seq%02d' % seq_id, 'tmp_frames'))

            if results['sequences'][seq_id - 1]['evaluated']:
                num_frames_in_sequence = len(results['sequences'][seq_id - 1]['frames'])

                frame_counter = 0
                for fr_id in range(num_frames_in_sequence):
                    sys.stdout.write("\rProcessing sequence %02d, image %03d / %03d" %
                                     (seq_id, (fr_id + 1), num_frames_in_sequence))
                    # feed, so it erases the previous line.
                    sys.stdout.flush()

                    seq_path = gt['dataset']['sequences'][seq_id - 1]['path']

                    # Load image
                    img_name = results['sequences'][seq_id - 1]['frames'][fr_id]['img_name']
                    img_name_split = img_name.split(".")
                    img = cv2.imread(os.path.join(args.data_path + seq_path, img_name))
                    # Convert BGR TO RGB for visualization
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Load segmentation output
                    seg = cv2.imread(os.path.join(args.segmentation_path, 'seq%02d' % seq_id,
                                                  args.method_name, '%04d.png' % (frame_counter*10)))

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
                                                     gt['dataset']['sequences'][seq_id-1]['frames'][fr_id])

                    if True:  #args.export_video:
                        fig1.savefig(os.path.join(args.output_path, args.method_name, 'seq%02d' % seq_id, 'tmp_frames', '%08d.png' % fr_id))
                    else:
                        plt.show()
                        
                    frame_counter += 1

            else:
                print('<Error>: Sequence %d was not evaluated' % seq_id)

        print('\n')
        
        if args.export_video:
            # Export frames to video
            cmd = ['ffmpeg', '-r', '2', '-i', os.path.join(args.output_path, args.method_name, 'seq%02d' % seq_id, 'tmp_frames', '%08d.png'),
                   '-r', '30', os.path.join(args.output_path, args.method_name, 'sequence_%02d.mp4' % seq_id)]

            ret_code = subprocess.call(cmd)
            if not ret_code == 0:
                raise ValueError('Error {} executing command: {}'.format(ret_code, ' '.join(cmd)))

            # Delete temporary image files used for generating video
            """ Uncomment when releasing the code...
            tmp_folder = os.path.join(args.output_path, args.method_name, 'seq%02d' % seq_id, 'tmp_frames')
            for filename in os.listdir(tmp_folder):
                file_path = os.path.join(tmp_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            """


if __name__ == "__main__":
    main()


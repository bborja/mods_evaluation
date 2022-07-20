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
from visualization import visualize_single_image, visualize_image_for_video
from configs import get_cfg


def get_arguments():
    """ Parse all the arguments provided from the CLI
    Returns: A list of parsed arguments
    """
    parser = argparse.ArgumentParser(description='Marine Obstacle Detection Benchmark.')
    parser.add_argument("method", type=str,
                        help="<Required> Method name. This should be equal to the folder name in which the "
                             "segmentation masks are located.")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output path where the results and statistics of the evaluation will be stored.")
    parser.add_argument("--sequences", type=int, nargs='+',
                        help="Sequences on which to perform visualization")
    parser.add_argument("--frame", type=int,
                        help="Frame number, for a single image visualization.")
    parser.add_argument("--export-video", type=bool, default=True,
                        help="Switch for exporting a video of sequence/s.")
    parser.add_argument("--delete-raw-imgs", type=bool, default=False,
                        help="Delete raw generated images after exporting video.")
    parser.add_argument("--config-file", type=str, default=None,
                        help="Config file to use. If not specified, the default config is used.")

    return parser.parse_args()


def main():
    args = get_arguments()
    cfg = get_cfg(args)

    # Read results JSON file
    with open(os.path.join(cfg.PATHS.RESULTS, 'results_%s.json' % args.method)) as f:
        results = json.load(f)

    # Create output folder for video export if it does not exist yet
    if args.export_video:
        os.makedirs(os.path.join(args.output_path, args.method), exist_ok=True)

    # Read GT json
    gt = read_gt_file(os.path.join(cfg.PATHS.DATASET, 'mods.json'))

    if args.frame is not None and isinstance(args.frame, int):
        if args.sequences is not None and len(args.sequences) == 1:
            seq_id = args.sequences[0]
            os.makedirs(os.path.join(args.output_path, args.method, 'seq%02d' % seq_id, 'tmp_frames'), exist_ok=True)

            # Get sequence path
            seq_path = gt['dataset']['sequences'][seq_id - 1]['path'][1:]
            # Load image
            img = cv2.imread(os.path.join(cfg.PATHS.DATASET, "sequences", seq_path,
                                          results['sequences'][seq_id-1]['frames'][args.frame-1]['img_name']))

            # Load segmentation output
            if cfg.SEGMENTATIONS.SEQ_FIRST:
                seg = cv2.imread(os.path.join(cfg.PATHS.SEGMENTATIONS, seq_path_split[0],
                                              args.method, '{}.png'.format(img_name_split[0])))
            else:
                seg = cv2.imread(os.path.join(cfg.PATHS.SEGMENTATIONS, args.method, seq_path_split[0],
                                              '{}.png'.format(img_name_split[0])))

            # Code mask to labels
            seg = code_mask_to_labels(seg, cfg.SEGMENTATIONS.INPUT_COLORS)

            # Code labels to colors
            seg = code_labels_to_colors(seg, cfg)

            # Visualize image
            visualize_single_image(img, seg, results['sequences'][seq_id-1]['frames'][args.frame],
                                   gt['dataset']['sequences'][seq_id-1]['frames'][args.frame])

        else:
            raise ValueError('Sequence for visualization not specified or multiple sequences given!')

    else:
        if args.sequences is None:
            args.sequences = np.arange(1, len(results['sequences']))

        for seq_id in args.sequences:
            # Create folders if they dont exist yet
            os.makedirs(os.path.join(args.output_path, args.method, 'seq%02d' % seq_id, 'tmp_frames'), exist_ok=True)

            if results['sequences'][seq_id - 1]['evaluated']:
                num_frames_in_sequence = len(results['sequences'][seq_id - 1]['frames'])

                frame_counter = 0
                for fr_id in range(num_frames_in_sequence):
                    sys.stdout.write("\rProcessing sequence %02d, image %03d / %03d" %
                                     (seq_id, (fr_id + 1), num_frames_in_sequence))
                    # feed, so it erases the previous line.
                    sys.stdout.flush()

                    seq_path = gt['dataset']['sequences'][seq_id - 1]['path'][1:]
                    seq_path_split = seq_path.split('/')

                    # Load image
                    img_name = results['sequences'][seq_id - 1]['frames'][fr_id]['img_name']
                    img_name_split = img_name.split('.')

                    img = cv2.imread(os.path.join(cfg.PATHS.DATASET, "sequences", seq_path, img_name))
                    # Convert BGR TO RGB for visualization
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Load segmentation output
                    if cfg.SEGMENTATIONS.SEQ_FIRST:
                        seg = cv2.imread(os.path.join(cfg.PATHS.SEGMENTATIONS, seq_path_split[0],
                                                      args.method, '{}.png'.format(img_name_split[0])))
                    else:
                        seg = cv2.imread(os.path.join(cfg.PATHS.SEGMENTATIONS, args.method, seq_path_split[0],
                                                      '{}.png'.format(img_name_split[0])))

                    seg = resize_image(seg, (img.shape[1], img.shape[0]))

                    # Code mask to labels
                    seg = code_mask_to_labels(seg, cfg.SEGMENTATIONS.INPUT_COLORS)

                    # Update segmentation mask with the over/under mask.
                    seg_ou = np.copy(seg)

                    # Code labels to colors
                    seg    = code_labels_to_colors(seg, cfg)
                    seg_ou = code_labels_to_colors(seg_ou, cfg)

                    # Visualize image
                    fig1 = visualize_image_for_video(img, seg, seg_ou, results['sequences'][seq_id-1]['frames'][fr_id],
                                                     gt['dataset']['sequences'][seq_id-1]['frames'][fr_id])

                    if args.export_video:
                        fig1.savefig(os.path.join(args.output_path, args.method, 'seq%02d' % seq_id,
                                                  'tmp_frames', '%08d.png' % fr_id))
                    else:
                        plt.show()

                    frame_counter += 1

            else:
                raise ValueError('<Error>: Sequence %d was not evaluated' % seq_id)

        print('\n')

        if args.export_video:
            # Export frames to video
            cmd = ['ffmpeg', '-r', '2', '-i', os.path.join(args.output_path, args.method,
                                                           'seq%02d' % seq_id, 'tmp_frames', '%08d.png'),
                   '-r', '30', os.path.join(args.output_path, args.method, 'sequence_%02d.mp4' % seq_id)]

            ret_code = subprocess.call(cmd)
            if not ret_code == 0:
                raise ValueError('Error {} executing command: {}'.format(ret_code, ' '.join(cmd)))

            # Delete temporary image files used for generating video
            if args.delete_raw_imgs:
                tmp_folder = os.path.join(args.output_path, args.method, 'seq%02d' % seq_id, 'tmp_frames')
                for filename in os.listdir(tmp_folder):
                    file_path = os.path.join(tmp_folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == "__main__":
    main()


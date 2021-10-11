import os
from yacs.config import CfgNode as CN

_C = CN()

_C.DATASET = CN()
_C.DATASET.NUM_SEQUENCES  = 94    # Number of sequences in the dataset
_C.DATASET.IMG_WIDTH      = 1278  # Width of images in the dataset
_C.DATASET.IMG_HEIGHT     = 958   # Height of images in the dataset
_C.DATASET.CAMERA_HEIGHT  = 0.7   # Height of the camera above the water surface (in meters)
_C.DATASET.DNG_ZONE_RANGE = 15    # Danger zone range (in meters)

# All Paths
_C.PATHS = CN()
_C.PATHS.RESULTS       = './results'             # Path to where the results will be saved
_C.PATHS.DATASET       = "E:/MODB/raw/"          # Path to where the dataset is stored
_C.PATHS.DATASET_CALIB = "E:/MODB/calibration/"  # Path to where calibration files are stored
_C.PATHS.SEGMENTATIONS = "E:/MODB_output/"       # Path to where the segmentation predictions are stored

# Segmentation Colors
_C.SEGMENTATIONS = CN()

_C.SEGMENTATIONS.SEQ_FIRST = True  # Sequence first or method first structure of the segmentation folder

#"""
_C.SEGMENTATIONS.INPUT_COLORS = [[  0,   0,   0],  # Obstacles RGB color code
                                 [255,   0,   0],  # Water RGB color code
                                 [  0, 255,   0]]  # Sky RGB color code
"""
_C.SEGMENTATIONS.INPUT_COLORS = [[247, 195,  37],
                                 [ 41, 167, 224],
                                 [ 90,  75, 164]]
"""

_C.SEGMENTATIONS.SKY_LABEL = [ 89,  78, 161]  # RGB Label for SKY semantic component
_C.SEGMENTATIONS.WAT_LABEL = [ 51, 168, 222]  # RGB Label for WATER semantic component
_C.SEGMENTATIONS.OBS_LABEL = [246, 193,  59]  # RGB Label for OBSTACLES semantic component
_C.SEGMENTATIONS.OVS_LABEL = [127,  51,   0]  # RGB Label for overshot water estimation
_C.SEGMENTATIONS.UNS_LABEL = [127,   0,  55]  # RGB Label for undershot water estimation
_C.SEGMENTATIONS.VOD_LABEL = [  0,   0,   0]  # RGB Label for VOID component (placeholder)

# Analysis/Evaluation Parameters
_C.ANALYSIS = CN()
_C.ANALYSIS.OBSTACLE_SIZE_CLASSES = [5*5,      # Tiny obstacles (surface area in pixels)
                                     15*15,    # Very small obstacles (surface area in pixels)
                                     30*30,    # Small obstacles (surface area in pixels)
                                     50*50,    # Medium obstacles (surface area in pixels)
                                     100*100,  # Large obstacles (surface area in pixels)
                                     200*200]  # Very large obstacles (surface area in pixels)

_C.ANALYSIS.OBSTACLE_TYPE_CLASSES = ['person',
                                     'ship',
                                     'other']

_C.ANALYSIS.MIN_OVERLAP    = 0.5     # Minimal overlap threshold (in range from 0 to 1)
_C.ANALYSIS.AREA_THRESHOLD = 5 * 5   # Surface area (in pixels). Obstacles with smaller surface area will be ignored
_C.ANALYSIS.EXPAND_LAND    = 0.01    # Alpha parameter to expand the land component (from interval 0 .. 1)
_C.ANALYSIS.EXPAND_OBJECTS = 0.01    # Alpha parameter to expand obstacles (from interval 0 .. 1)

# Visualization parameters
_C.VISUALIZATION = CN()
_C.VISUALIZATION.FONT_SIZE = 42  # Font size for matplotlib


def get_cfg(args):
    cfg = _C.clone()
    #if args.config_file:
    #    cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg
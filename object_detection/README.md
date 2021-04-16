README

The evaluation of object detection using bounding boxes on MODS dataset works as follows.

The dataset is converted to coco format and pycocotools library is used to get final results.
Some filtering of results and ground truth is performed in order to test different aspects of detector results.

Specifically, all detections in regions above the annotated sea edge are ignored.
Additionally, experiments are performed either on full image or only inside the IMU-calculated danger zone region.
Because accurately determining waterborne obstacle class is difficult, the option to ignore class is present.

Results are returned as F1 score for all objects, small, medium and large sized objects, i.e. for every detector you will get 4 numbers for a single experiment setup.

For evaluation you will need the results json file. The json structure needs to be exactly the same as the modd3.json annotation file proveide along with the dataset.
Each detection should be define with its bounding box (x,y,width,height) and class (ship, person, other).

The script evaluation.py should then be used to get the final results.
Additional functions, such as displaying the results per frames are available in script evaluate_objects_detection.py but they are still WIP.
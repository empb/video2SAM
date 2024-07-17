# Video Labeling Using SAM

**Installation**: Run `$ ./install_SAM.sh` to install SAM.

### Program Execution

The program parameters are:
`usage: video2SAM.py [-h] --input_video INPUT_VIDEO [--label_colors LABEL_COLORS] [--load_folder LOAD_FOLDER] [--output_folder OUTPUT_FOLDER] [--backup_folder BACKUP_FOLDER]`

* **--input_video**: Input video file.
* **--load_folder**: Directory from which masks can be loaded when starting the program. Default is `annotations/`.
* **--label_colors**: File containing the class and color information (in RGB) for segmentation. Default is `label_colors.txt`. It follows the [KITTI labeling format](https://docs.cvat.ai/docs/manual/advanced/formats/format-kitti/).
* **--output_folder**: Directory where masks can be saved at the end of the program. Default is `annotations/`.
* **--backup_folder**: Directory where backup copies of the work can be stored. Default is `backups/`.

The controls are displayed in the console.

A useful feature is that you can hold the mouse click to add points while holding the next frame button. This way, manual tracking of all objects can be done to later call SAM for all frames.

**Note**: When loading a set of masks, it is necessary that the colors used in them match those in the file with the color information.

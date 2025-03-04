# Video Labeling Using SAM

**Installation**: Run `$ ./install_SAM.sh` to install SAM.

### Program Execution

The program parameters are:
`usage: video2SAM.py [-h] --input_video INPUT_VIDEO [--label_colors LABEL_COLORS] [--load_folder LOAD_FOLDER] [--output_folder OUTPUT_FOLDER] [--backup_folder BACKUP_FOLDER] [--sam_model_folder SAM_MODEL_FOLDER]`

* **--input_video**: Input video file.
* **--load_folder**: Directory from which masks can be loaded when starting the program. Default is `annotations/`.
* **--label_colors**: File containing the class and color information (in RGB) for segmentation. Default is `label_colors.txt`. It follows the [KITTI labeling format](https://docs.cvat.ai/docs/manual/advanced/formats/format-kitti/).
* **--output_folder**: Directory where masks can be saved at the end of the program. Default is `annotations/`.
* **--backup_folder**: Directory where backup copies of the work can be stored. Default is `backups/`.
* **--sam_model_folder**: Directory where the SAM model is loaded/saved. Default is `models/`.

The controls are displayed in the console.

A useful feature is that you can hold the mouse click to add points while holding the next frame button. This way, manual tracking of all objects can be done to later call SAM for all frames.

**Note**: When loading a set of masks, it is necessary that the colors used in them match those in the file with the color information.

**Note 2**: Bounding boxes of the objects can be exported. These will be loaded in the file <load_folder>/bboxes. The format is as follows:
```
class_name instance_number bbox_x bbox_y bbox_w bbox_h
cup 1 220 330 50 65
yellow_duck 2 100 280 40 60
...
```

# Etiquetado de video utilizando SAM 

**Instalación**: ejecutar `$ ./install_SAM.sh` para instalar SAM.

### Ejecución del programa

Los parámetros del programa son:
`usage: video2SAM.py [-h] --input_video INPUT_VIDEO [--label_colors LABEL_COLORS] [--load_folder LOAD_FOLDER] [--output_folder OUTPUT_FOLDER] [--backup_folder BACKUP_FOLDER] [--sam_model_folder SAM_MODEL_FOLDER]
`
* **--input_video**: Video que se recibe como entrada.
* **--load_folder**: Directorio de donde se pueden cargar las máscaras al iniciar el programa. Por defecto es `annotations/`.
* **--label_colors**: Se pasa el fichero que tiene la información de las clases y colores (en RGB) para la segmentación. Por defecto es `label_colors.txt`. Sigue el [formato de etiquetado de KITTI](https://docs.cvat.ai/docs/manual/advanced/formats/format-kitti/).
* **--output_folder**: Directorio donde se guardan (si así se desea) las máscaras al final del programa. Por defecto es `annotations/`.
* **--backup_folder**: Directorio donde se pueden ir almacenando copias de seguridad del trabajo. Por defecto es `backups/`.
* **--sam_model_folder**: Directorio donde se carga/descarga el modelo de SAM. Por defecto es `models/`.

Los controles se muestran por consola.

Una feature de utilidad es que se puede mantener el click del ratón para añadir puntos mientras se mantiene el botón de siguiente frame. De esta forma, se puede hacer un tracking manual de todos los objetos para posteriormente poder llamar a SAM de todos los frames. 

**Nota**: Cuando se carga un conjunto de máscaras es necesario que los colores utilizados en ellas sean los mismos que los del fichero con la información de los colores.

**Nota 2**: Se pueden exportar las bounding boxes de los objetos. Estas se cargarán en el fichero <load_folder>/bboxes. El formato es el siguiente:
```
nombre_clase num_instancia bbox_x bbox_y bbox_w bbox_h
cup 1 220 330 50 65
yellow_duck 2 100 280 40 60
...
```
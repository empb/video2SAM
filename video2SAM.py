#!/usr/bin/env python

##########################################################################
# > ./video2SAM.py --input_video video2.MOV
##########################################################################

# The following environment variable is needed because otherwise SAM pytorch model 
# present racing conditions on some CUDA kernel executions (06/07/2024):
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Normal needed imports:
import requests
import argparse
import cv2
import numpy as np
import time
import zipfile
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Slow imports (only loaded if needed; they take at least a few seconds just to be loaded,
# so we only load them if all arguments are OK, and there are images to process):
def make_slow_imports():
    global SamPredictor, sam_model_registry, predict
    from segment_anything import SamPredictor, sam_model_registry

# Segment anything network initialization:
def init_SAM_predictor(folder):
    print('> Initializing SAM model...')
    # !wget -O sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    # file_path = '/home/user/workspace/models/SAM/sam_vit_h_4b8939.pth'
    if folder[-1] != '/': folder += '/'
    file_path = folder + 'sam_vit_h_4b8939.pth' 
    if not os.path.exists(file_path):
        print('  Downloading model...')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        response = requests.get(url)    
        with open(file_path, 'wb') as file:
            file.write(response.content)
            print(f'    Model downloaded and saved to {file_path}')
    else:
        print(f'    Model was already available in {file_path}.')
    sam = sam_model_registry['vit_h'](checkpoint=file_path)
    # Please note: the following two alternative models could also be used:
    #   sam = sam_model_registry['vit_b'](checkpoint='/home/user/workspace/models/SAM/sam_vit_b_01ec64.pth')
    #   sam = sam_model_registry['vit_l'](checkpoint='/home/user/workspace/models/SAM/sam_vit_l_0b3195.pth')
    #   sam = sam_model_registry['vit_b'](checkpoint='../../../workspace/models/SAM/sam_vit_b_01ec64.pth')
    #   sam = sam_model_registry['vit_l'](checkpoint='../../../workspace/models/SAM/sam_vit_l_0b3195.pth')
    # But we will stay here with the best (huge) segmentation model.
    sam.to(device='cuda')
    predictor = SamPredictor(sam)
    return predictor

##########################################################################
# Argument parsing and file management
##########################################################################

# Argument parsing:
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Processes a video and annotates it with the SAM model using the mouse.')
    # File parameters:
    parser.add_argument('--input_video', type=str, required=True, help='Input video')
    parser.add_argument('--label_colors', type=str, default='label_colors.txt', help='File with label colors')
    parser.add_argument('--load_folder', type=str, default='annotations/', help='Folder with masks to load')
    parser.add_argument('--output_folder', type=str, default='annotations/', help='Output folder for masks')
    parser.add_argument('--backup_folder', type=str, default='backups/', help='Folder for backups')
    parser.add_argument('--sam_model_folder', type=str, default='models/', help='Folder to store/load the SAM model')
    args = parser.parse_args()
    return args

# Video opening:
def frames_from_video(video_path):
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    # Get frames
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    # Close video
    cap.release()
    return frames

# Read colors from file:
# returns a dictionary where the key is the label name and the value is the RGB color
def labels_colors_from_file(file_path):
    labels = {}
    labels_order = []
    with open(file_path, 'r') as file:
        for line in file:
            # Format is: R G B LABELNAME. See KITTI annotations format
            # (https://docs.cvat.ai/docs/manual/advanced/formats/format-kitti/)
            r, g, b, label = line.split()
            if label != 'background':
                labels[label] = (int(r), int(g), int(b))
                labels_order.append(label)
    return labels_order, labels

# Load masks from folder:
def load_masks(folder):
    if not os.path.exists(folder + 'semantic_rgb/') or not os.path.exists(folder + 'instance/'):
        print(f'! Error: Folder {folder} does not contain semantic_rgb/ or instance/ subfolders')
        return None, None
    if folder[-1] != '/': folder += '/'
    # Take all the files in the folder
    filenames_mask = os.listdir(folder + 'semantic_rgb/')
    filenames_mask.sort()
    filenames_inst = os.listdir(folder + 'instance/')
    filenames_inst.sort()
    if len(filenames_mask) != len(filenames_inst):
        print(f'! Error: Number of semantic and instance masks do not match ({len(filenames_mask)} != {len(filenames_inst)})')
        return None, None

    # Load the masks and instances
    print(f'    Loading masks from {folder}... ')
    masks, instances = [], []
    folder_sem, folder_inst = folder + 'semantic_rgb/', folder + 'instance/'
    for file in filenames_mask:
        masks.append(cv2.cvtColor(cv2.imread(folder_sem + file), cv2.COLOR_BGR2RGB))
    for file in filenames_inst:
        # instances must be (H, W, 1) and not (H, W, 3)
        instances.append(cv2.imread(folder_inst + file, cv2.IMREAD_GRAYSCALE))
    # convert (H, W) to (H, W, 1)
    instances = [np.expand_dims(inst, axis=2) for inst in instances]
    print('    ... done!')
    return masks, instances

# Save masks in folder:
def save_masks(folder, sem_masks, instances, bboxes, label_colors, is_backup=False):
    if folder[-1] != '/': folder += '/'
    # If is a backup create a subfolder with time
    if is_backup:   folder += time.strftime('%Y%m%d_%H%M%S') + '/'
    elif os.path.exists(folder):        # if is not a backup and folder exists, clear the files
        os.system(f'rm -r {folder}*')
    # Loop over masks and save them
    print(f'    Saving masks in {folder}... ')
    # Save the semantic masks
    folder_sem = folder + 'semantic_rgb/'
    if not os.path.exists(folder_sem): os.makedirs(folder_sem)
    for i, mask in enumerate(sem_masks):
        cv2.imwrite(folder_sem + f'frame_{i:06d}.png', cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    # Save the instance masks
    folder_inst = folder + 'instance/'
    if not os.path.exists(folder_inst): os.makedirs(folder_inst)
    for i, mask in enumerate(instances):
        cv2.imwrite(folder_inst + f'frame_{i:06d}.png', cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    # Save the bboxes
    folder_bboxes = folder + 'bboxes/'
    if not os.path.exists(folder_bboxes): os.makedirs(folder_bboxes)
    # Create a diccionary where key is the color and value is the label
    colors_labels = {v: k for k, v in label_colors.items()}
    for i, bboxes_frame in enumerate(bboxes):
        with open(folder_bboxes + f'frame_{i:06d}.txt', 'w') as f:
            for bbox in bboxes_frame:
                (x, y, w, h), color, inst_id = bbox
                label = colors_labels[color]
                f.write(f'{label} {inst_id} {x} {y} {w} {h}\n')
    print('    ... done!')

# Create a zip file with the KITTI format
def create_zip_kitti(output_file, sem_masks_rgb, instance_masks, bboxes, colors_file):
    with zipfile.ZipFile(output_file, 'w') as zipf:
        # Go through the masks
        for i, (sem_rgb, inst_masks) in enumerate(zip(sem_masks_rgb, instance_masks)):
            zipf.writestr(f'kitti/default/instance/frame_{i:06d}.png', cv2.imencode('.png', cv2.cvtColor(inst_masks, cv2.COLOR_RGB2BGR))[1].tobytes())
            zipf.writestr(f'kitti/default/semantic_rgb/frame_{i:06d}.png', cv2.imencode('.png', cv2.cvtColor(sem_rgb, cv2.COLOR_RGB2BGR))[1].tobytes())
        # Go through the bboxes
        colors_labels = {v: k for k, v in label_colors.items()}
        for i, bboxes_frame in enumerate(bboxes):
            lines = []
            for bbox in bboxes_frame:
                (x, y, w, h), color, inst_id = bbox
                label = colors_labels[color]
                lines.append(f'{label} {inst_id} {x} {y} {w} {h}')
            zipf.writestr(f'kitti/default/bboxes/frame_{i:06d}.txt', '\n'.join(lines))
        # Add the label colors file
        with open(colors_file, 'r') as file:
            zipf.writestr('kitti/label_colors.txt', file.read())
    print(f'    ZIP file created: {output_file}')

# Load bboxes from file:
def bboxes_from_file(folder, labels_colors):
    if folder[-1] != '/': folder += '/'
    # Take all the files in the folder
    folder += 'bboxes/'
    filenames = os.listdir(folder)
    filenames.sort()
    # Load the bboxes
    print(f'    Loading bboxes from {folder}... ')
    bboxes = []
    for file in filenames:
        bboxes.append([])
        with open(folder + file, 'r') as f:
            for line in f:
                # Format is: class num_inst bbox_x bbox_y bbox_w bbox_h
                label, num_inst, x, y, w, h = line.split()
                bboxes[-1].append(((int(x), int(y), int(w), int(h)), labels_colors[label], int(num_inst)))
    print('    ... done!')
    return bboxes

##########################################################################
# Console printing
##########################################################################

# Print text in console with color:
def print_color(text, tuple_color, is_background=False):
    r, g, b = tuple_color
    if is_background:
        print(f'\033[48;2;{r};{g};{b}m' + '\033[38;2;0;0;0m' + text + '\033[0m')
    else:
        print(f'\033[38;2;{r};{g};{b}m' + text + '\033[0m')

# Print text in console with color:
def print_console(current_label, label_colors, current_frame, total_frames, tam_ker_op):
    print(chr(27) + '[2J')  # Clear screen
    print('#'*70)
    print("""CONTROLS:
    [left click]: Add positive point for SAM
    [right click]: Add negative point for SAM
    [c]: Change current label
    [s]: Call SAM for current frame
    [a]: Call SAM for all frames
    [t]: Clear current label points for current frame
    [p]: Clear all points for all frames
    [r]: Reset mask for current label and frame
          
    [+/-]: Increase/decrease opening kernel size
    
    [,]: Go to previous frame
    [.]: Go to next frame
          
    [v]: Show/hide mask
    [b]: Show/hide bboxes
    [i]: Swap between semantic and instance masks
    [k]: Create backup
    
    [q] or [ESC]: Quit
    """)
    print('-'*70)
    print('Current frame: {}/{}'.format(current_frame, total_frames-1))
    print('Current opening kernel size:', tam_ker_op)
    print('-'*70)
    print('Current label:')
    for label, color in label_colors.items():
        if label == current_label:
            print_color('    ' + label + ' '*(66-len(label)), color, is_background=True)
        else:
            print_color('    ' + label, color)
    print('#'*70)

##########################################################################
# OpenCV window management
##########################################################################

# Frame navigation:
def navigate_frames(frames, label_colors, sam_predictor, backup_folder, masks, bboxes, instances):
    # PARAMETERS
    H_frame, W_frame = frames[0].shape[:2]
    current_frame, total_frames = 0, len(frames)
    left_button_down, right_button_down = False, False
    last_point = None
    show_mask, show_bboxes, show_instances = True, False, False
    # Possitive and negative points for SAM
    # points are ( (x, y), label )
    positive_points = [[] for _ in range(total_frames)]
    negative_points = [[] for _ in range(total_frames)]
    # Masks and bboxes
    if masks is None:
        masks = [np.zeros((frames[0].shape[0], frames[0].shape[1], 3), dtype=np.uint8) for _ in range(total_frames)]
    if bboxes is None:
        bboxes = [[] for _ in range(total_frames)]
    if instances is None:
        instances = [np.zeros((frames[0].shape[0], frames[0].shape[1], 1), dtype=np.uint8) for _ in range(total_frames)]

    # Mouse callback function
    def click_event(event, x, y, flags, param):
        nonlocal current_frame, left_button_down, right_button_down, last_point, show_mask, H_frame, W_frame, current_label
        if event == cv2.EVENT_LBUTTONDOWN:
            left_button_down = True
            if 0 <= x < W_frame and 0 <= y < H_frame:
                positive_points[current_frame].append(((x, y), current_label))
                update_frame(show_mask, show_bboxes, show_instances, label_colors)  # Update frame after adding a point
        elif event == cv2.EVENT_LBUTTONUP:
            left_button_down = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            if 0 <= x < W_frame and 0 <= y < H_frame:
                right_button_down = True
                negative_points[current_frame].append(((x, y), current_label))
                update_frame(show_mask, show_bboxes, show_instances, label_colors)  # Update frame after adding a point
        elif event == cv2.EVENT_RBUTTONUP:
            right_button_down = False
        # Save the last point tracked
        last_point = (x, y)

    # Function to update the frame
    def update_frame(show_mask, show_bboxes, show_instances, label_colors):
        nonlocal current_frame
        frame_copy = frames[current_frame].copy()
        if show_instances:
            mask = instances2rgb(instances[current_frame])
        else:
            mask = cv2.cvtColor(masks[current_frame], cv2.COLOR_RGB2BGR)
        # Draw the points on the frame
        for point in positive_points[current_frame]:
            cv2.circle(frame_copy, point[0], 12, (0, 255, 0), -1)
            color = (label_colors[point[1]][2], label_colors[point[1]][1], label_colors[point[1]][0])
            cv2.circle(frame_copy, point[0], 9, color, -1)
        for point in negative_points[current_frame]:
            cv2.circle(frame_copy, point[0], 12, (0, 0, 255), -1)
            color = (label_colors[point[1]][2], label_colors[point[1]][1], label_colors[point[1]][0])
            cv2.circle(frame_copy, point[0], 9, color, -1)
        # Draw the bboxes on the frame
        if show_bboxes:
            for bbox in bboxes[current_frame]:
                (x, y, w, h), color, _ = bbox
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (color[2], color[1], color[0]), 3)
                if show_mask:
                    cv2.rectangle(mask, (x, y), (x+w, y+h), (color[2], color[1], color[0]), 3)
        # Show the frame
        if not show_mask:
            cv2.imshow('Video', frame_copy)
        else:   # Stack horzontally frame and mask
            frame_copy = add_frame_mask(frame_copy, mask)
            cv2.imshow('Video', np.hstack((frame_copy, mask)))

    # Create window and set callback
    cv2.namedWindow('Video', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('Video', click_event)
    update_frame(show_mask=True, show_bboxes=False, show_instances=False, label_colors=label_colors)

    # Screen loop
    labels_list = list(label_colors.keys())
    current_label = labels_list[0]
    tam_ker_op = 0
    print_console(current_label, label_colors, current_frame, total_frames, tam_ker_op)
    set_image = True
    while True:
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27: break # 'q' or 'ESC' key
        elif key == ord(','):   # Previous frame
            current_frame = max(0, current_frame - 1)
            if left_button_down and 0 <= last_point[0] < W_frame and 0 <= last_point[1] < H_frame:
                positive_points[current_frame].append((last_point, current_label))
            elif right_button_down and 0 <= last_point[0] < W_frame and 0 <= last_point[1] < H_frame:
                negative_points[current_frame].append((last_point, current_label))
            print_console(current_label, label_colors, current_frame, total_frames, tam_ker_op)
            update_frame(show_mask, show_bboxes, show_instances, label_colors)
            set_image = True
        elif key == ord('.'):   # Next frame
            current_frame = min(total_frames - 1, current_frame + 1)
            if left_button_down and 0 <= last_point[0] < W_frame and 0 <= last_point[1] < H_frame:
                positive_points[current_frame].append((last_point, current_label))
            elif right_button_down and 0 <= last_point[0] < W_frame and 0 <= last_point[1] < H_frame:
                negative_points[current_frame].append((last_point, current_label))
            print_console(current_label, label_colors, current_frame, total_frames, tam_ker_op)
            update_frame(show_mask, show_bboxes, show_instances, label_colors)
            set_image = True
        elif key == ord('r'):  # Reset mask for current label and frame
            # When color is equal to label_color[current_label], set it to (0, 0, 0)
            mask_bool = np.all(masks[current_frame] == label_colors[current_label], axis=2)
            masks[current_frame][mask_bool] = (0, 0, 0)
            # Remove instances from the mask
            instances2delete = [bbox[2] for bbox in bboxes[current_frame] if bbox[1] == label_colors[current_label]]
            for inst in instances2delete:
                instances[current_frame][instances[current_frame] == inst] = 0
            # Adjust the instances numbers
            instances_copy = instances[current_frame].copy()
            for inst in instances2delete:
                instances[current_frame][instances_copy > inst] -= 1
            # Remove bboxes too
            new_bboxes = []
            for bbox in bboxes[current_frame]:
                if bbox[1] != label_colors[current_label]: # if bbox is not from the current label
                    # Upadate the instance id
                    nbbox = bbox
                    for inst in instances2delete:
                        if bbox[2] > inst:
                            nbbox = ((nbbox[0][0], nbbox[0][1], nbbox[0][2], nbbox[0][3]), nbbox[1], nbbox[2]-1)
                    new_bboxes.append(nbbox)
            bboxes[current_frame] = new_bboxes
            update_frame(show_mask, show_bboxes, show_instances, label_colors)
        elif key == ord('p'):  # Clear all points for all frames
            positive_points = [[] for _ in range(total_frames)]
            negative_points = [[] for _ in range(total_frames)]
            update_frame(show_mask, show_bboxes, show_instances, label_colors)
        elif key == ord('t'):  # Clear all points for current frame and current label
            positive_points[current_frame] = [ p for p in positive_points[current_frame] if p[1] != current_label]
            negative_points[current_frame] = [ p for p in negative_points[current_frame] if p[1] != current_label]
            update_frame(show_mask, show_bboxes, show_instances, label_colors)
        elif key == ord('c'):  # Change label
            current_label = labels_list[(labels_list.index(current_label) + 1) % len(labels_list)]
            update_frame(show_mask, show_bboxes, show_instances, label_colors)
            print_console(current_label, label_colors, current_frame, total_frames, tam_ker_op)
        elif key == ord('s') or key == ord('a'):  # Call SAM for current frame or all frames
            running_frames = []
            if key == ord('s'):
                if len(positive_points[current_frame]) != 0: 
                    running_frames = [current_frame]
            else: running_frames = [ i for i in range(total_frames) if len(positive_points[i]) != 0]
            t0 = time.time()
            for i in running_frames:
                print('> Calling SAM for frame', i)
                for label in label_colors.keys():
                    pp = [point[0] for point in positive_points[i] if point[1] == label]
                    if pp != []:
                        pn = [point[0] for point in negative_points[i] if point[1] == label]
                        mask = sam_processing(frames[i], sam_predictor, pp, pn, set_image=set_image)
                        set_image = False
                        # Mask opening operation
                        if tam_ker_op > 0:
                            mask = cv2.morphologyEx(np.array(mask, dtype=np.uint8), cv2.MORPH_OPEN, np.ones((tam_ker_op, tam_ker_op), np.uint8))
                        # Save the mask with color
                        masks[i][mask != 0] = label_colors[label]
                        # Add instance to the mask
                        id_instance = len(np.unique(instances[i]))
                        instances[i][mask != 0] = id_instance
                        # Save the bbox ((x, y, w, h), color, instance_id)
                        bboxes[i].append((cv2.boundingRect(np.array(mask, dtype=np.uint8)), label_colors[label], id_instance))
                set_image = True
                # Clear the points
                positive_points[i] = []
                negative_points[i] = []
            print(f'> Done! {len(running_frames)} frames segmented in {time.time() - t0:.2f} seconds.')
            # Update the frame
            update_frame(show_mask, show_bboxes, show_instances, label_colors)
            set_image = False
            if key == ord('a'): # If all frames, clear all points
                positive_points = [[] for _ in range(total_frames)]
                negative_points = [[] for _ in range(total_frames)]
        elif key == ord('v'):   # Show/hide mask
            show_mask = not show_mask
            update_frame(show_mask, show_bboxes, show_instances, label_colors)
        elif key == ord('b'):   # Show/hide bboxes
            show_bboxes = not show_bboxes
            update_frame(show_mask, show_bboxes, show_instances, label_colors)
        elif key == ord('i'):   # Swap between semantic and instance masks
            show_instances = not show_instances
            update_frame(show_mask, show_bboxes, show_instances, label_colors)
        elif key == ord('k'):   # Create backup
            save_masks(backup_folder, masks, instances, bboxes, label_colors, is_backup=True)
        elif key == ord('+'):   # Increase opening kernel size
            tam_ker_op += 1
            print_console(current_label, label_colors, current_frame, total_frames, tam_ker_op)
        elif key == ord('-'):   # Decrease opening kernel size
            tam_ker_op = max(0, tam_ker_op - 1)
            print_console(current_label, label_colors, current_frame, total_frames, tam_ker_op)
        # elif key == ord('d'):   # Just for debugging
        #     update_frame(show_mask, show_bboxes, show_instances, label_colors)

    cv2.destroyAllWindows()
    return masks, instances, bboxes

##########################################################################
# Image processing functions
##########################################################################

# Fucntion to call SAM for a frame, return the mask of the frame
def sam_processing(frame, sam_predictor, positive_points, negative_points, set_image=True):
    # Calling the SAM predictor
    if set_image:
        sam_predictor.set_image(frame)
    masks, _, _ = sam_predictor.predict(
                point_coords=np.array(positive_points + negative_points),
                point_labels=np.array([1]*len(positive_points) + [0]*len(negative_points)),
                multimask_output=False
            )
    # With multimask_output=False, masks has always just one element
    return masks[0]

# Add the mask to the frame
def add_frame_mask(frame, mask):
    # Convert mask to 3 channels if needed
    if mask.shape[:2] == 1:
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_mask[mask == 1] = (255, 255, 255)
    else:
        colored_mask = mask
    # Add transparency to the mask  
    colored_mask_rgba = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2BGRA)
    colored_mask_rgba[:, :, 3] = 200
    # Add the mask to the frame
    rbga_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    cv2.addWeighted(rbga_frame, 1, colored_mask_rgba, 0.5, 0, rbga_frame)
    return cv2.cvtColor(rbga_frame, cv2.COLOR_BGRA2BGR)

# Convert instances map to RGB
def instances2rgb(instance_map):
    # Create the colormap
    cmap = plt.get_cmap('tab20')
    # Normalize the image values to the colormap range
    norm = Normalize(vmin=instance_map.min(), vmax=instance_map.max())
    # Apply the colormap to the normalized image and convert to RGB
    rgb_image = cmap(norm(instance_map))
    # Remove the alpha channel and convert to 0-255 values
    rgb_image = (np.squeeze(rgb_image)[:, :, :3] * 255).astype(np.uint8)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)
    # Set the background to black
    mask = np.zeros(instance_map.shape, dtype=np.uint8)
    mask[instance_map != 0] = 1
    return rgb_image*mask

# Create bboxes from masks
def bboxes_from_masks(sem_masks, instances):
    print(f'    Computing bboxes from masks...')
    bboxes = [[] for _ in range(len(sem_masks))]
    for i in range(len(sem_masks)):
        unique_instances = np.unique(instances[i])[1:] # 0 is the background
        for inst_id in unique_instances:
            # Find where the instance is equal to the actual
            y, x = np.where(np.squeeze(instances[i]) == inst_id)
            # Bounding box
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            # Extract the region of interest
            roi = sem_masks[i][y_min:y_max+1, x_min:x_max+1]
            # Convert roi to list of colors and counts
            colors, counts = np.unique(roi.reshape(-1, 3), axis=0, return_counts=True)
            # Get the most common color
            fcolor = colors[np.argmax(counts)]
            # If this is the background color, take the second most common
            if np.all(fcolor == (0, 0, 0)):
                fcolor = colors[np.argsort(counts)[-2]]
            # Save the bbox
            bboxes[i].append(((x_min, y_min, x_max-x_min+1, y_max-y_min+1), (int(fcolor[0]), int(fcolor[1]), int(fcolor[2])), inst_id))
    print('    ... done!')
    return bboxes

##########################################################################
# Main
##########################################################################
if __name__ == '__main__':
    # 0. Parse input arguments:
    args = parse_arguments()

    # 1. Read video:
    print(f'> Opening video {args.input_video}...')
    frames = frames_from_video(args.input_video)
    if frames is None:
        print(f'! Error opening video {args.input_video}')
        exit(1)
    print(f'    Video opened with {len(frames)} frames.')

    # 2. Read labels (dict where key is label name and value is RGB color)
    labels_order, label_colors = labels_colors_from_file(args.label_colors)

    # 3. Load masks from folder:
    if args.load_folder[-1] != '/': args.load_folder += '/'
    loaded_masks, loaded_bboxes, loaded_instances = None, None, None
    if os.path.exists(args.load_folder):
        answer = input(f'> Do you want to load masks from {args.load_folder}? (y/n): ')
        while answer.lower() != 'y' and answer.lower() != 'n':
            answer = input('    ! Please answer with \'y\' or \'n\': ')
        if answer.lower() == 'y':
            loaded_masks, loaded_instances = load_masks(args.load_folder)

            if loaded_masks is not None and len(frames) != len(loaded_masks):
                print(f'! Error: Number of frames in video and masks do not match ({len(frames)} != {len(loaded_masks)})')
                if len(loaded_masks) < len(frames):
                    answer = input('> Continue creating new masks? (y/n): ')
                    while answer.lower() != 'y' and answer.lower() != 'n':
                        answer = input('    ! Please answer with \'y\' or \'n\': ')
                    if answer.lower() == 'y':
                        # Complete the loaded masks with empty masks
                        loaded_masks += [np.zeros((frames[0].shape[0], frames[0].shape[1], 3), dtype=np.uint8) for _ in range(len(frames) - len(loaded_masks))]
                        loaded_instances += [np.zeros((frames[0].shape[0], frames[0].shape[1], 1), dtype=np.uint8) for _ in range(len(frames) - len(loaded_instances))]
                    else: exit(1)
            # If the folder loadfolder/bboxes exitsts and its not empty
            if os.path.exists(args.load_folder + 'bboxes/') and os.listdir(args.load_folder + 'bboxes/'):
                loaded_bboxes = bboxes_from_file(args.load_folder, label_colors)
            else:
                loaded_bboxes = bboxes_from_masks(loaded_masks, loaded_instances)

    # 4. Initialize the SAM model:
    make_slow_imports()
    sam_predictor = init_SAM_predictor(args.sam_model_folder)

    # 5. Navigate through frames and click the points
    sem_masks, instances, bboxes = navigate_frames(frames, label_colors, sam_predictor, 
            args.backup_folder, loaded_masks, loaded_bboxes, loaded_instances)

    # 6. Ask for saving the masks
    answer = input('> Do you want to save the masks?\n  (Process will overwrite files in \'' + args.output_folder + '\') (y/n): ')
    while answer.lower() != 'y' and answer.lower() != 'n':
        answer = input('    ! Please answer with \'y\' or \'n\': ')
    if answer.lower() == 'y':
        save_masks(args.output_folder, sem_masks, instances, bboxes, label_colors)
        # 7. Ask for exporting to KITTI format
        answer = input('> Do you want to export in KITTI format (for importing to CVAT)? (y/n): ')
        while answer.lower() != 'y' and answer.lower() != 'n':
            answer = input('    ! Please answer with \'y\' or \'n\': ')
        if answer.lower() == 'y':
            answer = input('    Output file for the KITTI exportation [default: kitti/kitti.zip]: ')
            if answer == '': 
                kitti_folder = 'kitti/'
                kitti_file = 'kitti.zip'
            else:
                kitti_folder, kitti_file = os.path.split(answer)
            # If folder does not exist, create it
            if not os.path.exists(kitti_folder): os.makedirs(kitti_folder)
            # Save the masks in KITTI format
            create_zip_kitti(kitti_folder + kitti_file, sem_masks, instances, bboxes, args.label_colors)

    exit(0)
    
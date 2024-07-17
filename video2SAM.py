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

# Slow imports (only loaded if needed; they take at least a few seconds just to be loaded,
# so we only load them if all arguments are OK, and there are images to process):
def make_slow_imports():
    global SamPredictor, sam_model_registry, load_model, load_image, predict, annotate, torch, transforms
    from segment_anything import SamPredictor, sam_model_registry
    import torch
    import torchvision.transforms as transforms

# Segment anything network initialization:
def init_SAM_predictor():
    print('Initializing SAM model...')
    # !wget -O sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    # file_path = '/home/user/workspace/models/SAM/sam_vit_h_4b8939.pth'
    file_path = '../../../workspace/models/SAM/sam_vit_h_4b8939.pth' 
    if not os.path.exists(file_path):
        print('  Downloading model...')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        response = requests.get(url)    
        with open(file_path, 'wb') as file:
            file.write(response.content)
            print(f'  Model downloaded and saved to {file_path}')
    else:
        print(f'  Model was already available in {file_path}.')
    sam = sam_model_registry['vit_h'](checkpoint='../../../workspace/models/SAM/sam_vit_h_4b8939.pth')
    # Please note: the following two alternative models could also be used:
    #   sam = sam_model_registry['vit_b'](checkpoint='/home/user/workspace/models/SAM/sam_vit_b_01ec64.pth')
    #   sam = sam_model_registry['vit_l'](checkpoint='/home/user/workspace/models/SAM/sam_vit_l_0b3195.pth')
    #   sam = sam_model_registry['vit_b'](checkpoint='../../../workspace/models/SAM/sam_vit_b_01ec64.pth')
    #   sam = sam_model_registry['vit_l'](checkpoint='../../../workspace/models/SAM/sam_vit_l_0b3195.pth')
    # But we will stay here with the best (huge) segmentation model.
    sam.to(device='cuda')
    predictor = SamPredictor(sam)
    print('Done!')
    return predictor

##########################################################################
# Argument parsing and file management
##########################################################################

# Argument parsing:
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Processes a (potentially large) video and annotates it with the SAM model using the mouse.')
    # File parameters:
    parser.add_argument('--input_video', type=str, required=True, help='Input video')
    parser.add_argument('--label_colors', type=str, default='label_colors.txt', help='File with label colors')
    parser.add_argument('--load_folder', type=str, default='annotations/', help='Folder with masks to load')
    parser.add_argument('--output_folder', type=str, default='annotations/', help='Output folder for masks')
    parser.add_argument('--backup_folder', type=str, default='backups/', help='Folder for backups')
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
    with open(file_path, 'r') as file:
        for line in file:
            # Format is: R G B LABELNAME. See KITTI annotations format
            # (https://docs.cvat.ai/docs/manual/advanced/formats/format-kitti/)
            r, g, b, label = line.split()
            if label != 'background':
                labels[label] = (int(r), int(g), int(b))
    return labels

# Load masks from folder:
def load_masks(folder):
    if folder[-1] != '/': folder += '/'
    masks = []
    print(f'Loading masks from {folder}... ', end='')
    filenames = os.listdir(folder)
    filenames.sort()
    for file in filenames:
        masks.append(cv2.cvtColor(cv2.imread(folder + file), cv2.COLOR_BGR2RGB))
    print('done!')
    return masks

# Save masks in folder:
def save_masks(folder, masks, is_backup=False):
    if folder[-1] != '/': folder += '/'
    # If is a backup create a subfolder with time
    if is_backup:   folder += time.strftime('%Y%m%d_%H%M%S') + '/'
    # If folder does not exist, create it
    if not os.path.exists(folder): os.makedirs(folder)
    # Loop over masks and save them
    print(f'Saving masks in {folder}... ', end='')
    for i, mask in enumerate(masks):
        cv2.imwrite(folder + f'frame_{i:06d}.png', cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    print('done!')

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
    [t]: Clear all points for current frame
    [p]: Clear all points for all frames
    [r]: Reset mask for current label and frame
          
    [+/-]: Increase/decrease opening kernel size
    
    [,]: Go to previous frame
    [.]: Go to next frame
          
    [v]: Show/hide mask
    [b]: Show/hide bboxes
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
def navigate_frames(frames, label_colors, sam_predictor, backup_folder, masks):
    # Flags for mouse buttons
    current_frame, total_frames = 0, len(frames)
    left_button_down, right_button_down = False, False
    last_point = None
    show_mask, show_bboxes = True, False
    # Possitive and negative points for SAM
    positive_points = [[] for _ in range(total_frames)]
    negative_points = [[] for _ in range(total_frames)]
    # Masks and bboxes
    if masks is None:
        masks = [np.zeros((frames[0].shape[0], frames[0].shape[1], 3), dtype=np.uint8) for _ in range(total_frames)]
    bboxes = [[] for _ in range(total_frames)]
    # To control if we must run the SAM predictor in each frame
    run_sam_frames = [False for _ in range(total_frames)]

    # Mouse callback function
    def click_event(event, x, y, flags, param):
        nonlocal current_frame, left_button_down, right_button_down, last_point, show_mask
        if event == cv2.EVENT_LBUTTONDOWN:
            left_button_down = True
            positive_points[current_frame].append((x, y))
            run_sam_frames[current_frame] = True
            update_frame(show_mask, show_bboxes)  # Update frame after adding a point
        elif event == cv2.EVENT_LBUTTONUP:
            left_button_down = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            right_button_down = True
            negative_points[current_frame].append((x, y))
            update_frame(show_mask, show_bboxes)  # Update frame after adding a point
        elif event == cv2.EVENT_RBUTTONUP:
            right_button_down = False
        # Save the last point tracked
        last_point = (x, y)

    # Function to update the frame
    def update_frame(show_mask, show_bboxes):
        nonlocal current_frame
        frame_copy = frames[current_frame].copy()
        mask = cv2.cvtColor(masks[current_frame], cv2.COLOR_RGB2BGR)
        # Draw the points on the frame
        for point in positive_points[current_frame]:
            cv2.circle(frame_copy, point, 12, (0, 255, 0), -1)
            cv2.circle(frame_copy, point, 9, (255, 255, 255), -1)
        for point in negative_points[current_frame]:
            cv2.circle(frame_copy, point, 12, (0, 0, 255), -1)
            cv2.circle(frame_copy, point, 9, (255, 255, 255), -1)
        # Draw the bboxes on the frame
        if show_bboxes:
            for bbox in bboxes[current_frame]:
                (x, y, w, h), color = bbox
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
    update_frame(show_mask=True, show_bboxes=False)

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
            if left_button_down:
                positive_points[current_frame].append(last_point)
                run_sam_frames[current_frame] = True
            elif right_button_down:
                negative_points[current_frame].append(last_point)
            print_console(current_label, label_colors, current_frame, total_frames, tam_ker_op)
            update_frame(show_mask, show_bboxes)
            set_image = True
        elif key == ord('.'):   # Next frame
            current_frame = min(total_frames - 1, current_frame + 1)
            if left_button_down:
                positive_points[current_frame].append(last_point)
                run_sam_frames[current_frame] = True
            elif right_button_down:
                negative_points[current_frame].append(last_point)
            print_console(current_label, label_colors, current_frame, total_frames, tam_ker_op)
            update_frame(show_mask, show_bboxes)
            set_image = True
        elif key == ord('r'):  # Reset mask for current label and frame
            mask = masks[current_frame]
            mask[mask == label_colors[current_label]] = 0
            bboxes[current_frame] = [bbox for bbox in bboxes[current_frame] if bbox[1] != label_colors[current_label]]
            update_frame(show_mask, show_bboxes)
        elif key == ord('p'):  # Clear all points for all frames
            positive_points = [[] for _ in range(total_frames)]
            negative_points = [[] for _ in range(total_frames)]
            run_sam_frames = [False for _ in range(total_frames)]
            update_frame(show_mask, show_bboxes)
        elif key == ord('t'):  # Clear all points for current frame
            positive_points[current_frame] = []
            negative_points[current_frame] = []
            run_sam_frames[current_frame] = False
            update_frame(show_mask, show_bboxes)
        elif key == ord('c'):  # Change label
            current_label = labels_list[(labels_list.index(current_label) + 1) % len(labels_list)]
            run_sam_frames[current_frame] = False
            update_frame(show_mask, show_bboxes)
            print_console(current_label, label_colors, current_frame, total_frames, tam_ker_op)
        elif key == ord('s') or key == ord('a'):  # Call SAM for current frame or all frames
            running_frames = [current_frame] if key == ord('s') else range(total_frames)
            for i in running_frames:
                if len(positive_points[i]) != 0 and run_sam_frames[i]:
                    print('Calling SAM for frame ', i)
                    t0 = time.time()
                    mask = sam_processing(frames[i], sam_predictor, positive_points[i], negative_points[i], set_image=set_image)
                    print(f'  Done! Segmentation done in {time.time() - t0:.2f} seconds.')
                    run_sam_frames[i] = False
                    # Mask opening operation
                    if tam_ker_op > 0:
                        mask = cv2.morphologyEx(np.array(mask, dtype=np.uint8), cv2.MORPH_OPEN, np.ones((tam_ker_op, tam_ker_op), np.uint8))
                    # Save the mask with color
                    masks[i][mask != 0] = label_colors[current_label]
                    # Save the bbox ((x, y, w, h), color)
                    bboxes[i].append((cv2.boundingRect(np.array(mask, dtype=np.uint8)), label_colors[current_label]))
                    # Update the frame
                    update_frame(show_mask, show_bboxes)
                else:
                    print('No positive points to segment in frame ', i)
            set_image = False
        elif key == ord('v'):   # Show/hide mask
            show_mask = not show_mask
            update_frame(show_mask, show_bboxes)
        elif key == ord('b'):   # Show/hide bboxes
            show_bboxes = not show_bboxes
            update_frame(show_mask, show_bboxes)
        elif key == ord('k'):   # Create backup
            save_masks(backup_folder, masks, is_backup=True)
        elif key == ord('+'):   # Increase opening kernel size
            tam_ker_op += 1
            print_console(current_label, label_colors, current_frame, total_frames, tam_ker_op)
        elif key == ord('-'):   # Decrease opening kernel size
            tam_ker_op = max(0, tam_ker_op - 1)
            print_console(current_label, label_colors, current_frame, total_frames, tam_ker_op)

    cv2.destroyAllWindows()
    return masks

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


##########################################################################
# Main
##########################################################################
if __name__ == '__main__':
    # 0. Parse input arguments:
    args = parse_arguments()

    # 1. Read video:
    frames = frames_from_video(args.input_video)
    if frames is None:
        print(f'Error opening video {args.input_video}')
        exit(1)
    print(f'Video {args.input_video} opened with {len(frames)} frames.')

    # 2. Read labels (dict where key is label name and value is RGB color)
    label_colors = labels_colors_from_file(args.label_colors)

    # 3. Load masks from folder:
    if args.load_folder[-1] != '/': args.load_folder += '/'
    answer = input(f'Do you want to load masks from {args.load_folder}? [y/n]: ')
    if answer.lower() == 'y':
        loaded_masks = load_masks(args.load_folder)
    else:
        loaded_masks = None

    if loaded_masks and len(frames) != len(loaded_masks):
        print('Error: Number of frames in video and masks do not match.')
        exit(1)

    # 4. Initialize the SAM model:
    make_slow_imports()
    sam_predictor = init_SAM_predictor()

    # 5. Navigate through frames and click the points
    segm_masks = navigate_frames(frames, label_colors, sam_predictor, args.backup_folder, loaded_masks)

    # 6. Ask for saving the masks
    answer = input('Do you want to save the masks? (Process will overwrite files in \'' + args.output_folder + '\') [y/n]: ')
    if answer.lower() == 'y':
        save_masks(args.output_folder, segm_masks)

    exit(0)
    
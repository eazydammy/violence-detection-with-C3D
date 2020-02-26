import cv2
import numpy as np
from inference import Network

model = "../model/model37.xml"

video_path = "../video/vi22.avi" #non21.avi #vi22.avi

def prepare_frames(video_path):
    '''
    This function extracts all the image frames from the given video stream or file. It applies the necessary
    transformations including resizing, cropping and normalization. It returns a tensor of all frames
    '''
    cap = cv2.VideoCapture(video_path)
    cap.open(video_path)

    all_frames = np.empty((1,3,112,112))

    while cap.isOpened():
        flag, frame = cap.read()
        
        if not flag:
            break
        
        key_pressed = cv2.waitKey(1) #(1000/16 = 62.5fps)

        frame = frame/255

        frame = cv2.resize(frame, (171, 128))
        frame = crop_center(frame, 112, 112)
        frame = frame.transpose((2,0,1))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = normalize(frame, mean, std)

        frame = frame.reshape(1,3,112,112)

        all_frames = np.append(all_frames, frame, axis=0) # (num, 3, 112, 112)
        
        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture, and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
    return all_frames

def prepare_stacks(all_frames):

    start_cut = []
    num_frames = all_frames.shape[0]

    clip_num = num_frames // 8

    for i in range(int(clip_num) - 2): #trim last incomplete stack
        start_cut.append(8*i+1)

    all_stacks = np.empty((1,16,3,112,112))

    for clip_start in start_cut:
        temp_stack = all_frames[clip_start:clip_start+16]
        temp_stack = temp_stack.reshape(1,16,3,112,112)

        all_stacks = np.append(all_stacks, temp_stack, axis=0)

    #stack is ((1, 16, 3, h-112, w-112))

    all_stacks = all_stacks.transpose((0,2,1,3,4))
    #stack now is ((1,3,16,w-112,h-112))

    return all_stacks

def draw_indicator(frame, result, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= 0.5:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
    return frame

def perform_inference(model, all_stacks):
    
    # Initialize the Inference Engine
    plugin = Network()
    
    # Load the network model into the IE
    plugin.load_model(model, "CPU")

    for stack in all_stacks:

        stack = stack.reshape(1,3,16,112,112)
        
        # perform inference on each stack
        result = plugin.sync_inference(stack)

        print(result)

def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx, :]

def normalize(frame, mean, std):
    output = np.zeros(frame.shape)
    for channel in range(3):
        output[channel] = (frame[channel] - mean[channel]) / std[channel]

    return output

def main():
    all_frames = prepare_frames(video_path)
    all_stacks = prepare_stacks(all_frames)
    perform_inference(model, all_stacks)

if __name__ == "__main__":
    main()

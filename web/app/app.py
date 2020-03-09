#!/usr/bin/env python3
import sys
import cv2
import string
import random
import argparse
import numpy as np
from inference import Network

model_path = "../model/model37.xml"

video_path = "../video/vi16.avi"
#non21.avi non25.avi
#vi22.avi vi16.avi

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The path to the model XML file"
    v_desc = "The path to the video file"

    # -- Create the arguments
    parser.add_argument("-m", help=m_desc, default="../model/model37.xml")
    parser.add_argument("-v", help=v_desc, default="../video/vi22.avi")
    args = parser.parse_args()

    return args


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
        
        #key_pressed = cv2.waitKey(60)

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
        # if key_pressed == 27:
        #     break

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
    # print("Number of frames", num_frames)
    # print("Number of stacks", all_stacks.shape)

    return all_stacks

def draw_indicator(video_path, results):
    '''
    Draw bounding boxes onto the frame.
    '''

    # perform trick to stretch out overlapping stacked results
    aug_results = []
    for i, result in enumerate(results):
        if i > 0:
            aug_results = aug_results + ([result] * 16)
        else:
            aug_results = aug_results + ([result] * 8)

    cap = cv2.VideoCapture(video_path)
    cap.open(video_path)

    true_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    true_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    true_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    #right pad (with stretch) in case of missing frame results
    while len(aug_results) < true_num_frames:
        aug_results.append(aug_results[-1])
    
    letters = string.ascii_lowercase
    file_name = ''.join(random.choice(letters) for i in range(10)) + ".mp4"

    out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (true_width,true_height))


    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (10, 33) 
    fontScale = 0.5
    text_color = (255, 255, 255) 
    thickness = 1
   
    while cap.isOpened():
        flag, frame = cap.read()
        
        if not flag:
            break

        frame_is_violent = aug_results.pop(0)
        
        colors = [(0,255,0), (0,0,255)]
        color = colors[frame_is_violent]

        alerts = ['','Violent!']
        alert = alerts[frame_is_violent]
            
        out_frame = cv2.rectangle(frame, (10, 10), (70, 45), color, cv2.FILLED)
        out_frame = cv2.putText(out_frame, alert, org, font,  
                   fontScale, text_color, thickness, cv2.LINE_AA)
        #_, out_frame = cv2.imencode('.jpg', out_frame)
        
        # Send frame to the ffmpeg server
        #sys.stdout.buffer.write(out_frame)
        #sys.stdout.flush()

        # Write out the frame
        out.write(out_frame)
    
    # Release the capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    return file_name

def perform_inference(model, all_stacks):
    
    # Initialize the Inference Engine
    plugin = Network()
    
    # Load the network model into the IE
    plugin.load_model(model_path, "CPU")

    results = []

    for stack in all_stacks:

        stack = stack.reshape(1,3,16,112,112)
        
        # perform inference on each stack
        result = plugin.sync_inference(stack)

        results.append(np.argmax(result))

    return results

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
    args = get_args()
    all_frames = prepare_frames(args.v)
    all_stacks = prepare_stacks(all_frames)
    results = perform_inference(args.m, all_stacks)
    name = draw_indicator(args.v, results)
    return str(name)

if __name__ == "__main__":
    main()

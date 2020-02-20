import os
from PIL import Image
import numpy as np
import h5py
import torchvision.transforms as transforms

def create_frame_stack(path, name, length, height, width):
        if name == "train":
                number = 2838
        else:
                number = 944
        video = np.zeros((number, length, 3, height, width), dtype=np.float32)
        label = np.zeros((number,2), dtype=np.int64) 
        #label = np.zeros(number, dtype=np.int64) #assuming one-hot function will be called on this

        with open(path, 'r') as f:
                num = 0
                for line in f:
                        line = line.strip()
                        dir_name, frame, lbl = line.split()
                        images = np.zeros((length, 3, height, width), dtype=np.float32)
                        
                        for i in range(int(frame),int(frame)+length):
                                img_name = str(i) + '.jpg'
                                img = Image.open(os.path.join(dir_name,img_name))

                                img = transform(img)
                                img = img.numpy()
        
                                images[i-int(frame)] = img

                        video[num] = images 

                        if int(lbl) == 0:
                                label[num] = [1,0]
                        else:
                                label[num] = [0,1]
                        num = num + 1   

        video = np.asarray(video,dtype=np.float32)
        label = np.asarray(label,dtype=np.int64)

        filename = str(name)+'.h5'
        f = h5py.File(filename,'w')
        f['video'] = video                
        f['label'] = label         
        f.close()

# Number of frames per video clip
NUM_FRAMES_PER_CLIP = 16

# Images will be cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 112

transform = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

if __name__ == "__main__":
        create_frame_stack('./train.txt', 'train', NUM_FRAMES_PER_CLIP,CROP_SIZE,CROP_SIZE)
        create_frame_stack('./test.txt', 'test', NUM_FRAMES_PER_CLIP,CROP_SIZE,CROP_SIZE)
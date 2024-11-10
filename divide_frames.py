import glob
import os
import torchvision

#Dividing videos into frames
def divide_video_into_frame(input_dir):
    for dataset in ['train', 'val', 'test']:
        videos = glob.glob(os.path.join(input_dir, dataset, '*_color.mp4'))
        for video_file in videos:
            frames, _, _ = torchvision.io.read_video(video_file, pts_unit='sec')
            with open(video_file.replace('color.mp4', 'nframes'), 'w') as of:
                of.write(f'{frames.size(0)}\n')

input_directory = "C:\\Users\\emirh\\Downloads"

divide_video_into_frame(input_directory)
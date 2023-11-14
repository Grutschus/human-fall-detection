"""This script processes either a path to a video file or a folder containing videos. It produces a new folder containing videos derived from the 
trimmed input video(s), where each video is segmented into equal-length parts as specified by the user. This code has been adapted from the example 
demonstrated in a YouTube tutorial, available at the following link: https://www.youtube.com/watch?v=SGsJc1K5xj8.
"""

import os
import ffmpeg
import pathlib import path
import argparse


def trim_video(
    input_path: Path | str,
    output_path: Path | str,
    trim_duration: int 
) -> str:

    """Transforms input video(s) into equal-length smaller trimmed video(s) as specified by the user.

    Args:
        input_path: Path | str: path of the input video(s).
        output_folder: Path | str: path of output video(s).
        trim_duration: int: the length of the trimed video.
    Returns:
        str: message indicate the completion of  .
    """
    
    #return  the path(s) of the video(s)  in the input folder
    video_paths = list(Path(input_path).glob("*.avi")) 
    
    for video_path in video_paths:
        # return the video's properties
        video_probe = ffmpeg.probe(video_path)

        # return the video's duration
        video_duration = video_probe.get("format", {}).get("duration", None)
        print(video_duration)
        # return the video's stream 
        input_stream = ffmpeg.input(video_path)

        # calculate the number of trimmed videos to be generated
        
        trimmed_videos_number= np.ceil(float(video_duration)/trim_duration).astype(int)
        
        for n_trim in range(trimmed_videos_number):

            # calculate start and end time of the trim
            trim_start_time=trim_duration*n_trim
            
            if n_trim == trimmed_videos_number-1:
                trim_end_time= np.ceil(float(video_duration)).astype(int)
            else:
                trim_end_time=trim_duration*(n_trim+1)
                
            # trim the video from trim_start_time to trim_end_time  
            video = input_stream.trim(start=trim_start_time, end=trim_end_time).setpts("PTS-STARTPTS")
            
        # write the trimed video to the output folder 

            #set up the time format of output file 
            format_str="{:0"+str(len(str(np.ceil(float(video_duration)).astype(int))))+"d}" #e.g {:03d}


            #return the video file name
            video_file_name= os.path.splitext(os.path.basename(video_path))[0]
            
            #return the video file extention
            video_file_extension= os.path.splitext(os.path.basename(video_path))[1]

            # return the output video path
            output_file_path= output_path+'/'+video_file_name+str('_')+str(format_str.format(trim_start_time))+str('_')+str(format_str.format(trim_end_time))+video_file_extension

            # write the video to the output path
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            output = ffmpeg.output(video, output_file_path)
            output.run()
           
            
            
    print('Videos trimming completed successfully.')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converts  videos stream to equal-length smaller trimmed video(s) as specified by the user"
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        help="path of the input video(s)",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="path of output video(s)",
    )
    parser.add_argument(
        "--trim_duration",
        default=10,
        type=int,
        help="the length of the trimed video",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    transform(
        input_path=args.output_path,
        output_path=args.output_path,
        trim_duration=args.trim_duration,
    )
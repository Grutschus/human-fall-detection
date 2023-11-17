"""This script processes input, which can be either the path to a video file or a folder containing videos. The output is an 
annotation file that includes the paths of the processed videos and the corresponding classes associated with each video.
"""

import os
import ffmpeg
import pandas as pd
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def trimmed_videos_annotation(
    annotation_file_path: Path | str,
    input_videos_folder: Path | str,
    output_folder: Path| str 
) -> None:
    """Generate an multiclass annotation file for videos.
    Args:
        annotation_file_path: Path | str: Path to the annotation files of the videos before trimming .
        input_videos_folder: Path | str: Path to the trimmed video(s) to be annotated.
        output_folder: Path | str: Path to the folder where the annotation file should be saved.
    Returns:
        None
    """
    
    #return  the path(s) of the video(s)  in the videos_folder
    video_paths = list(Path(input_videos_folder).glob("*.avi")) 
    
    #return the annotation file data as pandas dataframe
    annotation_file = pd.read_csv(annotation_file_path)
    
    # initilize annotation file for the trimmed videos
    trimmed_videos_annotation_file=[]
    
    #Generate the annotation file for the input videos
    for video_path in video_paths:
        
        # return the original video name before trimming 
        video_name= re.sub('_\d+', '', video_path.stem) 
        #(e.g video_path='OutputVideos\Fall27_Cam4_0_10.avi'-> video_name='Fall27_Cam4.avi')
        
        #return the trim stat and end time of the video
        [trim_start_time,trim_end_time]= [float(time) for time in re.findall('\d+', video_path.stem)[-2:]]
        
        # return start and end of the falling and lying events from original videos annotation file
        fall_start_time= float(annotationfile.loc[annotationfile['video_name'] == video_name]['fall_start'])
        fall_end_time= float(annotationfile.loc[annotationfile['video_name'] == video_name]['fall_end'])
        lying_start_time= float(annotationfile.loc[annotationfile['video_name'] == video_name]['lying_start'])
        lying_end_time= float(annotationfile.loc[annotationfile['video_name'] == video_name]['lying_end'])
        
        # set the multiclass for each trimmed video
        if trim_start_time < fall_start_time:
            if trim_end_time < fall_start_time:
                action_class='0'
            elif  fall_start_time <= trim_end_time< fall_end_time:
                action_class='0 1'
            elif  lying_start_time<= trim_end_time < lying_end_time:
                action_class='0 1 2'
            else:
                action_class='0 1 2'
        elif  fall_start_time <= trim_start_time< fall_end_time:
            if  fall_start_time <= trim_end_time< fall_end_time:
                action_class='1'
            elif  lying_start_time<= trim_end_time < lying_end_time:
                action_class='1 2'
            else:
                action_class='0 1 2'
        elif  lying_start_time<= trim_start_time < lying_end_time:
            if  lying_start_time<= trim_end_time < lying_end_time:
                action_class='2'
            else:
                action_class='0 2'
        else:
            action_class='0'
        
        # append the video path and action class to the annotation dictionary
        trimmed_videos_annotation_file.append({'video_path':str(video_path),'action_class':action_class})
     
    # save the annotation dictionary as CSV file in output_folder
    output_file_path= Path(output_folder+'/trimmed_videos_annotation_file.csv')
    output_file_path.parent.mkdir(parents=True, exist_ok=True)  
    pd.DataFrame(trimmed_videos_annotation_file).to_csv(output_file_path) 
    
    
    logger.info("Annotation file for trimmed Videos completed successfully.")
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an multiclass annotation file for videos."
    )
    parser.add_argument(
        "--annotation_file_path",
        default="data/Fall_Simulation_Data/videos/annotations",
        type=Path,
        help="Path to the annotation files of the videos before trimming",
    )
    parser.add_argument(
        "--input_videos_folder",
        default="data/Fall_Simulation_Data/trimmed_videos/",
        type=Path,
        help="Path to the trimmed video(s) to be annotated.",
    )
    parser.add_argument(
        "--output_folder",
        default='',
        type=Path,
        help="Path to the folder where the annotation file should be saved.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set the logging level",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger.setLevel(args.log_level)
    trimmed_videos_annotation(
        annotation_file_path=args.annotation_file_path,
        input_videos_folder=args.input_videos_folder,
        output_folder=args.output_folder,
    )
  
 
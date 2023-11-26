from pathlib import Path
import ffmpeg
import argparse
import numpy as np
import logging
import pandas as pd
import math
import numpy as np

class Sampler:
    def __init__(self, input_path, ann_file_path):
        """
        The Sampler class contains sampling methods for the video input data.
        :param input_path: file path for the input videos
        :param ann_file_path: file path for annotation file belonging to the video input data
        """
        self.input = input_path
        self.ann_file = ann_file_path

    
    def stratifiedSampling(self, output_path, trim_len=10, n_fall_samples=3, samples_per_min=1):
        """
        Stratified sampling only samples from videos which contain all three 
        actions (ADL, falling, lying). From these videos, a base rate of fall 
        samples will be sampled uniformly on the fall interval (time where fall happens). 
        ADL and lying activities are sampled based on their ratio of duration based 
        on sample rate.
        :param trim_len: sample video length in seconds (defaults to 10) 
        :param fall_samples: amount of fall samples to collect from videos where 
        falls occur (defaults to 3) 
        :param samples_per_min: amount of samples per minute from ADL and lying activities 
        (defaults to 1)
        :param output_path: filepath for sample outputs
        """
        
        # Read annotation file to dataframe
        df = pd.read_csv(self.ann_file)
        
        # Filter for only fall videos
        df = df[df['category'] == "Fall"] 
        
        # Reset index after filtering
        df = df.reset_index()
        
        sample_list = []

        for i in df.index:
            
            # Get timestamps
            fall_start = float(df.loc[i, 'fall_start'])
            fall_end = float(df.loc[i, 'fall_end'])
            lying_start = float(df.loc[i, 'lying_start'])
            lying_end = float(df.loc[i, 'lying_end'])
            video_end = float(df.loc[i, 'length'])
            
            # Calculate action durations
            ADL1_time = fall_start
            ADL2_time = video_end - lying_end
            ADL_time = ADL1_time + ADL2_time
            fall_time = fall_end - fall_start
            lying_time = lying_end - lying_start
            
            # Calculate number of samples for ADL and lying activities
            n_samples = round((ADL_time + lying_time) / 60) * samples_per_min
            n_ADL_samples = round(n_samples * (ADL_time / (ADL_time + lying_time)))
            n_lying_samples = n_samples - n_ADL_samples
            n_ADL1_samples = round((ADL1_time/ADL_time)*n_ADL_samples)
            n_ADL2_samples = n_ADL_samples - n_ADL1_samples
            
            # Sample uniformly on the intervals
            ADL1_samples = np.random.uniform(0, ADL1_time, n_ADL1_samples)
            ADL2_samples = np.random.uniform(lying_end, video_end, n_ADL2_samples)
            fall_samples = np.random.uniform(fall_start, fall_end, n_fall_samples)
            lying_samples = np.random.uniform(lying_start, lying_end, n_lying_samples)
            
            # Create sample list [video path, [sample timestamps]]
            sample_list.append([
                df.loc[i, "video_path"], 
                np.concatenate((ADL1_samples, 
                               ADL2_samples, 
                               fall_samples, 
                               lying_samples)).tolist()
            ])
            
        # Generate samples
        self.outputSamples(sample_list, output_path)
    
    
    def outputSamples(self, sample_list, output_path, trim_len=10):
        """
        Utility function for trimming input videos and outputting them
        to the given output path (generating samples).
        :param sample_list: a list containing the video name and a list 
        of sample start timestamps, e.g. 
        [data/Fall_Simulation_Data/videos/Fall30_Cam3.avi, 
         [7.16443459, 15.836356, 104.36721318, 26.32500079]
         ]
        :param output_path: filepath for sample outputs
        :param trim_len: sample video length in seconds (defaults to 10) 
        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        #TODO: remove
        count = 0  
        
        for sample in sample_list:
            
            # TODO: remove
            if count > 0:
                break
            count += 1
            
             # Create path
            path = Path("../" + sample[0])
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Store sample timestamps
            timestamps = sample[1]
            print(f'Timestamps: {timestamps}')
            
            # Get video data
            video_probe = ffmpeg.probe(path)
            video_duration = video_probe.get("format", {}).get("duration", None)
            logger.debug(f"Video duration: {video_duration}")
            input_stream = ffmpeg.input(path)
            
            # Output samples
            for t in timestamps:
                
                # Trim video
                video = input_stream.trim(start=t, end=t+trim_len).setpts(
                    "PTS-STARTPTS"
                )
                
                # Create output path
                output_file_path = output_path.joinpath(
                    Path(sample[0]).stem + f"_{round(t, 3)}_{round(t+trim_len, 3)}" + ".mp4"
                )  # e.g. output_path/ADL1_Cam2_20_30.avi

                # Output
                output = ffmpeg.output(video, output_file_path.as_posix(), f="mp4")
                output.run()
                
        logger.info("Videos trimming completed successfully.")
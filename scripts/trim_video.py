"""This script processes either a path to a video file or a folder containing videos. It produces a new folder containing videos derived from the
trimmed input video(s), where each video is segmented into equal-length parts as specified by the user. This code has been adapted from the example
demonstrated in a YouTube tutorial, available at the following link: https://www.youtube.com/watch?v=SGsJc1K5xj8.
"""


import ffmpeg
from pathlib import Path
import argparse
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def trim_video(
    input_path: Path | str, output_path: Path | str, trim_duration: int
) -> None:
    """Transforms input video(s) into equal-length smaller trimmed video(s) as specified by the user.

    Args:
        input_path: Path | str: path of the input video(s).
        output_folder: Path | str: path of output video(s).
        trim_duration: int: the length of the trimed video.
    Returns:
        str: message indicate the completion of  .
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # return  the path(s) of the video(s)  in the input folder
    if input_path.is_file():
        video_paths = [input_path]
    else:
        video_paths = list(input_path.glob("*.avi"))

    for video_path in video_paths:
        # return the video's properties
        video_probe = ffmpeg.probe(video_path)

        # return the video's duration
        video_duration = video_probe.get("format", {}).get("duration", None)
        logger.debug(f"Video duration: {video_duration}")
        # return the video's stream
        input_stream = ffmpeg.input(video_path)

        # calculate the number of trimmed videos to be generated
        trimmed_videos_number = np.ceil(float(video_duration) / trim_duration).astype(
            int
        )

        for n_trim in range(trimmed_videos_number):
            # calculate start and end time of the trim
            trim_start_time = trim_duration * n_trim

            if n_trim == trimmed_videos_number - 1:
                trim_end_time = np.ceil(float(video_duration)).astype(int)
            else:
                trim_end_time = trim_duration * (n_trim + 1)

            # trim the video from trim_start_time to trim_end_time
            video = input_stream.trim(start=trim_start_time, end=trim_end_time).setpts(
                "PTS-STARTPTS"
            )

            # write the trimed video to the output folder

            # return the output video path
            output_file_path = output_path.joinpath(
                Path(video_path).stem + f"_{trim_start_time}_{trim_end_time}" + ".mp4"
            )  # e.g. output_path/ADL1_Cam2_20_30.avi

            # write the video to the output path
            output = ffmpeg.output(video, output_file_path.as_posix(), f="mp4")
            output.run()

    logger.info("Videos trimming completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converts  videos stream to equal-length smaller trimmed video(s) as specified by the user"
    )
    parser.add_argument(
        "--input_path",
        default="data/Fall_Simulation_Data/videos/Fall1_Cam1.avi",
        type=Path,
        help="path of the input video(s)",
    )
    parser.add_argument(
        "--output_path",
        default="data/Fall_Simulation_Data/trimmed_videos/",
        type=Path,
        help="path of output video(s)",
    )
    parser.add_argument(
        "--trim_duration",
        default=10,
        type=int,
        help="the length of the trimed video",
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
    trim_video(
        input_path=args.input_path,
        output_path=args.output_path,
        trim_duration=args.trim_duration,
    )

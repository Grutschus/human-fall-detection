"""This script takes the human-readable annotation CSV file we made
from hand, unpivots the columns and outputs them in an easier
to use format for downstream processing.
"""


from pathlib import Path
import pandas as pd
from typing import Optional
import re
import argparse


def transform(
    annotation_path: Path | str,
    video_path: Path | str,
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Transforms our custom annotations and videos into an easy to use
    format for downstream tasks.

    Args:
        annotation_path (Path | str): Path to our annotations.
        video_path (Path | str): Path to the folder containing all videos.
        output_path (Path | str): Path to the output CSV file. Defaults to not saving a CSV file.

    Returns:
        pd.Dataframe: The transformed annotations.
    """

    def extract_scenario_id(video_name: str) -> int:
        """Extracts the scenario ID from the video name."""
        if match := re.search(r"(ADL|Fall)(\d+)", video_name):
            return int(match.group(2))
        else:
            raise ValueError(
                f"Video name {video_name} does not match the expected format."
                "Example for correct format: ADL1_Cam1.avi"
            )

    def extract_category(video_name: str) -> str:
        """Extracts the category from the video name."""
        if match := re.search(r"(ADL|Fall)", video_name):
            return match.group(1)
        else:
            raise ValueError(
                f"Video name {video_name} does not match the expected format."
                "Example for correct format: ADL1_Cam1.avi"
            )

    # Scan the given path for all videos
    video_paths = list(Path(video_path).glob("*.avi"))
    result = pd.DataFrame(columns=["video_path"], data=video_paths)

    # Extract video metadata from video path
    result["video_name"] = result["video_path"].apply(lambda x: x.stem)
    result["category"] = result["video_name"].apply(extract_category)
    result["camera_id"] = result["video_name"].apply(lambda x: x.split("_")[-1][-1])
    result["scenario_id"] = result["video_name"].apply(extract_scenario_id)

    # Read annotation CSV
    annotation = pd.read_csv(annotation_path)

    # Merge annotation timestamps with video metadata
    annotation_timestamps = annotation[
        [
            "scenario_id",
            "category",
            "fall_start",
            "fall_end",
            "lying_start",
            "lying_end",
            "length",
        ]
    ]

    result = result.merge(
        annotation_timestamps,
        left_on=["scenario_id", "category"],
        right_on=["scenario_id", "category"],
        how="left",
    )

    # Unpivot fall_visible_cam, lying_visible_cam, ranking_cam
    annotation_camera = annotation.filter(regex=("(cam|scenario_id|category)"))
    melted = annotation_camera.melt(
        id_vars=["scenario_id", "category"], var_name="variable", value_name="value"
    )
    # Split the 'variable' column into 'camera_id' and 'field' columns
    melted[["field", "camera_id"]] = melted["variable"].str.rsplit(
        "_", n=1, expand=True
    )
    annotation_camera = melted.drop(columns=["variable"])
    annotation_camera["field"] = annotation_camera["field"].str.replace("_cam", "")

    annotation_camera = annotation_camera.pivot_table(
        index=["scenario_id", "category", "camera_id"],
        columns=["field"],
        values="value",
    ).reset_index()

    annotation_camera[["fall_visible", "lying_visible"]] = annotation_camera[
        ["fall_visible", "lying_visible"]
    ].astype(bool)

    # Merge annotation_camera with video metadata
    result = result.merge(
        annotation_camera,
        left_on=["scenario_id", "category", "camera_id"],
        right_on=["scenario_id", "category", "camera_id"],
        how="left",
    )

    if output_path is not None:
        result.to_csv(output_path, index=False)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converts the human-readable annotations into a "
        "format that is easier to use for downstream tasks."
    )
    parser.add_argument(
        "--annotation_path",
        default="data/Fall_Simulation_Data/high_quality_fall_annotations.csv",
        type=str,
        help="Path to the annotation CSV file",
    )
    parser.add_argument(
        "--video_path",
        default="data/Fall_Simulation_Data/videos",
        type=str,
        help="Path to the directory containing the videos",
    )
    parser.add_argument(
        "--output_path",
        default="data/Fall_Simulation_Data/annotations.csv",
        type=str,
        help="Path to the output CSV file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    transform(
        annotation_path=args.annotation_path,
        video_path=args.video_path,
        output_path=args.output_path,
    )

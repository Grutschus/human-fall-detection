"""This script takes the path to our converted annotation file and
splits it into a training, testing, and optionally validation dataset."""


import argparse
import logging
from math import isclose
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def split_dataset(
    annotations: Path | str | pd.DataFrame,
    output_path: Path | str | None = None,
    split: tuple[float, float, float] = (0.7, 0.2, 0.1),
    random_seed: int = 42,
    stratify: str | None = None,
    save_splits: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the dataset into training, testing, and validation datasets.

    Args:
        annotations (Path | str | pd.Dataframe): Path to the (complete) annotation CSV file.
            Or a pandas dataframe containing the annotations.
        output_path (Path | str | None): Path to the output directory. The output
            files will have the same name as the input file, with the suffixes _train,
            _test, and _val. Defaults to None and uses the same folder as the
            `annotation_path`.
        split (tuple[float, float, float]): Tuple of three floats indicating the proportions
            of the dataset to use for training, testing, and validation (in that order). Each
            value should be between 0.0 and 1.0, and the sum of the three values should be 1.0.
            Defaults to (0.7, 0.2, 0.1).
        random_seed (int): Random seed to use for splitting the dataset. Defaults to 42.
        stratify (str | None): Column name to use for stratification. Defaults to None.
        save_splits (bool): Whether to save the splits as CSV files. Defaults to False.
    """
    annotation_path = None
    if isinstance(annotations, (Path, str)):
        annotation_path = Path(annotations)
        annotations = pd.read_csv(annotations)

    assert isclose(sum(split), 1), "The sum of the split values should be 1.0"
    assert all(
        0.0 <= v <= 1.0 for v in split
    ), "All split values should be between 0.0 and 1.0"
    train_portion, test_portion, val_portion = split

    # Split the dataset
    np.random.seed(random_seed)
    train_df, test_df = train_test_split(
        annotations,
        test_size=test_portion + val_portion,
        random_state=random_seed,
        stratify=annotations[stratify] if stratify else None,
    )
    if val_portion > 0.0:
        test_df, val_df = train_test_split(
            test_df,
            test_size=val_portion / (test_portion + val_portion),
            random_state=random_seed,
            stratify=test_df[stratify] if stratify else None,
        )
    else:
        val_df = pd.DataFrame()

    # Save the splits
    if save_splits or __name__ == "__main__":
        if annotation_path is None:
            file_stem = "annotations"
            if output_path is None:
                raise ValueError(
                    "If the annotations are given as a dataframe, "
                    "then the output path must be specified "
                    "to save the datasets."
                )
            output_path = Path(output_path)
        else:
            output_path = Path(output_path) if output_path else annotation_path.parent
            file_stem = annotation_path.stem

        if len(train_df) != 0:
            train_df.to_csv(output_path / f"{file_stem}_train.csv", index=False)
        if len(test_df) != 0:
            test_df.to_csv(output_path / f"{file_stem}_test.csv", index=False)
        if len(val_df) != 0:
            val_df.to_csv(output_path / f"{file_stem}_val.csv", index=False)

    return train_df, test_df, val_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Splits the dataset into training, testing, and optionally validation datasets."
    )
    parser.add_argument(
        "--annotation_path",
        default="data/Fall_Simulation_Data/annotations.csv",
        type=str,
        help="Path to the (complete) annotation CSV file",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="Path to the output directory. "
        "The output files will have the same name as the input "
        "file, with the suffixes _train, _test, and _val.",
    )

    class TupleAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            try:
                values = tuple(map(float, values.split(",")))
                if not all(0.0 <= v <= 1.0 for v in values):
                    raise ValueError
                if not sum(values) == 1.0:
                    raise ValueError
                if not len(values) == 3:
                    raise ValueError
                if not all(v > 0.0 for v in values[:-1]):
                    raise ValueError
            except ValueError:
                raise argparse.ArgumentError(
                    self,
                    "Invalid tuple of floats. Each value should be between 0.0 and 1.0. "
                    "And the sum of all values should be 1.0. "
                    "The tuple should have three values. "
                    "The values for training and testing should be greater than 0.0.",
                )
            setattr(namespace, self.dest, values)

    parser.add_argument(
        "--split",
        default=(0.7, 0.2, 0.1),
        type=str,
        action=TupleAction,
        help="Tuple of three floats indicating the proportions of the dataset "
        "to use for training, testing, and validation (in that order). Each "
        "value should be between 0.0 and 1.0, and the sum of the three values "
        "should be 1.0.",
    )

    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="Random seed to use for splitting the dataset",
    )

    parser.add_argument(
        "--stratify",
        default=None,
        type=str,
        help="Column name to use for stratification",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.debug(args)
    split_dataset(
        annotations=args.annotation_path,
        output_path=args.output_path,
        split=args.split,
        random_seed=args.random_seed,
        stratify=args.stratify,
    )

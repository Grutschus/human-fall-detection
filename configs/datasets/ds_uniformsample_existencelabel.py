custom_imports = dict(imports="datasets", allow_failed_imports=False)
type = "HighQualityFallDataset"
sampling_strategy = dict(type="UniformSampling", clip_len=10)
label_strategy = dict(
    type="ExistenceLabel",
    label_description=dict(
        names=["fall", "lying", "other"],
        start_timestamp_names=["fall_start", "lying_start"],
        end_timestamp_names=["fall_end", "lying_end"],
        visible_names=["fall_visible", "lying_visible"],
        other_class=2,
    ),
)
ann_file = "tests/test_data/test_annotation.csv"
pipeline = []  # type: ignore
multi_class = True
num_classes = 3
test_mode = True

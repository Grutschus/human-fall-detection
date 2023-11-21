from datasets.transforms.label_strategy import HQFD_LABEL_DESCRIPTION

custom_imports = dict(imports="datasets", allow_failed_imports=False)
type = "HighQualityFallDataset"
sampling_strategy = dict(type="UniformSampling", clip_len=10)
label_strategy = dict(type="ExistenceLabel", label_description=HQFD_LABEL_DESCRIPTION)
ann_file = "tests/test_data/test_annotation.csv"
pipeline = [
    dict(type="DecordInit"),
    dict(type="ClipVideo"),
    dict(type="SampleFrames", clip_len=16, frame_interval=4, num_clips=1),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="RandomResizedCrop"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]  # type: ignore
multiclass = True
num_classes = 3
test_mode = True

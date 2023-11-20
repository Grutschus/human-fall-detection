from datasets.transforms.label_strategy import HQFD_LABEL_DESCRIPTION, ExistenceLabel  # noqa
from datasets.transforms.sampling_strategy import UniformSampling  # noqa

custom_imports = dict(imports="datasets", allow_failed_imports=False)
type = "HighQualityFallDataset"
sampling_strategy = UniformSampling(clip_len=10)
label_strategy = ExistenceLabel(HQFD_LABEL_DESCRIPTION)
ann_file = "tests/test_data/test_annotation.csv"
pipeline = []  # type: ignore
multiclass = True
num_classes = 3
test_mode = True

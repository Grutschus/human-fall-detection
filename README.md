# human-fall-detection

[Insert Project description]

## Contribution

### Development Setup

Start by cloning the repo.

```bash
git clone https://github.com/Grutschus/human-fall-detection.git --recurse-submodules
```

#### Python Environment

Setup a Python environment. We recommend using [miniconda](https://docs.conda.io/projects/miniconda/en/latest/#).

```bash
conda env create -f environment.yml
conda activate human-fall-detection
```

#### MMAction2 framework

Unfortunately, the MMAction2 framework cannot be bundled into the conda environment.
Thus, install the MMAction2 framework by following [these](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html#best-practices) instructions.
Most of the steps are already taken care of. You should only need to run the following:

```bash
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose
cd mmaction2
pip install -v -e .
```

Verify the successful installation of the framework by running the steps ([further instructions here](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html#verify-the-installation))

```bash
cd mmaction2

mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .

python demo/demo.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
    tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
    demo/demo.mp4 tools/data/kinetics/label_map_k400.txt \
    --device "cuda:0"
```

Make sure to replace the device flag if you are running on a different machine (e.g. "mps" for Mac).

#### DVC

We use [DVC](https://dvc.org/) to store and version our dataset.
To run our models and the latest experiments it is not strictly necessary to setup DVC.
You can download the raw dataset here: [KU Leuven Datasets](https://iiw.kuleuven.be/onderzoek/advise/datasets). Afterwards you can run all of our preprocessing scripts to generate the data we have used for training and evaluation.

However, if you want to directly access the data we use, follow these steps.

**Prerequisites:**

1. DVC (+ optionally the VSCode extenstion) has to be installed. Follow these instructions: [DVC Installation](https://dvc.org/doc/install#installation)
2. Access Key + Secret for our S3 Bucket (not currently public yet)

**DVC Setup:**

To authenticate with our bucket, add the credentials to your **local** DVC config:

```bash
dvc remote modify --local storage access_key_id [ACCESS_KEY_ID]
dvc remote modify --local storage secret_access_key [SECRET_ACCESS_KEY]
```

Alternatively, you can set the environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in your terminal.
For more information please refer to the [DVC docs](https://dvc.org/doc/user-guide/data-management/remote-storage/amazon-s3#custom-authentication).

#### Pre-commit hooks

Although not strictly necessary, it does make sense to locally install the pre-commit hooks.

```bash
pre-commit install
```

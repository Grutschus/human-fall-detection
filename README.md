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

#### Pre-commit hooks

Although not strictly necessary, it does make sense to locally install the pre-commit hooks.

```bash
pre-commit install
```

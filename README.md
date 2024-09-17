
# ComPepper
## Step 1: Set up the dataset
You can download the dataset from https://www.ipb.uni-bonn.de/html/projects/shape_completion/shape_completion_challenge.zip. You should put the files in the ```sweetpepper_dataset```.


## Step 2: TCoRe_sweet_try Setup

First, create and activate a new Conda environment:
```bash
conda create --name tcore python=3.9
conda activate tcore
cd TCoRe_sweet_try
```

Install required system packages:
```bash
sudo apt install build-essential python3-dev libopenblas-dev
```

Install Python dependencies:
```bash
pip3 install -r requirements.txt
pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"
pip3 install -U -e .
```

To train the model:
```bash
python tcore/scripts/train_augumented.py --ckpt_path TCoRe_sweet_try/checkpoints/pretrained_model.ckpt
```

To generate test set point clouds:
```bash
python tcore/scripts/test.py
```

## Step 3: PointFT Setup

Create and set up another Conda environment:
```bash
conda create -n pointft python=3.8.12
conda activate pointft
cd PointFT
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Install Chamfer Distance module for PyTorch:
```bash
cd utils/ChamferDistancePytorch/chamfer3D
python setup.py install
```

To train the model:
```bash
python train_sweetpepper.py -c PointFT/cfgs/PointFT.yaml
```

To generate test set point clouds:
```bash
python test_sp_generate.py -c PointFT/cfgs/PointFT.yaml
```

## Step 4: Merging Point Clouds

To merge the point clouds from both setups:
```bash
python ./merge_pcd.py
```

Ensure all paths and environment names are correctly set as per your system configuration.




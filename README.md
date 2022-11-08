# meta-RL

Code for our GameSec 2022 paper: Robust Moving Target Defense against Unknown Attacks: A Meta-Reinforcement Learning Approach

## setup environment

Please run the following command to install required packages

```
# requirements
pip install -r requirements.txt
```

## Usage

```
# Default settings: pre-train 10000 steps, adpate 100 steps, test 1000 steps. 
You can change these settings in main.py and select coresponding environment from experiments.py
```

### Dataset:

```
We conducted numerical simulations using the real data from the National Vulnerability Database (NVD), which is stored at /experiments/nvdcve-1.1-2021.json. For more details, please visit: https://nvd.nist.gov/
```

### Reproduce experiments:
```
# Change the model dir to your own experiment
python3 main.py
```

## Citation
If you find our work useful in your research, please consider citing:
```


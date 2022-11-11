# meta-RL for MTD

Code for our GameSec 2022 paper: Robust Moving Target Defense against Unknown Attacks: A Meta-Reinforcement Learning Approach

## setup environment

Please run the following command to install required packages

```
# requirements
pip install -r requirements.txt
```

## Usage

```
# Data Processing: Extract nvdcve-1.1-2021.json from /experiments/nvdcve-1.1-2021.zip and use NVD_parser to parse .json files
# Default settings: pre-train 10000 steps, adpate 100 steps, test 1000 steps. 
You can change these settings in main.py and select coresponding environment from experiments.py
You can also directly test/adapt ready-to-go policies stored in /experiments
```

### Dataset:

```
We conducted numerical simulations using the real data from the National Vulnerability Database (NVD), which is stored at /experiments/nvdcve-1.1-2021.json. For more details, please visit: https://nvd.nist.gov/
```

### Reproduce experiment results:
```
# Change the model dir to your own experiment
python3 main.py
```

## Reference

NVD_parser is written by Tom Roginsky during our previous research on moving target defense supported by an NSF REU grant, the original code is avalible at https://github.com/MovingTargetDefenseCapstone/semester1


## Citation
If you find our work useful in your research, please consider citing:
```


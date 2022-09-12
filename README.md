# meta-RL

Moving target defense (MTD) provides a systematic framework to achieving proactive defense in the presence of advanced and stealthy attacks. To obtain robust MTD in the face of unknown attack strategies, a promising approach is to model the sequential attacker-defender interactions as a two-player Markov game, and formulate the defender's problem as finding the Stackelberg equilibrium (or a variant of it) with the defender and the leader and the attacker as the follower. To solve the game, however, existing approaches typically assume that the attacker type (including its physical, cognitive, and computational abilities and constraints) is known or is sampled from a known distribution. The former rarely holds in practice as the initial guess about the attacker type is often inaccurate, while the latter leads to suboptimal solutions even when there is no distribution shift between when the MTD policy is trained and when it is applied. On the other hand, it is often infeasible to collect enough samples covering various attack scenarios on the fly in security-sensitive domains. To address this dilemma, we propose a two-stage meta-reinforcement learning based MTD framework in this work. At the training stage, a meta-MTD policy is learned using experiences sampled from a set of possible attacks. At the test stage, the meta-policy is quickly adapted against a real attack using a small number of samples. We show that our two-stage MTD defense obtains superb performance in the face of uncertain/unknown attacker type and attack behavior.

Code is for our GameSec 2022 paper: Robust Moving Target Defense against Unknown Attacks: A Meta-Reinforcement Learning Approach

## Usage
### Dataset:
We conducted numerical simulations using the real data from the National Vulnerability Database (NVD), which is stored at /experiments/nvdcve-1.1-2021.json. For more details, please visit: https://nvd.nist.gov/

### Reproduce experiments:
meta-RL+MTD.ipynb

## Citation
If you find our work useful in your research, please consider citing:
```


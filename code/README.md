# Reinforced BEYBs Repository

The structure of this code directory is as follows.
- `report_specific`: Code for reproducing graphs and tables in the code (resource intensive, can take many hours)
- `examples`: Code for running simplified examples for reproducibility purposes

To run either, please consider the installation instructions below.


## Installation

Our code is extensively tested on the following Python
versions: 3.6, 3.7 and 3.8.
Please use any of these and proceed with the following commands.

```bash
pip install -r requirements.txt
```

This installs several packages for reinforcement learning algorithms.


## Usage

We invite you to go over the iPython notebooks in the `examples` folder.
The two notebooks in the top level are documented and can be run standalone,
provided the packages in `requirements.txt` are installed.
These are based on our custom package [dmarket_rl](https://github.com/zhy0/dmarket_rl).

We additionally provide benchmarking comparison with old repo
in the `Q_Learning_Examples`.

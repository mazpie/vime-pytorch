# vime-pytorch

This repo contains the PyTorch implementation of two Reinforcement Learning algorithms:

* PPO (Proximal Policy Optimization) ([paper](https://arxiv.org/abs/1707.06347))
* VIME-PPO (Variational Information Maximizing Exploration) ([paper](https://arxiv.org/abs/1605.09674))

The code makes use of [openai/baselines](https://github.com/openai/baselines).

The PPO implementation is mainly taken from [ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/).

The main novelty in this repository consists of the implementation of the VIME's exploration strategy using the PPO algorithm.

## Requirements

* Python 3
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
pip install -r requirements.txt
```

If you don't have mujoco installed, follow the intructions [here](https://github.com/openai/mujoco-py).

If having issues with OpenAI baselines, try:

```
# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

## Instructions

In order to run InvertedDoublePendulum-v2 with VIME, you can use the following command:

```bash
python main.py --env-name InvertedDoublePendulum-v2 --algo vime-ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --num-env-steps 1000000 --use-linear-lr-decay --no-cuda --log-dir /tmp/doublependulum/vimeppo/vimeppo-0 --seed 0 --use-proper-time-limits --eta 0.01
```
   
Instead, to run experiments with PPO, just replace `vime-ppo` with `ppo`.

## Results

For standard gym environments, I used `--eta 0.01`.

![MountainCar-v0](results/imgs/mountain_car.png)

![InvertedDoublePendulum-v2](results/imgs/double_pendulum.png)

For sparse gym environments, I used `--eta 0.0001`.

![MountainCar-v0-Sparse](results/imgs/mountain_car_continuous_sparse.png)

![HalfCheetah-v3-Sparse](results/imgs/half_cheetah_v3_sparse.png)

[the number in parenthesis represents how many experiments have been run]

#### Note:

Any gym-compatible environment can be run, but the hyperparameters have not been tested for all of them. 

However, the parameters used with the InvertedDoublePendulum-v2 example in the Instructions are, generally, good enough for other mujoco environments. 

## TODO:

* Integrate more args into the command line

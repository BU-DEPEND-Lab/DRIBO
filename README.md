# DRIBO: Robust Deep Reinforcement Learning via Multi-View Information Bottleneck

This repository is the official implementation of DRIBO. Our implementation of SAC is based on [SAC+AE](https://github.com/denisyarats/pytorch_sac_ae) by Denis Yarats.

## Installation
1. Install dm_control with [MuJoCo Pro 2.00](http://www.mujoco.org/);
    ```
    pip install dm_control
    pip install git+git://github.com/denisyarats/dmc2gym.git
    ```

2. All of the python dependencies are in the `setup.py` file. They can be installed manually or with the following command:
    ```
    pip install -e .
    ```

3. Running the natural video setting
You can download the [Kinetics
dataset](https://github.com/Showmax/kinetics-downloader) to replicate our setup.
* Grab the "arranging_flower" label from the train dataset to replace backgrounds during training. The videos are in folder `../kinetics-downloader/dataset/train/arranging_flowers/`.
    ```
    python download.py --classes 'arranging flowers'
    ```
* Download the test dataset to replace backgrounds during testing. The videos are in folder `../kinetics-downloader/dataset/test`.
    ```
    python download.py --test
    ```

## Instructions
1. To train a DRIBO agent on the `cartpole swingup` task under `the clean setting` run `./script/run_clean_bg_cartpole_im84_dim1024_no_stacked_frames.sh` from the root of this directory. The `run_clean_bg_cartpole_im84_dim1024_no_stacked_frames.sh` file contains the following command, which you can modify to try different environments / hyperparamters.

    ```
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --domain_name cartpole \
        --task_name swingup \
        --encoder_type rssm --work_dir ./clean_log \
        --action_repeat 8 --num_eval_episodes 8 \
        --pre_transform_image_size 100 --image_size 84 \
        --agent DRIBO_sac --frame_stack 1 --encoder_feature_dim 1024 --save_model  \
        --seed 0 --critic_lr 1e-5 --actor_lr 1e-5 --eval_freq 10000 --batch_size 8 --num_train_steps 890000
    ```

2. To train a DRIBO agent on the `cartpole swingup` task under `the natural video setting` run `./script/run_noisy_bg_cartpole_im84_dim1024_no_stacked_frames.sh` from the root of this directory. The `run_noisy_bg_cartpole_im84_dim1024_no_stacked_frames.sh` file contains the following command, which you can modify to try different environments / hyperparamters.

    ```
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --domain_name cartpole \
        --task_name swingup \
        --encoder_type rssm --work_dir ./log \
        --action_repeat 8 --num_eval_episodes 8 \
        --pre_transform_image_size 100 --image_size 84 --noisy_bg \
        --agent DRIBO_sac --frame_stack 1 --encoder_feature_dim 1024 --save_model  \
        --seed 0 --critic_lr 1e-5 --actor_lr 1e-5 --eval_freq 10000 --batch_size 8 --num_train_steps 890000
    ```

    The console output is available in a form:
    ```
    | train | E: 1 | S: 1000 | D: 34.7 s | R: 0.0000 | BR: 0.0000 | A_LOSS: 0.0000 | CR_LOSS: 0.0000 | MIB_LOSS: 0.0000 | skl: 0.0000 | beta: 0.0E+00
    ```
    a training entry decodes as:
    ```
    train - training episode
    E - total number of episodes
    S - total number of environment steps
    D - duration in seconds to train 1 episode
    R - episode reward
    BR - average reward of sampled batch
    A_LOSS - average loss of actor
    CR_LOSS - average loss of critic
    MIB_LOSS - average DRIBO loss
    skl - average value of symmetrized KL divergence
    beta - value of coefficient beta
    ```
    while an evaluation entry:
    ```
    | eval | S: 0 | ER: 22.1371
    ```
    which just tells the expected reward `ER` evaluating current policy after `S` steps. Note that `ER` is average evaluation performance over `num_eval_episodes` episodes (usually 8).

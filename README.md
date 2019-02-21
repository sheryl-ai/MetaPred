# MetaPred

## Overview

MetaPred is a meta-learning framework for  Clinical Risk Prediction using limited Patient Electronic Health Records. We given an example in the following figure:

<p align="center"><img src="figures/task-design.png" alt=" Illustration of the proposed learning procedure" width="350"></p>

Suppose we have multiple domains, our goal is to predict Alzheimer’s disease with few labeled patients, which give rise to a low-resource classification. The idea is to employ labeled patients from high-resource domains and design a learning to transfer framework with sources and a simulated target in meta-learning. There are four steps: (1) constructing episodes by sampling from the source domains and the simulated target domain; (2) learn the parameters of predictors in an episode-by-episode manner; (3) fine-tuning the model parameters on the genuine target domain; (4) predicting the target clinical risk. We respectively implemented Convolutional Neural Network (CNN) and Long-Shot Term Network (LSTM) Networks as base predictors. The model overview (meta-training procedure) is shown as follows:

<p align="center"><img src="figures/MetaPred.png" alt="MetaPred framework overview" width="500"></p>

The entire learning procedure can be viewed as: iteratively transfer the parameter Θ learned from source domains through utilizing it as the initialization of the parameter that needs to be updated in the target domain. The learn representation of five given disease domains are shown using the t-SNE. In detail, AD, PD, DM, AM, MCI are abbreviations of Alzheimer's Disease, Parkinson's Disease, Dementia, Amnesia and Mild Cognitive Impairment. As a patient might suffer multiple diseases, there are supposed to have some overlaps among the given domains.

<p align="center"><img src="figures/patient_vis_metapred.png" alt="Visualization of patient representation learned by MetaPred" width="350"></p>

To demonstrate the effectiveness of the proposed MetaPred in the context of domain adaptation, we compare it with the state-of-the-art meta-learning algorithm "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (MAML). The results on Alzheimer's Disease domain is presented in terms of AUC and F1-Score.

<p align="center"><img src="figures/vs_maml_ad.png" alt="Performance comparison of MetaPred and MAML on the top of Alzheimer's Disease" width="350"></p>


## Requirements
This package has the following requirements:
* An NVIDIA GPU.
* `Python 3.x`
* [TensorFlow 1.5](https://github.com/tensorflow/tensorflow)
* [Progress Bar](https://progressbar-2.readthedocs.io/en/latest/index.html)


## Usage
### How to Run
To run MetaPred on EHR data, you need to revise the learning settings in main.py and the network hyperparameters in models.py. Then run the shell script metapred.sh
```bash
bash metapred.sh
```
Our learning parameters are set as:
```bash
python main.py --method='cnn' --metatrain_iterations=10000 --meta_batch_size=32 --update_batch_size=4 --meta_lr=0.0001 --update_lr=1e-5 --num_updates=4 --n_total_batches=500000
```
or
```bash
python main.py --method='rnn' --metatrain_iterations=10000 --meta_batch_size=32 --update_batch_size=4 --meta_lr=0.0001 --update_lr=1e-5 --num_updates=4 --n_total_batches=500000
```
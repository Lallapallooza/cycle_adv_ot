#!/bin/bash

data_path=$(realpath data/)

python -m experiments.modern_office $data_path
python -m experiments.mnist_usps $data_path
python -m experiments.mnist_svhn $data_path
python -m experiments.mnist_mmnist $data_path

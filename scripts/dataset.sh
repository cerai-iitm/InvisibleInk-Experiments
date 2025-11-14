#!/bin/bash

set -exu

python3 dataset.py --dataset="mimic" --folder="./data/mimic" --embed_model="sentence-transformers/all-mpnet-base-v2" --embed
python3 dataset.py --dataset="yelp" --folder="./data/yelp" --embed_model="sentence-transformers/all-mpnet-base-v2" --embed
python3 dataset.py --dataset="tab" --folder="./data/tab" --embed_model="sentence-transformers/all-mpnet-base-v2" --embed
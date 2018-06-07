#!/bin/bash
python3 train.py --corpus_file data/2018-EI-reg-En-anger-dev.txt --model_dir test --embeddings glove/glove.twitter.27B.200d.txt --data_dir data
#python3 grid_search.py --corpus_file data/2018-EI-reg-En-anger-dev.txt --model_dir test --embeddings glove/glove.twitter.27B.200d.txt --data_dir data

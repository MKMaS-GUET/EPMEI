# Enhanced Packed Marker with Entity Information for Aspect Sentiment Triplet Extraction
This repository is an implementation of our paper "Simple Approach for Aspect Sentiment Triplet Extraction using Span-based Segment Tagging and Dual Extractors".
Create the bert folder bert_models/bert-base-uncased  from https://huggingface.co/google-bert/bert-base-uncased/tree/main

create result/ner/14lap

create result/sc/14lap

conda create -n EPMEI python=3.8.1

conda activate EPMEI

conda install -c conda-forge jsonnet

pip install allenlp==1.2.2

pip install allennlp-models==1.2.2

torch==1.7.1

pip install wandb

pip install tensorboardX 

pip install tqdm

pip install seqeval==1.2.2

cd EPMEI ##Go to the model folder

pip install --editable ./transformers

We use data_prepro_getGraph.py to process data/ASTE-Data-V2-EMNLP2020/ , so that we get the syntactic adjacency matrix of each sentence, and we put the processed data in ASTE-Data-V2-EMNLP2020_pro

First run run_ner.py to predict the entity and use test3.py to process the prediction results into json format.

Then run run_sc.py for sentiment classification to get the final results.

You can use average.py to find the average of 5 experimental results.

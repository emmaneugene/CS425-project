# BERT Emotional Classifier model 

Here is a description of the files in this directory:
- bert_files/data/combined_emotional_data.csv: the data file used for training
- bert_files/output/*: contains the final model files for the emotional classifier 
- bert_training.ipynb: the jupyter notebook used for training. This can run as is as training was done locally
- emotional_classifier.py: contains functions to call the trained model. called by run_model.py to integrate both this BERT emotional classifier and the BART dialogue generator
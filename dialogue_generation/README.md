# BART Dialogue Generation model 

Here is a description of the files in this directory:
- bart_files/epoch=2-step=11418.ckpt: the final model file for the dialogue generator 
- bart.ipynb: the jupyter notebook used for training. Training was done on google colab 
- bart.py: contains functions to call the trained model. called by run_model.py to integrate both this BART dialogue generator and the BERT emotional classifier
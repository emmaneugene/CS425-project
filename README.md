# Course: CS425-Natural Language Communication

# Project Title: FRANKLY

# Overview

FRANKLY is a chatbot that helps to provide emotional support and helpful advice during trying periods. It comprises 2 separate trained models for classifying the user's emotional state and generating responses to user inputs.

# Emotional classifier (BERT)

- This is an emotional classifier fined tuned from a pre-trained BERT model. There are 5 possible emotions: Joy, Sadness, Anger, Fear and Neutral. Each emotion is mapped to a fixed output that you can alter under the generate_response function in emotional_classifier.py.
- As this BERT code is written with legacy tensorflow libraries, it requires very specific versions of all libraries to run. Hence, it has to use python v3.7.1 and tensorflow v1.15.0. Before running this, you will need to download python 3.7.1 first. A virtual environment can then be created, and all dependencies required can be pip installed from the list in requirements.txt.

# Response generation (BART)

- This is a (conversational) model fine tuned from a pre-trained BART model, which consists of a BERT encoder as well as a GPT2 decoder.
- Conversational datasets were retrieved and processed from [ParlAI](https://github.com/facebookresearch/ParlAI).

## Set up

1. Download python 3.7.1 if you don't already have it
2. Run the following lines of code to create virtual enviroment and download dependencies:

```python
pip install virtualenv                    # if you don't already have it
virtualenv venv --python=python3.7.1      # create a venv with python 3.7.1
source venv/bin/activate                  # activate the virtual environment
which pip                                 # confirm pip is from virtual environment
python --version                          # check that python is indeed 3.7.1
pip install -r requirements.txt           # install required packages
```

3. You're all set!

4. To run FRANKLY:

```python
python run_models.py
```

*This file (run_models.py) imports functions from emotional_classifier/emotional_classifier.py to run the BERT emotional classifier, and dialogue_generation/bart.py to run the BART dialogue generator.*

5. To exit the venv

```python
deactivate
```

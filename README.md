# CS425-project

## emotional_classifier

Overview:
- This is an emotional classifier fined tuned from a pre-trained BERT model. There are 8 possible emotions: Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, and Anticipation. Each emotion is mapped to a fixed output that you can alter under the generate_response function in emotional_classifier.py. 
- As this BERT code is written with legacy tensorflow libraries, it requires very specific versions of all libraries to run. Hence, it has to use python v3.7.1 and tensorflow v1.15.0. Before running this, you will need to download python 3.7.1 first. A virtual environment can then be created, and all dependencies required can be pip installed from the list in requirements.txt. 

To run: 
1. Download python 3.7.1 if you don't already have it 
2. Run the following lines of code to create virtual enviroment and download dependencies:
- pip install virtualenv                    (if you don't already have)
- cd emotional_classifier                   (enter the emotional_classifier directory)
- virtualenv venv --python=python3.7.1      (create a venv with python 3.7.1)
- source venv/bin/activate                  (activate the virtual environment)
- which pip                                 (confirm pip is from virtual environment)
- python --version                          (check that python is indeed 3.7.1)
- pip install -r requirements.txt           (install required packages)
3. You're all set! You can now chat with the bot by running:
- python emotional_classifier.py
4. Type in any random sentence. To exit, press '0'
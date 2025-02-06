print("\nHello there! Do give me a while ah I'm loading... \n\n")

# load model & tokenizer from emotional_classifier
from emotional_classifier import emotional_classifier
ec_model, ec_tokenizer = emotional_classifier.load_model()

# load model & tokenizer from dialogue_generation 
from dialogue_generation import bart
bart_model, bart_tokenizer = bart.load_model()


# CONVERSATION 
print("Hello! This is FRANKLY. I'm so happy to chat with you!\nPlease type 'Bye!' to exit. \n\n")
sentence = input("What's on your mind?: ")

while sentence != 'Bye!':

    emotional_response = emotional_classifier.generate_reply(sentence, ec_model, ec_tokenizer)
    conversational_response = bart.generate_reply(sentence, bart_model, bart_tokenizer)
    print("FRANKLY: " + emotional_response + conversational_response+ "\n")
    sentence = input("What's on your mind?: ")

print("FRANKLY: Have a great day ahead! \n")
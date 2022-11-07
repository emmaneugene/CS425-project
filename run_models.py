print("\nGive me a while ah I'm loading... \n\n")

from emotional_classifier import emotional_classifier

ec_model, ec_tokenizer = emotional_classifier.build_model()


print("Hello! This is FRANKLY. I'm so happy to chat with you!\nPlease type '0' to exit. \n\n")
sentence = input("Please type your input here: ")

while sentence != '0':

    emotional_response = emotional_classifier.getResponse(sentence, ec_model, ec_tokenizer)
    print(emotional_response + "\n")
    sentence = input("Please type a statement here: ")

print("Thank you for chatting with me! \n")
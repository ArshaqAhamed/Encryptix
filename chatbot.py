def chatbot_response(user_input):
    # Convert the input to lowercase for easier matching
    user_input = user_input.lower()

    # Define responses for various inputs
    if "hello" in user_input:
        return "Hi there! How can I help you today?"
    elif "how are you" in user_input:
        return "I'm just a bot, but I'm doing great! How about you?"
    elif "bye" in user_input:
        return "Goodbye! Have a great day!"
    elif "your name" in user_input:
        return "I'm a simple chatbot created by OpenAI's GPT-4."
    elif "help" in user_input:
        return "Sure! I can respond to basic greetings and questions about myself."
    else:
        return "I'm sorry, I don't understand that. Can you please rephrase?"

# Main loop to interact with the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")

# Basic usage (maintaining backward compatibility)
from chatbot import greet, generate_response

print(greet("Hello there!"))
print(generate_response("What's your name?", ["My name is Nova."]))

# Advanced usage
from chatbot import ChatBot

# Create a chatbot with knowledge corpus
corpus = """
My name is Nova. I'm an AI assistant.
I can help with questions and information.
The current time is {time}.
Today is {date}.
"""
bot = ChatBot(corpus)

# Generate responses
print(bot.generate_response("What's your name?"))
print(bot.generate_response("What time is it?"))
print(bot.generate_response("Tell me about yourself"))

# Update knowledge
bot.update_corpus("I was created in 2023.")
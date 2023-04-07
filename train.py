# -*- coding: utf-8 -*-
"""chatgpt valitutto gpt2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r72Y1XQhTj9tvsFQgeVIgWKUM2z1u21z
"""

!pip install -qq transformers tensorflow

with open("all-messages.txt", "r", encoding="utf8") as f:
    messages = f.readlines()
    messages = [x.strip() for x in messages]

"""To remove non-alphabetic characters like emojis and special characters, we can use regular expressions. Here's a sample code that removes all non-alphabetic characters and special characters:"""

import re

# regular expression pattern to remove non-alphabetic characters and special characters
pattern = r'[^a-zA-Z\s\[\]]'

# apply pattern on each message and add cleaned message to a new list
cleaned_messages = []
for message in messages:
    cleaned_message = re.sub(pattern, '', message)
    cleaned_messages.append(cleaned_message)

print("Number of messages after cleaning:", len(cleaned_messages))

cleaned_messages

"""Great! The next step is to clone Lorenzo's chat as the training dataset. We'll create a new list called lorenzo_chat that contains all the messages in which Lorenzo participated. """

# replace "Lorenzo Valitutto" with the name of the person you want to clone
lorenzo_chat = []
for message in cleaned_messages:
    if "[me]" in message:
        lorenzo_chat.append(message)

print("Number of messages in Lorenzo's chat:", len(lorenzo_chat))

"""Great! Let's move on to the next step which is to split the messages to make it more like a conversation.

For this, we need to create a list of conversations where each conversation is a list of messages. Each conversation should start with a message written by Lorenzo, followed by a message written by another participant, and so on.
"""

# create a list of conversations
conversations = []

# initialize conversation with the first message
conversation = [cleaned_messages[0]]

# add messages to the conversation
for i in range(1, len(cleaned_messages)):
    # check if current message was written by Lorenzo
    if "[me]" in cleaned_messages[i]:
        # add current conversation to list of conversations
        conversations.append(conversation)
        # start new conversation with the message written by Lorenzo
        conversation = [cleaned_messages[i]]
    else:
        # add message to the current conversation
        conversation.append(cleaned_messages[i])

# add last conversation to list of conversations
conversations.append(conversation)

print("Number of conversations:", len(conversations))

conversations[0]

def trim_conversations(conversations, max_len=10):
    """
    Trims a list of conversations to a maximum length of `max_len` elements.
    """
    trimmed_conversations = []
    for conversation in conversations:
        if len(conversation) > max_len:
            # trim the conversation to `max_len` elements
            conversation = conversation[:max_len]
        trimmed_conversations.append(conversation)
    return trimmed_conversations

trimmed_conversations = trim_conversations(conversations)

trimmed_conversations[714]

with open("output.txt", "w") as f:
    for conversation in trimmed_conversations:
        for message in conversation:
            f.write(message.replace("[me] ","").replace("[others] ","") + "\n")

"""Great! The next step is to fine-tune a GPT language model using the conversations data.

For this, we'll use the GPT2LMHeadModel class from the transformers library. We'll use the base version of GPT-2 with 117M parameters. We'll also use the LineByLineTextDataset class from the transformers library to preprocess the input data.
"""

from transformers import BertTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, GPT2LMHeadModel

# initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-italian-uncased')

# initialize text dataset
text_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="output.txt",  # replace with the path to your Italian text
    block_size=128
)

# initialize data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# initialize training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=1000,
    logging_steps=1000,
    max_steps=1000
)

# initialize GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)

# initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=text_dataset,
    data_collator=data_collator
)

# train the model
trainer.train()

"""To save the trained model and tokenizer files for later use, you can use the save_pretrained() method of the GPT2LMHeadModel and GPT2Tokenizer classes. This method saves the model weights and configuration to disk, along with the tokenizer files."""

# mount Google Drive to the Colab environment
from google.colab import drive
drive.mount('/content/gdrive')

# set the path to save the model and tokenizer files
model_save_path = "/content/gdrive/MyDrive/model"
tokenizer_save_path = "/content/gdrive/MyDrive/tokenizer"

# save the model and tokenizer files
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

"""Great! Now that you've fine-tuned the GPT-2 model on the preprocessed and split conversations data, you can use it to generate new messages that look like they were written by Lorenzo.

To generate new messages from the trained model, you can use the generate() method of the GPT2LMHeadModel class. This method generates a sequence of text given a prompt, using the probabilities learned during training.
"""

# encode the starting sequence of tokens and move to GPU
input_ids = tokenizer.encode("<|startoftext|>", return_tensors="pt").to(model.device)

# generate new messages from the model
generated_text = model.generate(
    input_ids=input_ids, 
    max_length=50,
    do_sample=True,
    temperature=0.9,
    num_return_sequences=1
)

# decode the generated text
generated_messages = tokenizer.decode(generated_text[0], skip_special_tokens=True)

print("Generated messages:", generated_messages)

model_path = "/content/gdrive/MyDrive/model"
tokenizer_path = "/content/gdrive/MyDrive/tokenizer"

import torch

tokenizer1 = BertTokenizer.from_pretrained(tokenizer_path)
model1 = GPT2LMHeadModel.from_pretrained(model_path)

# set the device to use for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)

# create the function to generate reply messages
def generate_text():

    # encode the user message
    input_ids = tokenizer1.encode("<|startoftext|>", return_tensors="pt").to(device)

    # generate the reply text
    generated_text = model1.generate(
        input_ids=input_ids,
        max_length=100,
        do_sample=True,
        temperature=0.7,
        num_return_sequences=1
    )
    
    # decode the generated text
    generated_message = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    print(str(generated_message).replace("< | startoftext | > ", ""))

generate_text()
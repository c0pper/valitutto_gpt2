from telegram.ext import Updater, CommandHandler
from transformers import GPT2LMHeadModel, BertTokenizer
import torch
import os
import logging

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

bot_token = os.environ.get("token")

model_path = "model"
tokenizer_path = "tokenizer"

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# set the device to use for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def welcome(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text="Welcome")


# create the function to generate reply messages
def generate_text(update, context):
    # encode the user message
    input_ids = tokenizer.encode("<|startoftext|>", return_tensors="pt").to(device)

    # generate the reply text
    generated_text = model.generate(
        input_ids=input_ids,
        max_length=30,
        do_sample=True,
        temperature=0.9,
        num_return_sequences=1
    )

    # decode the generated text
    generated_message = tokenizer.decode(generated_text[0], skip_special_tokens=True)

    generated_message = str(generated_message).replace("< | startoftext | > ", "")

    # send the reply message
    context.bot.send_message(chat_id=update.message.chat_id, text=generated_message)


# create the Telegram bot and handle the /frase command
updater = Updater(token=bot_token, use_context=True)
dispatcher = updater.dispatcher
start_handler = CommandHandler('start', welcome)
start_handler = CommandHandler('frase', generate_text)
dispatcher.add_handler(start_handler)

# start the bot
updater.start_polling()
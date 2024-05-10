#!/usr/bin/env python3
import os
import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import numpy as np
import requests
from questions import answer_question
from functions import functions, run_function
import json

load_dotenv()  # take environment variables from .env.

CODE_PROMPT = """
Here are two input:output examples for code generation. Please use these and follow the styling for future requests that you think are pertinent to the request.
Make sure All HTML is generated with the JSX flavoring.
// SAMPLE 1
// A Blue Box with 3 yellow cirles inside of it that have a red outline
<div style={{   backgroundColor: 'blue',
  padding: '20px',
  display: 'flex',
  justifyContent: 'space-around',
  alignItems: 'center',
  width: '300px',
  height: '100px', }}>
  <div style={{     backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
  <div style={{     backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
  <div style={{     backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
</div>
"""

df = pd.read_csv('maven_pilot/processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tg_bot_token = os.getenv("TG_BOT_TOKEN")

messages = [{
  "role": "system",
  "content": "You are a helpful assistant that answers questions."
}, {
  "role": "system",
  "content": CODE_PROMPT
}]

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="I'm a bot, please talk to me!")
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    messages.append({"role": "user", "content": update.message.text})
    initial_response = openai.chat.completions.create(model="gpt-3.5-turbo",
                                                      messages=messages,
                                                      tools=functions)
    initial_response_message = initial_response.choices[0].message
    messages.append(initial_response_message)
    tool_calls = initial_response_message.tool_calls

    if tool_calls:
        for tool_call in tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            response = run_function(name, args)
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": name,
                "content": str(response),
            })
            if name == 'svg_to_png_bytes':
                await context.bot.send_photo(chat_id=update.effective_chat.id,
                                             photo=response)

        # Generate the final response after processing all tool calls
        final_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        final_answer = final_response.choices[0].message

        # Send the final response if it exists
        if final_answer:
            messages.append(final_answer)
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text=final_answer.content)
        else:
            # Send an error message if something went wrong
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text='Something went wrong, please try again')
    else:
        # If no functions were executed, send the initial response
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text=initial_response_message.content)  
  
async def mozilla(update: Update, context: ContextTypes.DEFAULT_TYPE):
      answer = answer_question(df, question=update.message.text, debug=True)
      await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
  response = openai.images.generate(prompt=update.message.text,
                                    model="dall-e-3",
                                    n=1,
                                    size="1024x1024")
  image_url = response.data[0].url
  image_response = requests.get(image_url)
  await context.bot.send_photo(chat_id=update.effective_chat.id,
                               photo=image_response.content)


if __name__ == '__main__':
  application = ApplicationBuilder().token(tg_bot_token).build()

  start_handler = CommandHandler('start', start)
  mozilla_handler = CommandHandler('mozilla', mozilla)
  image_handler = CommandHandler('image', image)
  chat_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), chat)

  application.add_handler(start_handler)
  application.add_handler(mozilla_handler)
  application.add_handler(image_handler)
  application.add_handler(chat_handler)

  application.run_polling()


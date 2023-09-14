from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ChatAction
import yaml

with open('./config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
TOKEN = cfg["token"]

# Словарь для хранения истории диалогов
instructions = {}

import sys
sys.path.insert(1, "/media/ssd-3t/isviridov/mdetr_work/git_projects/codellama/llama")
from llama import Llama

main_dir = "/media/ssd-3t/isviridov/mdetr_work/git_projects/codellama"
ckpt_dir= main_dir+"/CodeLlama-7b-Instruct"
tokenizer_path=main_dir+"/CodeLlama-7b-Instruct/tokenizer.model"
generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=4096*4,
    max_batch_size=1,
)
print("Model built!")

def start(update, context):
    update.message.reply_text("Hi! I am CodeLLama-Instruct model with 7 billion parameters. Can I help you with coding?", 
                              parse_mode="Markdown")

def echo(update, context):
    user_id = update.message.from_user.id
    text = update.message.text

    if user_id not in instructions:
        instructions[user_id] = [
            [
                {"role": "user", "content": text},
            ]
        ]
    else:
        instructions[user_id][0].append({"role": "user", "content": text})

    try:
        context.bot.send_chat_action(chat_id=update.effective_chat.id,
                                     action=ChatAction.TYPING)
        res = generator.chat_completion(
            instructions[user_id],
            max_gen_len=None,
            temperature=0.2,
            top_p=0.95,
        )[0]
    except AssertionError:
        update.message.reply_text("Your context length is bigger than maximum of 16384.")
        clear_context(update, context)
    else:
        if len(res) > 4096:
            for x in range(0, len(res), 4096):
                update.message.reply_text(res['generation']['content'][x:x+4096], parse_mode="Markdown")
            else:
                update.message.reply_text(res['generation']['content'], parse_mode="Markdown")
        else:
            update.message.reply_text(res['generation']['content'], parse_mode="Markdown")
        instructions[user_id][0].append({"role": "assistant", "content": res['generation']['content']})

def clear_context(update, context):
    user_id = update.message.from_user.id
    if user_id in instructions:
        del instructions[user_id]
        update.message.reply_text("Previous context cleared.", parse_mode="Markdown")
    else:
        update.message.reply_text("You don't have a current context yet.", parse_mode="Markdown")

def help_command(update, context):
    command_list = [
        "/start - Start a dialogue",
        "/clear - Clear previous context"
    ]
    help_text = "\n".join(command_list)
    update.message.reply_text("Hi! I am CodeLLama-Instruct model with 7 billion parameters. I can help you with coding. Available commands:\n" + help_text, parse_mode="Markdown")

def main():
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))
    dp.add_handler(CommandHandler("clear", clear_context))
    dp.add_handler(CommandHandler("help", help_command))
    print("Bot starting...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()

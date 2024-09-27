import logging
import os
from telegram import Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext,
)
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv(".env", override=True)

# Загружаем модель и токенизатор
model_name = "./ru-bert-finetuned-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Загружаем BERT для расчета sentence similarity
modelPath = "./LaBSE-ru-turbo"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Загружаем векторную базу
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# Логирование
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# Функция для поиска ответа
def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)

    answer_start = outputs.start_logits.argmax()
    answer_end = outputs.end_logits.argmax() + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    return answer


# Обработчик команды /start
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Привет! Задай мне вопрос, и я постараюсь найти ответ!")


# Обработчик сообщений (вопросов)
def handle_message(update: Update, context: CallbackContext) -> None:
    question = update.message.text

    # Получаем документ из системы ретривера
    docs = retriever.invoke(question)

    if not docs:
        update.message.reply_text("Не удалось найти информацию для ответа.")
        return

    # Получаем ответ на вопрос
    answer = answer_question(question, docs[0].page_content)

    if answer.strip():
        update.message.reply_text(f"Ответ: {answer}")
    else:
        update.message.reply_text("Извините, не могу найти ответ.")


# Обработчик ошибок
def error(update: Update, context: CallbackContext) -> None:
    logger.warning(f"Ошибка: {context.error}")


# Основная функция
def main() -> None:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

    # Создаем объект Updater и передаем ему токен бота
    updater = Updater(TELEGRAM_TOKEN, use_context=True)

    # Получаем диспетчер для регистрации обработчиков
    dispatcher = updater.dispatcher

    # Регистрируем обработчики
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(
        MessageHandler(Filters.text & ~Filters.command, handle_message)
    )

    # Логирование ошибок
    dispatcher.add_error_handler(error)

    # Запуск бота
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()

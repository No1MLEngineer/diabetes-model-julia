import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define a command handler.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message when the command /start is issued."""
    await update.message.reply_text("Hello! I am your social media assistant bot.")
    await update.message.reply_text("You can use me to post messages and generate hashtags. Use /post <your message> to create a post.")

async def post_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Logs the message to be posted."""
    # context.args is a list of strings, separated by spaces
    message_parts = context.args
    if not message_parts:
        await update.message.reply_text("Please provide a message to post after the /post command. \nExample: /post Hello world!")
        return

    message_to_post = " ".join(message_parts)
    logger.info(f"New post requested: {message_to_post}")
    # For now, we just confirm. Later we'll add actual posting logic.
    # Let's try to generate some hashtags from the message content
    generated_hashtags = _generate_hashtags_from_text(message_to_post)

    post_with_hashtags = message_to_post
    hashtags_string = ""
    if generated_hashtags:
        hashtags_string = " ".join(generated_hashtags)
        post_with_hashtags += f"\n\n{hashtags_string}"
        logger.info(f"Generated hashtags: {hashtags_string}")
    else:
        logger.info("No hashtags were generated for the post.")

    logger.info(f"New post requested (with hashtags): {post_with_hashtags}")
    reply_message = f"Got it! Your message to post:\n\"{message_to_post}\""
    if generated_hashtags:
        reply_message += f"\n\nSuggested hashtags: {hashtags_string}"
    else:
        reply_message += "\n\n(No hashtags were automatically suggested for this post.)"

    await update.message.reply_text(reply_message)

def _generate_hashtags_from_text(text: str) -> list[str]:
    """Generates simple hashtags based on the input text."""
    if not text or not text.strip():
        logger.info("Hashtag generation: Input text is empty.")
        return []

    words = text.strip().split()
    # Use the first few words (up to 3) as a proxy for the topic.
    # Filter out very short words from being topic words unless they are the only word.
    potential_topic_words = [word for word in words[:3] if len(word) > 2 or len(words) == 1]

    if not potential_topic_words: # If all first 3 words were too short
        logger.info(f"Hashtag generation: No suitable topic words found in '{text[:30]}...'")
        # Fallback: try to use any longer word from the text if no topic words found initially
        longer_words_in_text = [word for word in words if len(word) > 3]
        if longer_words_in_text:
            potential_topic_words = longer_words_in_text[:2] # Take first two longer words
            logger.info(f"Hashtag generation: Using fallback topic words: {potential_topic_words}")
        else: # Still no suitable words
             logger.info(f"Hashtag generation: No suitable topic words found at all in '{text[:30]}...'")
             return []


    hashtags = []

    # Hashtag from the combined topic words
    # e.g., "New Cool Product" -> #NewCoolProduct
    camel_case_topic = "".join(word.capitalize() for word in potential_topic_words)
    if camel_case_topic:
        hashtags.append(f"#{camel_case_topic}")

    # Hashtags from individual topic words
    for word in potential_topic_words:
        # Already filtered for length mostly, but ensure it's not just symbols or too short
        if len(word) > 2 or (len(word) > 1 and word.isalnum()): # Allow 2-char if alphanumeric (e.g. AI)
            hashtags.append(f"#{word.capitalize()}")

    # A few generic related terms based on keywords in the whole text
    # Convert full text to lower for keyword spotting
    lower_text = text.lower()
    if "ai" in lower_text or "artificial intelligence" in lower_text:
        hashtags.extend(["#AI", "#MachineLearning", "#DeepLearning"])
    if "social media" in lower_text: # Check for "social media" as a phrase
        hashtags.extend(["#SocialMediaMarketing", "#DigitalMarketing"])
    elif "social" in lower_text and "post" in lower_text: # Check for "social" and "post"
        hashtags.extend(["#SocialPosting", "#OnlinePresence"])

    # Remove duplicates by converting to set and back to list
    return list(set(hashtags))

def main() -> None:
import os

    """Start the bot."""
    # Get the token from environment variable
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set!")
        return

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(token).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("post", post_message))
    # application.add_handler(CommandHandler("generate_hashtags", generate_hashtags_command)) # Removed

    # Add an error handler
    application.add_error_handler(error_handler)

    # Run the bot until the user presses Ctrl-C
    logger.info("Starting bot...")
    application.run_polling()

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.error(f"Update {update} caused error {context.error}")
    # Optionally, notify the user or take other actions
    # if isinstance(context.error, telegram.error.NetworkError):
    #     if update and isinstance(update, Update) and update.effective_message:
    #         await update.effective_message.reply_text("Sorry, there was a network problem. Please try again later.")

if __name__ == "__main__":
    main()

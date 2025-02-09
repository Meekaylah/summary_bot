import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import openai
from dotenv import load_dotenv
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

app = AsyncApp(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
)

openai_client = openai.OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    timeout=30.0,
    max_retries=3,
)

USER_CACHE: Dict[str, str] = {}
ONGOING_GISTS: Dict[str, asyncio.Task] = {}
CHUNK_SIZE = 5000


async def get_user_display_name(client, user_id: str) -> str:
    """Gets user's display name with efficient caching."""
    if user_id not in USER_CACHE:
        try:
            user_info = await client.users_info(user=user_id)
            profile = user_info["user"]["profile"]
            USER_CACHE[user_id] = profile.get("display_name") or profile.get(
                "real_name"
            )
        except Exception as e:
            logger.error(f"Failed to fetch user info for {user_id}: {e}")
            return "Unknown User"
    return USER_CACHE[user_id]


async def get_channel_messages(
    client, channel_id: str, oldest_ts: str
) -> List[Dict]:
    """Gets messages efficiently using pagination and concurrency."""
    try:
        response = await client.conversations_history(
            channel=channel_id, oldest=oldest_ts, limit=1000, inclusive=True
        )

        messages = list(reversed(response["messages"]))
        threads = [msg for msg in messages if msg.get("thread_ts")]

        if threads:
            tasks = [
                asyncio.create_task(
                    client.conversations_replies(
                        channel=channel_id, ts=thread["thread_ts"], limit=1000
                    )
                )
                for thread in threads
            ]
            thread_responses = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            for thread, resp in zip(threads, thread_responses):
                if isinstance(resp, dict):
                    thread_index = messages.index(thread)
                    thread_messages = resp.get("messages", [])
                    thread_messages = [
                        msg
                        for msg in thread_messages
                        if msg["ts"] != thread["ts"]
                    ]
                    messages[thread_index + 1 : thread_index + 1] = (
                        thread_messages
                    )

        return messages
    except Exception as e:
        logger.error(f"Error fetching messages: {e}")
        return []


def chunk_messages(formatted_messages: str) -> List[str]:
    """Split messages into chunks of approximately CHUNK_SIZE tokens."""
    messages = formatted_messages.split("\n")
    chunks = []
    current_chunk = []
    current_size = 0

    for message in messages:
        message_size = len(message) // 4

        if current_size + message_size > CHUNK_SIZE:
            chunks.append("\n".join(current_chunk))
            current_chunk = [message]
            current_size = message_size
        else:
            current_chunk.append(message)
            current_size += message_size

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


async def format_messages(messages: List[Dict], client) -> str:
    """Formats messages concurrently maintaining chronological order."""
    formatted = []
    user_tasks = {}

    for msg in messages:
        user_id = msg.get("user")
        if user_id and user_id not in user_tasks:
            user_tasks[user_id] = asyncio.create_task(
                get_user_display_name(client, user_id)
            )

    await asyncio.gather(*user_tasks.values())

    for msg in messages:
        try:
            user_id = msg.get("user")
            if not user_id:
                continue

            username = USER_CACHE.get(user_id, "Unknown User")
            is_thread_reply = (
                "thread_ts" in msg and msg["thread_ts"] != msg["ts"]
            )
            prefix = "  └ " if is_thread_reply else ""
            formatted.append(
                f"{prefix}{username}: {msg.get('text', '')}"
            )

        except Exception as e:
            logger.error(f"Error formatting message: {e}")

    return "\n".join(formatted)


def generate_chunk_summary(
    chunk: str, chunk_number: int, total_chunks: int
) -> str:
    """Generates summary for a specific chunk with context."""
    try:
        if chunk_number == 1:
            context = "Start the gist from beginning. This is the first part."
        else:
            context = (
                "This na continuation of the gist. Make e flow like one story."
            )

        response = openai_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You be Nigerian wey dey share gist for slack channel. "
                        "Rules:\n"
                        "- Use pure Nigerian Pidgin English\n"
                        "- Keep am short but detailed\n"
                        "- Focus only on the messages provided\n"
                        "- Add funny Nigerian expressions and reactions\n"
                        "- Make e funny but still pass the message\n"
                        "- No need to mention time stamps\n"
                        f"- {context}\n"
                    ),
                },
                {"role": "user", "content": f"Continue the gist:\n{chunk}"},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Summary generation error for chunk {chunk_number}: {e}")
        return "System don tire! Try again later."


async def process_gist(channel_id: str, client):
    try:
        bot_id = (await client.auth_test())["user_id"]
        response = await client.conversations_history(
            channel=channel_id, limit=1000
        )

        last_bot_ts = None
        for msg in response.get("messages", []):
            if msg.get("user") == bot_id:
                last_bot_ts = msg["ts"]
                break

        messages = await get_channel_messages(
            client, channel_id, last_bot_ts or "0"
        )

        messages = [msg for msg in messages if msg.get("user") != bot_id]

        if not messages:
            await client.chat_postMessage(
                channel=channel_id,
                text="Nothing new don happen since my last gist o!",
            )
            return

        formatted_text = await format_messages(messages, client)
        chunks = chunk_messages(formatted_text)

        logger.info(f"Processing summary for channel {channel_id}")

        first_summary = await asyncio.get_event_loop().run_in_executor(
            None, generate_chunk_summary, chunks[0], 1, len(chunks)
        )

        forward_text = ""
        if last_bot_ts:
            try:
                permalink_response = await client.chat_getPermalink(
                    channel=channel_id, message_ts=last_bot_ts, limit=1000
                )
                permalink = permalink_response.get("permalink", "")
                if permalink:
                    forward_text = (
                        f"As we been yarn for the last gist {permalink}\n\n"
                    )
            except Exception as e:
                logger.error(f"Failed to get permalink for previous gist: {e}")

        initial_message = await client.chat_postMessage(
            channel=channel_id,
            text=f"{forward_text}Make we continue from where we stop!:\n\n{first_summary}",
        )
        thread_ts = initial_message["ts"]

        with ThreadPoolExecutor() as executor:
            for i, chunk in enumerate(chunks[1:], 2):
                summary = await asyncio.get_event_loop().run_in_executor(
                    executor, generate_chunk_summary, chunk, i, len(chunks)
                )
                await client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=summary,
                )

    except Exception as e:
        logger.error(f"Gist processing error: {e}")
        await client.chat_postMessage(
            channel=channel_id,
            text="Ahh! Error don happen. Make you try again.",
        )
    finally:
        if channel_id in ONGOING_GISTS:
            del ONGOING_GISTS[channel_id]


@app.command("/gist")
async def handle_gist(ack, body, client):
    """Handles /gist command with improved concurrency."""
    channel_id = body["channel_id"]
    user_id = body["user_id"]

    await ack()

    if channel_id in ONGOING_GISTS and not ONGOING_GISTS[channel_id].done():
        await client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text="Hold on, I still dey process the last gist request!",
        )
        return

    await client.chat_postEphemeral(
        channel=channel_id,
        user=user_id,
        text="I don receive your gist request. E go soon ready!",
    )

    ONGOING_GISTS[channel_id] = asyncio.create_task(
        process_gist(channel_id, client)
    )


@app.command("/summarize")
async def handle_summarize(ack, body, client):
    """Alias for /gist command"""
    await handle_gist(ack, body, client)


async def main():
    """Main entry point using async handler."""
    logger.info("Starting Threaded Summary Bot")
    handler = AsyncSocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())

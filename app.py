import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from transformers import pipeline

app = Flask(__name__)

load_dotenv()

# Load environment variables
SLACK_TOKEN = os.getenv('SLACK_TOKEN')

# Initialize Slack client
client = WebClient(token=SLACK_TOKEN)

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/summarize", methods=["POST"])
def summarize():
    """Handles the /summarize Slack command."""
    try:
        # Extract channel ID from the incoming request
        channel_id = request.form.get("channel_id")

        if not channel_id:
            return jsonify({"error": "No channel_id found in the request."})

        # Fetch messages from the last hour
        response = client.conversations_history(channel=channel_id, limit=100)

        messages = response["messages"]
        text_to_summarize = " ".join([msg["text"] for msg in messages])
        # print(text_to_summarize)

        # Generate summary
        summary = summarizer(text_to_summarize, max_length=300, min_length=30, do_sample=False)
        formatted_summary = summary[0]["summary_text"]

        # print(formatted_summary)
        # Send summary back to Slack
        client.chat_postMessage(channel=channel_id, text=f"*Summary of the last hour:*\n {formatted_summary}")

        return jsonify({"response_type": "in_channel", "text": "Summarization complete!"})

    except SlackApiError as e:
        return jsonify({"error": f"Slack API error: {e.response['error']}"})
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"})

if __name__ == "__main__":
    app.run(port=3000)

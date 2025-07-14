## 🎯 AI Lead Generation Agent

This Streamlit application searches Quora with DuckDuckGo (filtered to the `quora.com` domain) and Reddit using the official API. It downloads each page with simple HTTP requests, extracts user interactions with a local Mistral model and saves the results to an Excel file.

### Features
- Searches Quora links via DuckDuckGo
- Queries Reddit posts through the Reddit API
- Fetches pages directly via HTTP requests
- Uses a local `mistral:7b-instruct` model for extraction
- Outputs the collected data to an Excel spreadsheet
- Choose how many links to process

### Setup

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:

   ```bash
   streamlit run ai_lead_generation_agent.py
   ```

When prompted, provide the Excel filename and describe the leads you are looking for.

The Reddit API requires credentials set in the `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT` environment variables. If these are not provided, Reddit search will be skipped.

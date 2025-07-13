## 🎯 AI Lead Generation Agent

This Streamlit application searches Quora using DuckDuckGo, downloads each page with simple HTTP requests, extracts user interactions with a local Mistral model and saves the results to an Excel file.

### Features
- Searches Quora links via DuckDuckGo
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

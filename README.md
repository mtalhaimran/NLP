# LLMs and AI Agents In Progress

This repository contains experiments with large language models and agent-based tools.
One of the main utilities is the **Autolead** app which crawls potential leads from the web
and generates a PDF report using multiple AI agents.

## Running Autolead

Autolead is located in `Enigma_grp_diag_sys/Autolead.py` and uses Streamlit for the UI.
Follow the steps below to start the application:

1. **Install dependencies**
   ```bash
   cd Enigma_grp_diag_sys
   pip install -r requirements.txt
   ```
2. **Launch the Streamlit app**
   ```bash
   streamlit run Autolead.py
   ```
3. **Use the interface**
   - Enter a short description of your target leads.
   - Click **Find Leads** to let the app search Quora and score the results.
   - After reviewing the score, click **Analyze Lead** (or **Analyze Anyway** if the score is low).
   - Once the analysis completes, download the generated PDF report.

The application searches Quora for questions related to your query, evaluates the number of answers
and views, and then runs a multi-agent pipeline to produce a marketing report.

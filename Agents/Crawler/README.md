## 🎯 AI Lead Generation Agent - Powered by Firecrawl's Extract Endpoint

The AI Lead Generation Agent automates the process of finding and qualifying potential leads from Quora. It uses Firecrawl's search and the new Extract endpoint to identify relevant user profiles, extract valuable information, and organize it into a structured format in Google Sheets. This agent helps sales and marketing teams efficiently build targeted lead lists while saving hours of manual research.

### Features
- **Targeted Search**: Uses Firecrawl's search endpoint to find relevant Quora URLs based on your search criteria
- **Intelligent Extraction**: Leverages Firecrawl's new Extract endpoint to pull user information from Quora profiles
- **Automated Processing**: Formats extracted user information into a clean, structured format
- **Google Sheets Integration**: Automatically creates and populates Google Sheets with lead information
- **Customizable Criteria**: Allows you to define specific search parameters to find your ideal leads for your niche

### Setup

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. When running the Streamlit app you'll be prompted for a **HuggingFace Repo ID**. Provide the repo for the chat model you would like to use (e.g. `meta-llama/Llama-3-8B-Instruct`).

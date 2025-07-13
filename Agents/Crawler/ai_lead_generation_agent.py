import streamlit as st
from agno.agent import Agent
from duckduckgo_search import ddg
from langchain_community.chat_models.huggingface import ChatHuggingFaceHub
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from composio_phidata import Action, ComposioToolSet
import json

class QuoraUserInteractionSchema(BaseModel):
    username: str = Field(description="The username of the user who posted the question or answer")
    bio: str = Field(description="The bio or description of the user")
    post_type: str = Field(description="The type of post, either 'question' or 'answer'")
    timestamp: str = Field(description="When the question or answer was posted")
    upvotes: int = Field(default=0, description="Number of upvotes received")
    links: List[str] = Field(default_factory=list, description="Any links included in the post")

class QuoraPageSchema(BaseModel):
    interactions: List[QuoraUserInteractionSchema] = Field(description="List of all user interactions (questions and answers) on the page")

def search_for_urls(company_description: str, num_links: int) -> List[str]:
    query = f"site:quora.com {company_description}"
    results = ddg(query, max_results=num_links) or []
    return [r.get("href") for r in results if r.get("href")]

def extract_user_info_from_urls(urls: List[str], hf_api_token: str) -> List[dict]:
    user_info_list = []
    loader = PlaywrightURLLoader(urls, continue_on_failure=True)
    docs = loader.load()

    llm = ChatHuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.0, "max_new_tokens": 512},
        huggingfacehub_api_token=hf_api_token,
    )

    parser = PydanticOutputParser(pydantic_object=QuoraPageSchema)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Extract all user interactions from this Quora page. {format_instructions}"),
            ("human", "{page_content}"),
        ]
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    for url, doc in zip(urls, docs):
        try:
            result = chain.predict(
                page_content=doc.page_content,
                format_instructions=parser.get_format_instructions(),
            )
            parsed = parser.parse(result)
            if parsed.interactions:
                user_info_list.append(
                    {
                        "website_url": url,
                        "user_info": [i.dict() for i in parsed.interactions],
                    }
                )
        except Exception:
            pass

    return user_info_list

def format_user_info_to_flattened_json(user_info_list: List[dict]) -> List[dict]:
    flattened_data = []
    
    for info in user_info_list:
        website_url = info["website_url"]
        user_info = info["user_info"]
        
        for interaction in user_info:
            flattened_interaction = {
                "Website URL": website_url,
                "Username": interaction.get("username", ""),
                "Bio": interaction.get("bio", ""),
                "Post Type": interaction.get("post_type", ""),
                "Timestamp": interaction.get("timestamp", ""),
                "Upvotes": interaction.get("upvotes", 0),
                "Links": ", ".join(interaction.get("links", [])),
            }
            flattened_data.append(flattened_interaction)
    
    return flattened_data

def create_google_sheets_agent(composio_api_key: str, hf_api_token: str) -> Agent:
    composio_toolset = ComposioToolSet(api_key=composio_api_key)
    google_sheets_tool = composio_toolset.get_tools(actions=[Action.GOOGLESHEETS_SHEET_FROM_JSON])[0]
    
    google_sheets_agent = Agent(
        model=ChatHuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            model_kwargs={"temperature": 0.0, "max_new_tokens": 512},
            huggingfacehub_api_token=hf_api_token,
        ),
        tools=[google_sheets_tool],
        show_tool_calls=True,
        system_prompt="You are an expert at creating and updating Google Sheets. You will be given user information in JSON format, and you need to write it into a new Google Sheet.",
        markdown=True
    )
    return google_sheets_agent

def write_to_google_sheets(flattened_data: List[dict], composio_api_key: str, hf_api_token: str) -> str:
    google_sheets_agent = create_google_sheets_agent(composio_api_key, hf_api_token)
    
    try:
        message = (
            "Create a new Google Sheet with this data. "
            "The sheet should have these columns: Website URL, Username, Bio, Post Type, Timestamp, Upvotes, and Links in the same order as mentioned. "
            "Here's the data in JSON format:\n\n"
            f"{json.dumps(flattened_data, indent=2)}"
        )
        
        create_sheet_response = google_sheets_agent.run(message)
        
        if "https://docs.google.com/spreadsheets/d/" in create_sheet_response.content:
            google_sheets_link = create_sheet_response.content.split("https://docs.google.com/spreadsheets/d/")[1].split(" ")[0]
            return f"https://docs.google.com/spreadsheets/d/{google_sheets_link}"
    except Exception:
        pass
    return None

def create_prompt_transformation_agent(hf_api_token: str) -> Agent:
    return Agent(
        model=ChatHuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            model_kwargs={"temperature": 0.0, "max_new_tokens": 512},
            huggingfacehub_api_token=hf_api_token,
        ),
        system_prompt="""You are an expert at transforming detailed user queries into concise company descriptions.
Your task is to extract the core business/product focus in 3-4 words.

Examples:
Input: "Generate leads looking for AI-powered customer support chatbots for e-commerce stores."
Output: "AI customer support chatbots for e commerce"

Input: "Find people interested in voice cloning technology for creating audiobooks and podcasts"
Output: "voice cloning technology"

Input: "Looking for users who need automated video editing software with AI capabilities"
Output: "AI video editing software"

Input: "Need to find businesses interested in implementing machine learning solutions for fraud detection"
Output: "ML fraud detection"

Always focus on the core product/service and keep it concise but clear.""",
        markdown=True
    )

def main():
    st.title("🎯 AI Lead Generation Agent")
    st.info("Generate leads from Quora by searching for relevant posts and extracting user information.")

    with st.sidebar:
        st.header("API Keys")
        hf_api_token = st.text_input("HuggingFace API Token", type="password")
        composio_api_key = st.text_input("Composio API Key", type="password")
        st.caption(" Get your Composio API key from [Composio's website](https://composio.ai)")
        
        num_links = st.number_input("Number of links to search", min_value=1, max_value=10, value=3)
        
        if st.button("Reset"):
            st.session_state.clear()
            st.experimental_rerun()

    user_query = st.text_area(
        "Describe what kind of leads you're looking for:",
        placeholder="e.g., Looking for users who need automated video editing software with AI capabilities",
        help="Be specific about the product/service and target audience. The AI will convert this into a focused search query."
    )

    if st.button("Generate Leads"):
        if not all([hf_api_token, composio_api_key, user_query]):
            st.error("Please fill in all the API keys and describe what leads you're looking for.")
        else:
            with st.spinner("Processing your query..."):
                transform_agent = create_prompt_transformation_agent(hf_api_token)
                company_description = transform_agent.run(f"Transform this query into a concise 3-4 word company description: {user_query}")
                st.write("🎯 Searching for:", company_description.content)

            with st.spinner("Searching for relevant URLs..."):
                urls = search_for_urls(company_description.content, num_links)
            
            if urls:
                st.subheader("Quora Links Used:")
                for url in urls:
                    st.write(url)
                
                with st.spinner("Extracting user info from URLs..."):
                    user_info_list = extract_user_info_from_urls(urls, hf_api_token)
                
                with st.spinner("Formatting user info..."):
                    flattened_data = format_user_info_to_flattened_json(user_info_list)
                
                with st.spinner("Writing to Google Sheets..."):
                    google_sheets_link = write_to_google_sheets(flattened_data, composio_api_key, hf_api_token)
                
                if google_sheets_link:
                    st.success("Lead generation and data writing to Google Sheets completed successfully!")
                    st.subheader("Google Sheets Link:")
                    st.markdown(f"[View Google Sheet]({google_sheets_link})")
                else:
                    st.error("Failed to retrieve the Google Sheets link.")
            else:
                st.warning("No relevant URLs found.")

if __name__ == "__main__":
    main()
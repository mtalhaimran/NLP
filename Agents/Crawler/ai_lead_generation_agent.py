import streamlit as st
from duckduckgo_search import ddg
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from typing import List
import pandas as pd
import json


def search_for_urls(company_description: str, num_links: int) -> List[str]:
    query = f"site:quora.com {company_description}"
    results = ddg(query, max_results=num_links) or []
    return [r.get("href") for r in results if r.get("href")]

def extract_user_info_from_urls(urls: List[str]) -> List[dict]:
    user_info_list = []
    loader = PlaywrightURLLoader(urls, continue_on_failure=True)
    docs = loader.load()

    llm = ChatOllama(model="mistral:7b-instruct")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Extract all user interactions from this Quora page as JSON. "
                    "Return a dictionary with an 'interactions' key containing a list "
                    "of objects with 'username', 'bio', 'post_type', 'timestamp', "
                    "'upvotes' and 'links'."
                ),
            ),
            ("human", "{page_content}"),
        ]
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    for url, doc in zip(urls, docs):
        try:
            result = chain.predict(page_content=doc.page_content)
            parsed = json.loads(result)
            interactions = parsed.get("interactions", [])
            if interactions:
                user_info_list.append(
                    {
                        "website_url": url,
                        "user_info": interactions,
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


def write_to_excel(flattened_data: List[dict], path: str) -> None:
    df = pd.DataFrame(flattened_data)
    df.to_excel(path, index=False)


def transform_query(user_query: str) -> str:
    llm = ChatOllama(model="mistral:7b-instruct")
    system_prompt = """You are an expert at transforming detailed user queries into concise company descriptions.
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

Always focus on the core product/service and keep it concise but clear."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}"),
    ])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.predict(query=user_query).strip()

def main():
    st.title("🎯 AI Lead Generation Agent")
    st.info("Generate leads from Quora by searching for relevant posts and extracting user information.")

    with st.sidebar:
        st.header("Configuration")
        output_file = st.text_input("Output Excel filename", value="leads.xlsx")
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
        if not all([user_query, output_file]):
            st.error("Please provide a description and output filename.")
        else:
            with st.spinner("Processing your query..."):
                company_description = transform_query(user_query)
                st.write("🎯 Searching for:", company_description)

            with st.spinner("Searching for relevant URLs..."):
                urls = search_for_urls(company_description, num_links)
            
            if urls:
                st.subheader("Quora Links Used:")
                for url in urls:
                    st.write(url)
                
                with st.spinner("Extracting user info from URLs..."):
                    user_info_list = extract_user_info_from_urls(urls)
                
                with st.spinner("Formatting user info..."):
                    flattened_data = format_user_info_to_flattened_json(user_info_list)
                
                with st.spinner("Writing to Excel..."):
                    write_to_excel(flattened_data, output_file)
                st.success("Lead generation completed successfully!")
                st.subheader("Saved File:")
                st.write(output_file)
            else:
                st.warning("No relevant URLs found.")

if __name__ == "__main__":
    main()

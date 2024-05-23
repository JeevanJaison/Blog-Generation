import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to get the response from the Llama 2 model
def getLlamaResponse(input_text, no_words, blog_style):
    # Calling the Llama model from the local using the ctransformers
    llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type="llama",
                        config={"max_new_tokens": 256, "temperature": 0.01})
    
    # Prompt template
    template = "Write a blog for a {blog_style} job profile on the topic {input_text} within {no_words} words."

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"], template=template)

    # Generate response from the Llama 2 model
    response = llm.invoke(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)
    return response

st.set_page_config(page_title="Generate Blogs", page_icon="👽", layout="centered", initial_sidebar_state="collapsed")

st.header("Generate Blogs 👽")

# For the input text field 
input_text = st.text_input("Enter the blog topic: ")

# Creating two more columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("No of words: ")
with col2:
    blog_style = st.selectbox("Writing the blog for", ("Researcher", "Data Scientist", "Common people"), index=0)

# Creating button
submit = st.button("Generate")

# Final response
if submit:
    st.write(getLlamaResponse(input_text, no_words, blog_style))

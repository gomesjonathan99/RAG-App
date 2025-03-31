# '''Importing necessary libraries'''

# import yaml
# from pathlib import Path

# def get_prompt(user_input: str, docs: str): #<class 'str'>
    
#     file_path = Path(__file__).parent / "prompt.yaml"
    
#     if not file_path.exists():
#         raise FileNotFoundError(f"File not found: {file_path}")

#     with file_path.open("r", encoding="utf-8") as file:
#         prompt = yaml.safe_load(file)

#     template = prompt["ai_prompt"]["template"]
#     prompt = template.replace("{{ user_input }}", user_input).replace("{{ docs }}", docs)
#     return prompt


# # Example Usage
# user_input = "What are AI agents?"
# docs = "AI agents are software programs that perform automated tasks."
# print(get_prompt(user_input, docs))

from langchain.prompts import PromptTemplate, ChatPromptTemplate
# def get_prompt() -> PromptTemplate:
#     """
#     Function to create a prompt template for the AI assistant.

#     Args:
#         user_input (str): The user's question or input.
#         docs (str): The relevant documents based on the user's query.

#     Returns:
#         PromptTemplate: A formatted prompt template for the AI assistant.
#     """
#     ai_prompt = PromptTemplate(
#         input_variables=["user_input", "docs"],
#         template="""
#         You are an advanced AI assistant that answers user queries based on the provided relevant documents.

#         ## Context:
#         You have access to multiple relevant documents that contain information related to the user's question.

#         ## Instructions:
#         - Carefully read the provided documents.
#         - Use **only** the information from the documents to generate your answer.
#         - If the information is unclear or incomplete, respond with **"I don't have enough information"** instead of making up an answer.
#         - Keep your response **concise (100-500 words)**.
#         - Follow this structure:
#         - **Title**: A clear and concise heading summarizing the response.
#         - **Body**: A well-structured explanation with key points.
#         - **Conclusion**: A brief summary of the key points.

#         ## Inputs:
#         User Question:
#         {user_input}

#         Relevant Documents based on the User Query:
#         {docs}

#         ## Output:
#         - **Response must be in plain text format (no Markdown, no special formatting)**.
#         - Use proper sentence structure and paragraph spacing.
#         - Do **not** include any additional information outside of the documents.

#         **Response Format (Strictly Follow This Structure):**

#         Title: [Insert Title Here]

#         Body:
#         - [Detailed answer based on the documents]
#         - [Use paragraphs and bullet points if necessary]

#         Conclusion:
#         - [Summarize key takeaways in a few sentences]
#         """
#     )
#     return ai_prompt

def get_system_prompt() -> str:
    """
    Function to create a system prompt for the AI assistant.

    Returns:
        str: A formatted system prompt for the AI assistant.
    """
    system_prompt = (
    "You are an advanced AI assistant specializing in question-answering tasks. "
    "Use the retrieved context below to provide accurate and relevant answers. "
    "If the answer is not found in the provided context, clearly state that you don't know."
    "\n\n"
    "Context:\n{context}\n\n"
    "Ensure responses are concise, well-structured, and directly address the question."
)
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
    return system_prompt, prompt

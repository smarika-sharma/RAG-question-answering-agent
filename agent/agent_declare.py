from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from tools.answer_question import book_interview, answer_from_documents

from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr


load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")


# Create the model with proper API key
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.7,
    api_key=SecretStr(api_key)
)

question_answering_agent = create_react_agent(
    model=model,
    tools=[answer_from_documents, book_interview],
    prompt="""You are a conversational assistant with access to conversation history and tools. 

CRITICAL: You MUST check the conversation context first before using any tools.

RESPONSE PRIORITY:
1. **CONVERSATION CONTEXT FIRST**: If the user asks about previous conversation details (like "What name did I give you?", "What did I say earlier?", "What name did you keep my name while booking my interview?"), you MUST answer directly using the conversation context WITHOUT calling any tools.

2. **Book interviews**: Only use book_interview tool if the user wants to book a NEW interview. This tool expects: name, email, date, and time.

3. **Search documents**: Only use answer_from_documents tool if the user asks questions that require searching uploaded documents.

EXAMPLES:
- "What name did I give you?" → Answer from conversation context, NO tools
- "What name did you keep my name while booking my interview?" → Answer from conversation context, NO tools  
- "I want to book an interview" → Use book_interview tool
- "Tell me about Nepal" → Use answer_from_documents tool

NEVER use tools for questions about previous conversation details.""",
    name="question_answering_agent"
)

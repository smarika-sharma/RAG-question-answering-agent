from fastapi import APIRouter, Request
from pydantic import BaseModel
from agent.agent_declare import question_answering_agent
from services.redis_service import redis_service

router = APIRouter()

class Query(BaseModel):
    query: str
    # sesion is changed per user or per session
    session_id: str = "default"

@router.post("/chat")
async def chat(payload: Query):
    user_query = payload.query
    session_id = payload.session_id
    
    try:
        conversation_context = redis_service.get_conversation_context(session_id)
        
        if conversation_context:
            enhanced_query = f"Context: {conversation_context}\n\nCurrent question: {user_query}"
        else:
            enhanced_query = user_query
        
        # invoke the question answering agent to handle user query
        response = question_answering_agent.invoke({
            "messages": [{"role": "user", "content": enhanced_query}]
        })
        
        #  final response from the agent
        if response and "messages" in response:
            # Get all messages to find the final result
            messages = response["messages"]
            
            # looking for tool response cause it contains the actual answer
            final_response = None
            agent_name = "unknown"
            
            for message in messages:
                if hasattr(message, 'content'):
                    content = message.content
                elif isinstance(message, dict) and 'content' in message:
                    content = message['content']
                else:
                    content = str(message)
                
                # Look for ToolMessage which contains the actual tool response
                if hasattr(message, '__class__') and 'ToolMessage' in str(message.__class__):
                    if content and not any(keyword in content.lower() for keyword in ['transfer', 'routing', 'assign']):
                        final_response = content
                        break

                # if no tools is used, then it is a direct AI response
                elif hasattr(message, '__class__') and 'AIMessage' in str(message.__class__) and content:
                    if not any(keyword in content.lower() for keyword in ['transfer', 'routing', 'assign', 'ready to provide', 'back']):
                        final_response = content
                        agent_name = getattr(message, 'name', 'question_answering_agent')
                        break

                # checking for agent responses
                elif hasattr(message, 'name') and getattr(message, 'name') in ['answer_assistant', 'booking_assistant'] and content:
                    if not any(keyword in content.lower() for keyword in ['transfer', 'routing', 'assign', 'ready to provide', 'back']):
                        final_response = content
                        agent_name = getattr(message, 'name', 'unknown')
                        break

                elif isinstance(message, dict) and message.get('name') in ['answer_assistant', 'booking_assistant'] and content:
                    if not any(keyword in content.lower() for keyword in ['transfer', 'routing', 'assign', 'ready to provide', 'back']):
                        final_response = content
                        agent_name = message.get('name', 'unknown')
                        break
            
            if final_response:
                # redis implementation for storing conversation 
                redis_service.store_conversation(session_id, user_query, final_response, agent_name)
                
                return {
                    "response": final_response,
                    "session_id": session_id,
                    "status": "success",
                    "agent_used": agent_name
                }
            else:
                return {
                    "response": "I apologize, but I couldn't get a proper response from the agents.",
                    "session_id": session_id,
                    "status": "error"
                }
        else:
            return {
                "response": "I apologize, but I couldn't process your request.",
                "session_id": session_id,
                "status": "error"
            }
            
    except Exception as e:
        return {
            "response": f"An error occurred: {str(e)}",
            "session_id": session_id,
            "status": "error"
        }


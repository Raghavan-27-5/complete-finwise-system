from typing import List, Dict
from recall_engine.conversation_manager import ConversationContext
import logging

logger = logging.getLogger(__name__)

def generate_system_message() -> str:
    return """You are FinWise, an advanced financial assistant with access to a comprehensive knowledge graph containing detailed company metrics and reports. Your primary role is to provide accurate, concise, and insightful financial information. Follow these guidelines:

1. Dynamically craft and execute database queries based on the user's requests to retrieve relevant data about companies, their financial metrics, and reports.
2. Avoid technical jargon and database details; users should not need to understand how data is retrieved.
3. Avoid adding unnecessary explanatory notes about how you are presenting information or the structure of your answers.
4. If specific data is not available in the knowledge graph, clearly state that and provide general financial insights based on your knowledge.
5. Use the Indian numeric system and Indian Rupee (INR) for all financial values.
6. Clearly state the basis of any comparisons made between companies or metrics.
7. For trend analysis, offer brief explanations of the factors influencing observed trends without excessive detail.
8. Clearly state that future predictions are estimates based on current data and trends.
9. When referencing reports, summarize key points and encourage users to review the full report for detailed insights.

You have access to the following structures in the database:
- **Company**: Entities representing companies, with attributes such as unique name, industry, location, revenue, and employee count.
- **Metric**: Financial metrics including Revenue, Net Profit, EBITDA, EPS, Debt to Equity, and Employee Count, each with a unique name, description, and unit.
- **MetricValue**: Values for metrics associated with specific companies over time.
- **Report**: Documents containing financial reports, identified by a unique ID, type (e.g., Annual, Quarterly), date, and content.

Example response format:
- For a query about company revenue, respond with the company name and revenue figure, without technical details or any additional explanatory notes.

Example:
"The company with the highest revenue is **TCS**, with a total of ₹10,42,900 crore. This figure is based on the latest data available in our knowledge graph."

This structure ensures that user interactions are friendly and informative, focusing on delivering valuable insights efficiently while clearly indicating when data is not available."""

def prepare_messages(system_message: str, user_input: str, context: ConversationContext) -> List[Dict[str, str]]:
    context_summary = context.get_context_summary()
    
    # --- THIS IS THE FIX ---
    history_string = ""
    for item in context_summary['recent_history']:
        history_string += f"User: {item['user_input']}\nAI: {item['ai_response']}\n"
    # --- END OF FIX ---
    
    context_string = f"""
    Current Entities: {context_summary['current_entities']}
    Current Intent: {context_summary['current_intent']}
    Knowledge Graph Data: {context_summary['kg_data']}
    
    Recent Conversation History:
    {history_string}
    """
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Context: {context_string}\n\nUser Query: {user_input}"}
    ]
    return messages

def chatbot_with_context(user_input: str, context: ConversationContext, model) -> str:
    system_message = generate_system_message()
    messages = prepare_messages(system_message, user_input, context)

    full_prompt = messages[0]['content'] + "\n\n" + messages[1]['content']
    
    try:
        
        response = model.generate_content(full_prompt)
        return response.text
    
    except Exception as e:
        logger.error(f"Error in LLM request: {e}")
        return "I apologize, but I encountered an error while processing your request. Please try again."

def chatbot_no_context(user_input: str, model) -> str:
    system_message = "Accurately help with the query, be precise and return only whats asked, no extra words."
    full_prompt = system_message + "\n\nUser Query: " + user_input
    
    try:
        # --- THIS IS THE CHANGE ---
        response = model.generate_content(full_prompt)
        return response.text
        # --- END OF CHANGE ---
    except Exception as e:
        logger.error(f"Error in LLM request: {e}")
        return "I apologize, but I encountered an error while processing your request. Please try again."
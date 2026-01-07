from typing import List, Dict, Any
from collections import deque
from datetime import datetime
import uuid

class ConversationContext:
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.current_entities = {
            "companies": [],
            "metrics": [],
            "limit": [],
            "startDate": [],
            "endDate": [],
            "industry": None
        }
        self.current_intent = {}
        self.kg_data = ""
        self.last_ai_response = ""

    def update(
        self,
        user_input: str,
        companies: List[str],
        metrics: List[str],
        start_date: List[str],
        end_date: List[str],
        industry: str,
        intent: Dict[str, Any],
        kg_response: str
    ):
        self.current_entities["companies"] = companies if companies else self.current_entities["companies"]
        self.current_entities["metrics"] = metrics if metrics else self.current_entities["metrics"]
        self.current_entities["startDate"] = start_date if start_date else self.current_entities["startDate"]
        self.current_entities["endDate"] = end_date if end_date else self.current_entities["endDate"]
        self.current_entities["industry"] = industry if industry else self.current_entities["industry"]
        self.current_intent = intent
        self.kg_data = kg_response

        self.history.append({
            "user_input": user_input,
            "entities": self.current_entities.copy(),
            "intent": self.current_intent.copy(),
            "kg_data": self.kg_data,
            "ai_response": self.last_ai_response
        })

    def update_ai_response(self, ai_response: str):
        self.last_ai_response = ai_response
        if self.history:
            self.history[-1]["ai_response"] = ai_response

    def get_context_summary(self) -> Dict[str, Any]:
        return {
            "current_entities": self.current_entities,
            "current_intent": self.current_intent,
            "recent_history": list(self.history),
            "kg_data": self.kg_data,
            "last_ai_response": self.last_ai_response
        }

class Conversation:
    def __init__(self, id=None, title=None):
        self.id = id or str(uuid.uuid4())
        self.title = title or "Untitled Conversation"
        self.messages = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.context = ConversationContext()
        self.add_message("assistant", "How can I help you with financial information today?")

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.updated_at = datetime.now()

        # Set title to first user message if no title is given
        if role == "user" and self.title == "Untitled Conversation":
            self.title = content[:20] + "..."  # Use the first 20 characters of the first user message as title

        if role == "user":
            # Update context with user input
            self.context.update(
                content,
                [],  # companies
                [],  # metrics
                [],  # startDate
                [],  # endDate
                None,  # industry
                {},  # intent
                ""
            )
        elif role == "assistant":
            self.context.update_ai_response(content)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "messages": self.messages,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "context": self.context.get_context_summary()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        conversation = cls(id=data['id'], title=data['title'])
        conversation.messages = data['messages']
        conversation.created_at = datetime.fromisoformat(data['created_at'])
        conversation.updated_at = datetime.fromisoformat(data['updated_at'])
        
        # Restore context
        context_data = data['context']
        conversation.context.current_entities = context_data['current_entities']
        conversation.context.current_intent = context_data['current_intent']
        conversation.context.history = deque(context_data['recent_history'], maxlen=conversation.context.max_history)
        conversation.context.kg_data = context_data['kg_data']
        conversation.context.last_ai_response = context_data['last_ai_response']

        return conversation

def save_conversation(conversation: Conversation, conversations: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    conversations[conversation.id] = conversation.to_dict()
    return conversations

def load_conversation(conversation_id: str, conversations: Dict[str, Dict[str, Any]]) -> Conversation:
    if conversation_id in conversations:
        return Conversation.from_dict(conversations[conversation_id])
    return None
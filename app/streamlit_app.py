import os
import sys
# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from typing import Dict, Any, List, Tuple
import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
from core.llm_provider import get_llm_client
from datetime import datetime
from recall_engine.database_manager import DatabaseManager
from recall_engine.conversation_manager import Conversation, ConversationContext, save_conversation, load_conversation
from recall_engine.nlp_processor import extract_entities_and_intent
from recall_engine.query_generator import QueryGenerator
from recall_engine.chatbot import chatbot_with_context
from dotenv import load_dotenv
from kg.kg_writer import KGWriter
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AURA_CONNECTION_URI = os.environ.get("AURA_CONNECTION_URI")
AURA_USERNAME = os.environ.get("AURA_USERNAME")
AURA_PASSWORD = os.environ.get("AURA_PASSWORD")
# LLM client will be initialized in FinWiseApp.__init__

def create_driver(uri: str, username: str, password: str):
    try:
        return GraphDatabase.driver(uri, auth=(username, password))
    except Exception as e:
        logger.error(f"Error creating driver: {e}")
        return None

class FinWiseApp:
    def __init__(self):
        self.driver = create_driver(AURA_CONNECTION_URI, AURA_USERNAME, AURA_PASSWORD)
        self.kg_writer = KGWriter(self.driver)
        self.model = get_llm_client()  # Auto-selects OpenRouter or Gemini
        self.query_generator = QueryGenerator(self.model)
        self.initialize_session_state()

    def initialize_session_state(self):
        if 'current_conversation' not in st.session_state:
            st.session_state.current_conversation = Conversation()
        if 'conversations' not in st.session_state:
            st.session_state.conversations = {}
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = DatabaseManager(self.driver)
            if st.session_state.db_manager.database_is_empty():
                st.error("The database is empty. Please add some data before using FinWise AI.")
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None
        if 'chart_data' not in st.session_state:
            st.session_state.chart_data = None

    def sidebar(self):
        with st.sidebar:
            st.title("FinWise AI")
            if st.button("🔄 New Conversation"):
                self.new_conversation()
            self.past_conversations()
            self.file_uploader()
            self.database_manager()
            self.recent_insights()
            self.help_section()

    def new_conversation(self):
        st.session_state.conversations = save_conversation(st.session_state.current_conversation, st.session_state.conversations)
        st.session_state.current_conversation = Conversation()
        st.rerun()

    def past_conversations(self):
        st.subheader("Past Conversations")
        conversations = sorted(
            st.session_state.conversations.values(),
            key=lambda x: datetime.fromisoformat(x['updated_at']),
            reverse=True
        )
        self.display_conversations(conversations[:10])
        if len(conversations) > 10:
            with st.expander("Show more"):
                self.display_conversations(conversations[10:])
        st.divider()

    def display_conversations(self, conversations):
        for conv in conversations:
            conv_id = conv['id']
            conv_title = f"{conv['title'][:20]}..."
            cols = st.columns([5, 1, 1])
            if cols[0].button(conv_title, key=f"conv_{conv_id}"):
                st.session_state.current_conversation = load_conversation(conv_id, st.session_state.conversations)
                st.rerun()
            if cols[1].button("✏️", key=f"rename_btn_{conv_id}", help="Rename"):
                self.rename_conversation(conv, conv_title)
            if cols[2].button("🗑️", key=f"delete_btn_{conv_id}", help="Delete"):
                self.delete_conversation(conv_id)

    def rename_conversation(self, conv: Dict[str, Any], conv_title: str):
        new_title = st.text_input(f"Rename {conv_title}:", value=conv['title'], key=f"rename_input_{conv['id']}")
        if st.button("Submit", key=f"submit_rename_{conv['id']}"):
            if new_title != conv['title'] and new_title.strip():
                conv['title'] = new_title.strip()
                st.session_state.conversations = save_conversation(Conversation.from_dict(conv), st.session_state.conversations)
                st.rerun()

    def delete_conversation(self, conv_id: str):
        if st.button("Confirm Delete", key=f"confirm_delete_{conv_id}"):
            del st.session_state.conversations[conv_id]
            st.rerun()

    def database_manager(self):
        with st.expander("📊 Database", expanded=False):
            new_uri = st.text_input("Neo4j Connection URI", value=AURA_CONNECTION_URI)
            new_username = st.text_input("Username", value=AURA_USERNAME)
            new_password = st.text_input("Password", value=AURA_PASSWORD, type="password")
            if st.button("Load Database"):
                self.load_database(new_uri, new_username, new_password)
            if st.session_state.db_manager:
                self.display_database_stats()

    def load_database(self, uri: str, username: str, password: str):
        try:
            new_driver = create_driver(uri, username, password)
            if new_driver is not None:
                st.session_state.db_manager = DatabaseManager(new_driver)
                stats = st.session_state.db_manager.get_database_stats()
                st.success("Database loaded successfully.")
                self.display_database_stats()
            else:
                st.error("Failed to connect to the database. Please check your credentials.")
        except Exception as e:
            st.error(f"Error loading database: {e}")

    def display_database_stats(self):
        stats = st.session_state.db_manager.get_database_stats()
        for key, value in stats.items():
            st.write(f"{key.capitalize()}: {value}")

    def recent_insights(self):
        st.subheader("Recent Insights")
        insight = self.generate_insight()
        st.info(insight)

    def generate_insight(self) -> str:
        prompt = "Fetch the latest data from our database and return a meaningful, bite-sized, and concise insight about the data. Return as short and concise a message as possible."
        return self.generate_standalone_insight(prompt)

    def generate_standalone_insight(self, prompt: str) -> str:
        insight = chatbot_with_context(prompt, ConversationContext(), self.model)
        return insight

    def help_section(self):
        with st.expander("ℹ️ Help / About", expanded=False):
            st.write("FinWise AI is your advanced financial assistant. Ask questions about companies, financial metrics, and market trends to get insightful answers backed by our comprehensive database.")
            st.write("You can:")
            st.write("- Compare companies and their financial metrics")
            st.write("- Analyze trends over time")
            st.write("- Get insights on specific industries")
            st.write("- Upload financial data for analysis")
            st.write("For the best experience, be specific in your questions and mention company names, metrics, and time periods when relevant.")

    def file_uploader(self):
        st.subheader("📁 Upload File")
        uploaded_file = st.file_uploader("Upload a TXT or CSV file", type=["txt", "csv"], key="file_uploader")
        if uploaded_file and uploaded_file != st.session_state.uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            self.handle_file_upload(uploaded_file)

    def main_chat_area(self):
        st.header(st.session_state.current_conversation.title)
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.current_conversation.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
        
        if st.session_state.chart_data is not None:
            self.display_chart(st.session_state.chart_data)
        
        user_input = st.chat_input("Type your message here...")
        if user_input:
            self.handle_user_input(user_input)

    def handle_file_upload(self, uploaded_file):
        file_contents = self.read_file(uploaded_file)
        with st.spinner("Processing file..."):
            response = self.process_file_contents(file_contents, uploaded_file.name)
        st.success(f"File processed: {uploaded_file.name}")
        st.session_state.current_conversation.add_message("user", f"Uploaded file: {uploaded_file.name}")
        st.session_state.current_conversation.add_message("assistant", response)
        st.session_state.conversations = save_conversation(st.session_state.current_conversation, st.session_state.conversations)
        st.rerun()

    def read_file(self, uploaded_file):
        if uploaded_file.type == "text/csv":
            return pd.read_csv(uploaded_file)
        return uploaded_file.getvalue().decode("utf-8")

    def process_file_contents(self, file_contents, filename):
        prompt = f"""
        Analyze the following file contents from {filename}:

        {file_contents[:2000]}  # Limit to first 2000 characters for brevity

        Identify any relevant information about companies, their metrics, and financial data.
        If you find relevant information that can be added to our database, list all of it and ask the user if they want to add it to the database.

        If no relevant information is found, please state that.
        """
        return self.process_user_input(prompt)

    def handle_user_input(self, user_input: str):
        st.session_state.current_conversation.add_message("user", user_input)
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = self.process_user_input(user_input)
            st.write(response)
        st.session_state.current_conversation.add_message("assistant", response)
        st.session_state.conversations = save_conversation(st.session_state.current_conversation, st.session_state.conversations)

    def process_user_input(self, user_input: str) -> str:
        entities, intent = extract_entities_and_intent(user_input)
        
        try:
            query, is_valid, explanation = self.query_generator.generate_and_validate_query(intent, entities)
            
            if is_valid:
                logger.info(f"Valid query generated: {query}")
                
                required_params = self.query_generator.extract_parameters_from_query(query)
                
                parameters = self._prepare_query_parameters(entities, required_params)
                
                logger.debug(f"Executing Query with Parameters: {parameters}")
                
                try:
                    kg_response = st.session_state.db_manager.execute_query(query, parameters)
                    
                    self._update_conversation_context(user_input, entities, intent, kg_response)
                    
                    # Process the kg_response to generate chart data if applicable
                    st.session_state.chart_data = self.process_chart_data(kg_response, intent)
                    
                except Exception as e:
                    logger.error(f"Neo4j query execution failed: {e}")
                    kg_response = f"I encountered an error while fetching the data: {str(e)}. Please try rephrasing your question or providing more specific information."
                    self._update_conversation_context(user_input, entities, intent, kg_response)
            
            else:
                logger.warning(f"Invalid query generated: {query}")
                logger.warning(f"Validation explanation: {explanation}")
                kg_response = f"I apologize, but I couldn't generate a valid query to answer your question. The issue was: {explanation}. Could you please rephrase or provide more details?"
        
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            kg_response = "I encountered an unexpected error while processing your question. Please try again or rephrase your query."
        
        ai_response = chatbot_with_context(user_input, st.session_state.current_conversation.context, self.model)
        st.session_state.current_conversation.context.update_ai_response(ai_response)
        return ai_response

    def _prepare_query_parameters(self, entities: Dict[str, List[str]], required_params: List[str]) -> Dict[str, Any]:
        parameters = {}
        for param in required_params:
            if param in entities and entities[param]:
                if param == "limit":
                    parameters[param] = int(entities[param][0])
                elif param in ["startDate", "endDate"]:
                    parameters[param] = entities[param][0]
                else:
                    parameters[param] = entities[param]
            else:
                parameters[param] = None
        return parameters

    def _update_conversation_context(self, user_input: str, entities: Dict[str, List[str]], intent: Dict[str, Any], kg_response: str):
        st.session_state.current_conversation.context.update(
            user_input=user_input,
            companies=entities.get("companies", []),
            metrics=entities.get("metrics", []),
            start_date=entities.get("startDate", ["latest"]),
            end_date=entities.get("endDate", []),
            industry=entities.get("industry", None),
            intent=intent,
            kg_response=str(kg_response)
        )

    def process_chart_data(self, kg_response, intent):
        # This method would process the kg_response and generate chart data based on the intent
        # For now, we'll return None as a placeholder
        return None

    def display_chart(self, chart_data):
        # This method would display a chart based on the chart_data
        # For now, we'll just display a placeholder message
        st.write("Chart placeholder: Data visualization would be displayed here")

def main():
    app = FinWiseApp()
    app.sidebar()
    app.main_chat_area()

if __name__ == "__main__":
    main()
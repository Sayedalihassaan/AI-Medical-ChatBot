import streamlit as st
from System.helper import embedding
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from System.strcture import system_prompt
import os
import time
from typing import Dict, List
import logging
import speech_recognition as sr
import pyttsx3
import threading
import io
import tempfile
from pydub import AudioSegment
from pydub.playback import play
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration constants
class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "medicalbot"
    MODEL_NAME = "gemini-1.5-flash"
    TEMPERATURE = 0.01
    RETRIEVAL_K = 10

class VoiceManager:
    """Handles voice input and output functionality"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.setup_tts()
    
    def setup_tts(self):
        """Configure text-to-speech engine"""
        try:
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            
            # Try to set a female voice for medical assistant
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', 150)  # Slower for medical content
            self.tts_engine.setProperty('volume', 0.8)
            
        except Exception as e:
            logger.error(f"TTS setup error: {e}")
    
    def speech_to_text(self, audio_data):
        """Convert speech to text using Google Speech Recognition"""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Use Google Speech Recognition
            text = self.recognizer.recognize_google(audio_data, language='en-US')
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio. Please try again."
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return "Speech recognition service error. Please try typing your question."
    
    def text_to_speech(self, text):
        """Convert text to speech"""
        try:
            # Remove markdown formatting for speech
            clean_text = self.clean_text_for_speech(text)
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                self.tts_engine.save_to_file(clean_text, tmp_file.name)
                self.tts_engine.runAndWait()
                return tmp_file.name
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    def clean_text_for_speech(self, text):
        """Clean text for better speech output"""
        import re
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'#{1,6}\s*(.*)', r'\1', text)  # Headers
        text = re.sub(r'•\s*', '', text)              # Bullet points
        text = re.sub(r'-\s*', '', text)              # Dashes
        
        # Replace medical disclaimer with shorter version
        text = re.sub(r'⚠️.*?medical advice\.', 
                     'Please consult a healthcare professional for medical advice.', 
                     text, flags=re.DOTALL)
        
        # Limit length for speech (first 500 characters)
        if len(text) > 500:
            text = text[:500] + "... For complete information, please read the full response."
        
        return text

def initialize_rag_chain():
    """Initialize the RAG chain with error handling"""
    try:
        # Initialize embedding
        embedding_model = embedding()
        
        # Initialize vector store
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=Config.INDEX_NAME, 
            embedding=embedding_model
        )
        retriever = docsearch.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": Config.RETRIEVAL_K}
        )
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            google_api_key=Config.GOOGLE_API_KEY
        )
        
        # Enhanced system prompt
        system_prompt_text = (
            "You are a knowledgeable medical assistant powered by The Gale Encyclopedia of Medicine (3rd Edition). "
            "Your role is to provide accurate, evidence-based medical information based solely on the provided context. "
            "\n\nGuidelines:\n"
            "- Analyze all provided context thoroughly before responding\n"
            "- Synthesize information across multiple sources when available\n"
            "- Provide clear, well-structured answers with appropriate medical terminology\n"
            "- If information is incomplete or unavailable in the context, clearly state 'I don't have enough information to answer this question based on the available sources.'\n"
            "- Include relevant details such as symptoms, causes, treatments, and prognosis when available\n"
            "- Maintain a professional, helpful tone\n"
            "- Always remind users to consult healthcare professionals for medical advice\n\n"
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_text), 
            ("human", "{input}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain
        
    except Exception as e:
        logger.error(f"Error initializing RAG chain: {str(e)}")
        st.error("Failed to initialize the medical bot. Please check your configuration.")
        return None

def validate_environment():
    """Validate required environment variables"""
    required_vars = ["GOOGLE_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        st.stop()

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
        <style>
        /* Main container styling */
        .main > div {
            padding-top: 2rem;
        }
        
        /* Chat container */
        .chat-container {
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        
        /* Voice controls styling */
        .voice-controls {
            display: flex;
            gap: 10px;
            margin: 10px 0;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .voice-button {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);
        }
        
        .voice-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4);
        }
        
        .voice-button:active {
            transform: translateY(0);
        }
        
        .recording {
            background: linear-gradient(135deg, #dc3545, #c82333) !important;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        /* Message styling */
        .chat-message {
            margin-bottom: 16px;
            padding: 12px 16px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            display: flex;
            align-items: flex-start;
            gap: 12px;
            animation: fadeIn 0.3s ease-in;
            position: relative;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .chat-message.user {
            background: linear-gradient(135deg, #0066cc, #004499);
            color: white;
            flex-direction: row-reverse;
            margin-left: 10%;
        }
        
        .chat-message.bot {
            background: linear-gradient(135deg, #f0f8ff, #e6f3ff);
            color: #1a1a1a;
            border-left: 4px solid #0066cc;
            margin-right: 10%;
        }
        
        /* Voice control button in messages */
        .message-voice-control {
            position: absolute;
            top: 8px;
            right: 8px;
            background: rgba(0,0,0,0.1);
            border: none;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }
        
        .message-voice-control:hover {
            background: rgba(0,0,0,0.2);
            transform: scale(1.1);
        }
        
        /* Avatar styling */
        .chat-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            border: 2px solid rgba(255,255,255,0.2);
            flex-shrink: 0;
        }
        
        /* Header styling */
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #0066cc, #004499);
            color: white;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,102,204,0.3);
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 600;
        }
        
        .header p {
            margin: 8px 0 0 0;
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        /* Input styling */
        .stTextInput > div > div > input {
            font-size: 16px;
            padding: 12px;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            background-color: white;
            color: black !important;
            transition: border-color 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #0066cc;
            box-shadow: 0 0 0 3px rgba(0,102,204,0.1);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #0066cc, #004499);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,102,204,0.3);
        }
        
        /* Audio player styling */
        .audio-player {
            margin: 10px 0;
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 8px;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        
        /* Loading spinner */
        .loading-spinner {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #0066cc;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Message content styling */
        .message-content {
            line-height: 1.6;
            word-wrap: break-word;
            flex: 1;
        }
        
        .message-content strong {
            font-weight: 600;
        }
        
        .message-content em {
            font-style: italic;
        }
        
        /* Timestamp styling */
        .timestamp {
            font-size: 0.8rem;
            opacity: 0.7;
            margin-top: 4px;
        }
        
        /* Voice status indicator */
        .voice-status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 14px;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { transform: translateX(100%); }
            to { transform: translateX(0); }
        }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with bot information"""
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=100)
        st.title("🏥 Medical Bot")
        
        st.markdown("---")
        
        # Voice settings
        with st.expander("🎤 Voice Settings"):
            st.markdown("**Voice Features:**")
            voice_enabled = st.checkbox("Enable Voice Output", value=True, key="voice_enabled")
            auto_play = st.checkbox("Auto-play Bot Responses", value=False, key="auto_play")
            
            st.markdown("**Speech Recognition:**")
            st.info("Click the microphone button to use voice input")
            
            st.markdown("**Text-to-Speech:**")
            st.info("Click the speaker button to hear bot responses")
        
        with st.expander("🤖 Model Information"):
            st.markdown("""
            **Technology Stack:**
            - **LLM:** Gemini-1.5-Flash
            - **Vector Store:** Pinecone
            - **Framework:** LangChain RAG
            - **Knowledge Base:** The Gale Encyclopedia of Medicine (3rd Edition)
            - **Voice:** Speech Recognition + Text-to-Speech
            
            **Capabilities:**
            - Medical condition descriptions
            - Symptom analysis
            - Treatment information
            - Diagnostic procedures
            - Voice interaction
            """)
        
        with st.expander("👨‍💻 Developer Info"):
            st.markdown("""
            **Sayed Ali Elsayed**  
            *AI Engineer & ML Specialist*
            
            Passionate about developing AI solutions for healthcare and medical applications.
            
            [🔗 LinkedIn Profile](https://www.linkedin.com/in/sayed-ali-482668262/)
            """)
        
        st.markdown("---")
        
        with st.expander("⚠️ Disclaimer"):
            st.warning("""
            This bot provides educational information only. 
            Always consult qualified healthcare professionals 
            for medical advice, diagnosis, or treatment.
            """)
        
        st.markdown("---")
        st.markdown("*© 2025 Sayed Ali Elsayed*")

def format_message_with_timestamp(message: str, role: str) -> str:
    """Format message with timestamp"""
    timestamp = time.strftime("%H:%M")
    return f"""
    <div class="message-content">{message}</div>
    <div class="timestamp">{timestamp}</div>
    """

def create_audio_player(audio_file_path):
    """Create an audio player for the response"""
    if audio_file_path and os.path.exists(audio_file_path):
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            b64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
            <div class="audio-player">
                <audio controls style="width: 100%;">
                    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            """
            return audio_html
    return ""

def render_chat_message(message: Dict, user_avatar: str, doctor_avatar: str, voice_manager: VoiceManager):
    """Render a single chat message with voice controls"""
    import html
    safe_message = html.escape(message["message"]).replace('\n', '<br>')
    
    if message["role"] == "user":
        st.markdown(f'''
            <div class="chat-message user">
                <img class="chat-avatar" src="{user_avatar}" alt="User"/>
                <div class="message-content">{safe_message}</div>
                <div class="timestamp">{time.strftime("%H:%M")}</div>
            </div>
        ''', unsafe_allow_html=True)
    else:
        # Create unique key for each bot message
        message_id = f"msg_{hash(message['message'])}"
        
        # Add voice control button for bot messages
        col1, col2 = st.columns([1, 0.1])
        
        with col1:
            st.markdown(f'''
                <div class="chat-message bot">
                    <img class="chat-avatar" src="{doctor_avatar}" alt="Medical Bot"/>
                    <div class="message-content">{safe_message}</div>
                    <div class="timestamp">{time.strftime("%H:%M")}</div>
                </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            if st.button("🔊", key=f"speak_{message_id}", help="Listen to response"):
                # Generate and play audio
                audio_file = voice_manager.text_to_speech(message["message"])
                if audio_file:
                    st.session_state[f"audio_{message_id}"] = audio_file
                    st.rerun()
        
        # Display audio player if audio was generated
        if f"audio_{message_id}" in st.session_state:
            audio_html = create_audio_player(st.session_state[f"audio_{message_id}"])
            if audio_html:
                st.markdown(audio_html, unsafe_allow_html=True)

def record_audio():
    """Record audio from microphone"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
            
            st.success("✅ Recording complete! Processing...")
            
            # Convert speech to text
            text = recognizer.recognize_google(audio, language='en-US')
            return text
            
    except sr.WaitTimeoutError:
        st.error("⏰ Recording timeout. Please try again.")
        return None
    except sr.UnknownValueError:
        st.error("❌ Could not understand audio. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error during recording: {e}")
        return None

def clean_html_from_text(text: str) -> str:
    """Remove HTML tags and clean up the text"""
    import re
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def get_bot_response(rag_chain, user_input: str) -> str:
    """Get response from the bot with error handling"""
    try:
        with st.spinner("🔍 Searching medical knowledge base..."):
            response = rag_chain.invoke({"input": user_input})
            bot_answer = response.get("answer", "I apologize, but I couldn't generate a response.")
            
            # Clean any HTML artifacts from the response
            bot_answer = clean_html_from_text(bot_answer)
            
            # Add medical disclaimer
            bot_answer += "\n\n⚠️ **Medical Disclaimer:** This information is for educational purposes only. Please consult a healthcare professional for medical advice."
            
            return bot_answer
            
    except Exception as e:
        logger.error(f"Error getting bot response: {str(e)}")
        return "I apologize, but I encountered an error while processing your question. Please try again."

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="Medical Bot | AI Healthcare Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Validate environment
    validate_environment()
    
    # Apply custom styling
    apply_custom_css()
    
    # Initialize voice manager
    if "voice_manager" not in st.session_state:
        st.session_state.voice_manager = VoiceManager()
    
    # Render sidebar
    render_sidebar()
    
    # Initialize RAG chain
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = initialize_rag_chain()
    
    if st.session_state.rag_chain is None:
        st.stop()
    
    # Header
    st.markdown("""
        <div class="header">
            <h1>🏥 Medical Bot</h1>
            <p>Your AI-powered medical knowledge assistant with voice interaction</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Avatar URLs
    doctor_avatar = "https://img.poki-cdn.com/cdn-cgi/image/quality=78,width=628,height=628,fit=cover,f=auto/92516ad387af67f26d5a217c5381a41f.png"
    user_avatar = "https://cdn-icons-png.freepik.com/512/6596/6596121.png"
    
    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []
        # Add welcome message
        welcome_msg = """
        Welcome to Medical Bot! 👋

        I'm here to help you with medical information sourced from The Gale Encyclopedia of Medicine. 
        You can ask me about:
        
        • Medical conditions and diseases
        • Symptoms and their meanings  
        • Treatment options and procedures
        • Diagnostic tests and procedures
        • Anatomy and physiology
        
        💡 **New Voice Features:**
        • Use the microphone button to ask questions by voice
        • Click the speaker button to hear my responses
        
        What would you like to know about today?
        """
        st.session_state.history.append({"role": "bot", "message": welcome_msg})
    
    # Voice input section
    st.markdown("### 🎤 Voice Input")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🎤 Start Recording", key="start_recording"):
            st.session_state.recording = True
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop & Process", key="stop_recording"):
            if st.session_state.get("recording", False):
                voice_text = record_audio()
                if voice_text:
                    st.session_state.voice_input = voice_text
                    st.session_state.recording = False
                    st.rerun()
    
    with col3:
        if st.session_state.get("voice_input"):
            st.info(f"🎤 Voice Input: {st.session_state.voice_input}")
            if st.button("✅ Use Voice Input", key="use_voice"):
                user_input = st.session_state.voice_input
                st.session_state.history.append({"role": "user", "message": user_input})
                
                # Get bot response
                bot_response = get_bot_response(st.session_state.rag_chain, user_input)
                st.session_state.history.append({"role": "bot", "message": bot_response})
                
                # Auto-play if enabled
                if st.session_state.get("auto_play", False):
                    audio_file = st.session_state.voice_manager.text_to_speech(bot_response)
                    if audio_file:
                        st.session_state.last_audio = audio_file
                
                # Clear voice input
                del st.session_state.voice_input
                st.rerun()
    
    # Text input section
    st.markdown("### ⌨️ Text Input")
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask your medical question:",
            placeholder="e.g., What is acoustic neuroma?",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("📤 Send", use_container_width=True)
    
    with col3:
        clear_button = st.button("🧹 Clear", use_container_width=True)
    
    # Handle text input
    if send_button or (user_input and st.session_state.get("last_input") != user_input):
        if user_input.strip():
            # Add user message
            st.session_state.history.append({"role": "user", "message": user_input})
            
            # Get bot response
            bot_response = get_bot_response(st.session_state.rag_chain, user_input)
            st.session_state.history.append({"role": "bot", "message": bot_response})
            
            # Auto-play if enabled
            if st.session_state.get("auto_play", False):
                audio_file = st.session_state.voice_manager.text_to_speech(bot_response)
                if audio_file:
                    st.session_state.last_audio = audio_file
            
            # Clear input and rerun
            st.session_state.last_input = user_input
            st.rerun()
    
    # Handle clear button
    if clear_button:
        st.session_state.history = []
        if "last_input" in st.session_state:
            del st.session_state.last_input
        if "voice_input" in st.session_state:
            del st.session_state.voice_input
        st.rerun()
    
    # Display chat history
    st.markdown("### 💬 Chat History")
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.history:
        render_chat_message(message, user_avatar, doctor_avatar, st.session_state.voice_manager)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick action buttons
    if len(st.session_state.history) <= 1:  # Only show if minimal chat history
        st.markdown("### 🚀 Quick Start")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🫀 Heart Conditions"):
                st.session_state.history.append({"role": "user", "message": "Tell me about common heart conditions"})
                st.rerun()
        
        with col2:
            if st.button("🧠 Neurological Disorders"):
                st.session_state.history.append({"role": "user", "message": "What are common neurological disorders?"})
                st.rerun()
        
        with col3:
            if st.button("💊 Medication Info"):
                st.session_state.history.append({"role": "user", "message": "How do I understand medication side effects?"})
                st.rerun()
    
    # Status indicators
    if st.session_state.get("recording", False):
        st.markdown("""
            <div class="voice-status">
                🎤 Recording... Speak now!
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
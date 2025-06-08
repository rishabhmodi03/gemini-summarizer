import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import os
import time

# --- Configure Google Generative AI ---
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key not found.")
    st.markdown("Please set it in Streamlit secrets (`.streamlit/secrets.toml`) as `GOOGLE_API_KEY` or as an environment variable.")
    st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {e}")
    st.stop()

# --- App Configuration ---
st.set_page_config(
    page_title="Multimodal Gemini Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Configuration ---
MODEL_NAME = "gemini-2.0-flash"

@st.cache_resource
def get_generative_model():
    """Caches the generative model instance."""
    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        return model
    except Exception as e:
        st.error(f"Failed to initialize the generative model '{MODEL_NAME}': {e}")
        st.stop()

model = get_generative_model()

# --- Session State Initialization ---
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful, creative, and friendly AI assistant. Respond concisely."

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# --- Helper Functions ---
def display_message(message):
    """Displays a message dictionary containing 'role' and 'parts'."""
    with st.chat_message(message["role"]):
        for part in message["parts"]:
            if isinstance(part, str):
                st.markdown(part)
            elif isinstance(part, Image.Image):
                st.image(part, caption="Uploaded Image", use_column_width=True)
            else:
                st.warning(f"Could not display part of type: {type(part)}")

def send_message_to_gemini(parts):
    """
    Sends a list of parts (text and PIL Images) to the Gemini model using the chat session.
    """
    try:
        # Initialize or reset chat session if needed
        if st.session_state.chat_session is None:
            # Start with system prompt as the first message
            initial_history = [
                {"role": "user", "parts": [st.session_state.system_prompt]},
                {"role": "model", "parts": ["Understood! How can I assist you?"]}
            ]
            st.session_state.chat_session = model.start_chat(history=initial_history)
            # Add initial system prompt exchange to display history
            if not st.session_state.messages:
                st.session_state.messages.append({"role": "user", "parts": [st.session_state.system_prompt]})
                st.session_state.messages.append({"role": "model", "parts": ["Understood! How can I assist you?"]})

        # Send user input parts to the chat session
        response = st.session_state.chat_session.send_message(parts, stream=True)

        # Display the model's streaming response
        full_response = ""
        with st.chat_message("model"):
            message_placeholder = st.empty()
            for chunk in response:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        return full_response

    except Exception as e:
        st.error(f"An API error occurred: {e}")
        error_message = f"Error: Could not get response from model. ({e})"
        st.session_state.messages.append({"role": "model", "parts": [error_message]})
        return None

def reset_chat():
    """Resets the chat session, clears history, and resets the file uploader."""
    st.session_state.chat_session = None
    st.session_state.messages = []
    st.session_state.uploader_key += 1
    st.rerun()

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("Adjust settings for the multimodal Gemini chat.")

    st.subheader("System Prompt")
    new_system_prompt = st.text_area(
        "Instruct the AI's behavior, persona, style, etc.",
        value=st.session_state.system_prompt,
        key="system_prompt_input",
        height=150
    )

    # Warn if system prompt changed and there are messages
    if new_system_prompt != st.session_state.system_prompt and len(st.session_state.messages) > 0:
        st.info("System prompt changed. The chat history will be reset on your next message to apply the new prompt.")
        st.session_state.system_prompt = new_system_prompt

    st.markdown("---")
    if st.button("🔄 Reset Chat", type="secondary", use_container_width=True):
        reset_chat()

    st.markdown("---")
    st.subheader("API Key")
    st.markdown("Ensure your Google AI Studio API key is correctly set.")
    st.markdown("[Get your API Key](https://aistudio.google.com/app/apikey)")
    st.markdown("---")
    st.markdown("App built with ❤️ using Streamlit & Google Gemini")

# --- Main Chat Interface ---
st.title("💬 Multimodal Gemini Chat")
st.write("Upload images or text files and enter text prompts to interact with Google's Gemini model.")

# Display previous messages
for message in st.session_state.messages:
    display_message(message)

# --- User Input Area ---
uploaded_files = st.file_uploader(
    "🖼️ Upload images or 📄 text files (optional)",
    type=["png", "jpg", "jpeg", "webp", "txt"],
    accept_multiple_files=True,
    key=st.session_state.uploader_key,
    help="Upload images or text files relevant to your question."
)

prompt_text = st.chat_input("Enter your message here...")

# Process user input
if prompt_text or uploaded_files:
    user_message_parts_for_display = []
    parts_for_gemini = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type
            file_name = uploaded_file.name
            try:
                if file_type.startswith('image/'):
                    img = Image.open(uploaded_file)
                    user_message_parts_for_display.append(img)
                    parts_for_gemini.append(img)
                elif file_type == 'text/plain':
                    file_content = uploaded_file.getvalue().decode('utf-8')
                    user_message_parts_for_display.append(f"**Content of `{file_name}`:**\n\n```\n{file_content}\n```")
                    parts_for_gemini.append(file_content)
                else:
                    st.warning(f"Skipping unsupported file type: {file_name} ({file_type})")
            except Exception as e:
                st.error(f"Could not process file {file_name}: {e}")

    if prompt_text:
        user_message_parts_for_display.append(prompt_text)
        parts_for_gemini.append(prompt_text)

    if parts_for_gemini:
        st.session_state.messages.append({"role": "user", "parts": user_message_parts_for_display})
        display_message({"role": "user", "parts": user_message_parts_for_display})

        with st.spinner("Thinking..."):
            model_response_text = send_message_to_gemini(parts_for_gemini)

        if model_response_text is not None:
            st.session_state.messages.append({"role": "model", "parts": [model_response_text]})

        st.session_state.uploader_key += 1
        st.rerun()

st.markdown("---")
st.markdown("This app demonstrates multimodal chat capabilities using Google's Gemini 1.5 Flash model.")
st.markdown("Images and text files uploaded in a turn are sent to the model along with the text prompt for that turn.")
st.markdown("Chat history is maintained within the session.")
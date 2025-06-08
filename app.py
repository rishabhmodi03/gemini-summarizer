import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import os
import time # To simulate loading if needed, or just for clarity

# --- Configure Google Generative AI ---
# Check for API key in Streamlit secrets or environment variables
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key not found.")
    st.markdown("Please set it in Streamlit secrets (`.streamlit/secrets.toml`) as `GOOGLE_API_KEY` or as an environment variable.")
    st.stop() # Stop the app if the API key is not found

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {e}")
    st.stop()

# --- App Configuration ---
st.set_page_config(
    page_title="Multimodal Gemini Chat",
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar open by default
)

# --- Model Configuration ---
# Use a model that supports multimodal input and stateful chat
# gemini-1.5-flash-latest is recommended for interleaved text/image history
MODEL_NAME = "gemini-1.5-flash-latest"

@st.cache_resource
def get_generative_model():
    """Caches the generative model instance to avoid re-initializing on each rerun."""
    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        # Optional: Add a small check to see if the model is accessible
        # model.generate_content("hello", stream=True)
        return model
    except Exception as e:
        st.error(f"Failed to initialize the generative model '{MODEL_NAME}': {e}")
        st.stop()

model = get_generative_model()

# --- Session State Initialization ---
# Initialize chat session object
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None

# Initialize list to store messages for display (includes images)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize system prompt state with a default value
if "system_prompt" not in st.session_state:
     st.session_state.system_prompt = "You are a helpful, creative, and friendly AI assistant. Respond concisely."

# Key to reset the file uploader widget
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
                # Display PIL Image directly
                st.image(part, caption="Uploaded Image", use_column_width=True)
            else:
                 # Handle unexpected part types gracefully
                 st.warning(f"Could not display part of type: {type(part)}")


def send_message_to_gemini(parts, system_instruction=None):
    """
    Sends a list of parts (text and PIL Images) to the Gemini model
    using the chat session. Handles initialization and system instruction changes.
    """
    try:
        # Check if chat session needs initialization or reset
        # A new session is needed if:
        # 1. There is no session yet.
        # 2. The system prompt has changed AND there is existing chat history.
        if st.session_state.chat_session is None or \
           (system_instruction != st.session_state.system_prompt and len(st.session_state.messages) > 0):

             if st.session_state.chat_session is not None: # If resetting existing chat
                 st.warning("System prompt changed. Resetting chat history to apply new instruction.")
                 st.session_state.messages = [] # Clear display history
             else:
                 st.info("Starting new chat session.")


             # Start a new chat session with the current system instruction
             st.session_state.chat_session = model.start_chat(
                 history=[], # Start with empty history when system prompt changes
                 system_instruction=system_instruction
             )
             # Update the stored system prompt
             st.session_state.system_prompt = system_instruction

        # Send the current user input parts to the existing or newly created session
        # The chat_session object automatically handles appending these to its internal history
        response = st.session_state.chat_session.send_message(parts, stream=True)

        # Display the model's streaming response
        full_response = ""
        with st.chat_message("model"):
            message_placeholder = st.empty() # Create an empty container to write the streaming response
            for chunk in response:
                full_response += chunk.text
                # Update the placeholder with the current accumulated response + cursor
                message_placeholder.markdown(full_response + "▌")

            # Final update to remove the cursor
            message_placeholder.markdown(full_response)

        # Return the full response text for storing in the display history
        return full_response

    except Exception as e:
        st.error(f"An API error occurred: {e}")
        # Add an error message to the display history
        error_message = f"Error: Could not get response from model. ({e})"
        st.session_state.messages.append({"role": "model", "parts": [error_message]})
        return None # Indicate that the response failed


def reset_chat():
    """Resets the chat session, clears history, and resets the file uploader."""
    st.session_state.chat_session = None
    st.session_state.messages = []
    # Increment uploader key to clear the file uploader widget
    st.session_state.uploader_key += 1
    st.rerun() # Rerun the app to apply the changes


# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("Adjust settings for the multimodal Gemini chat.")

    st.subheader("System Prompt")
    # Text area for system prompt, linked to session state value
    # The value updates st.session_state.system_prompt_input directly
    new_system_prompt = st.text_area(
        "Instruct the AI's behavior, persona, style, etc.",
        value=st.session_state.system_prompt, # Use session state as initial value
        key="system_prompt_input", # Unique key for the widget
        height=150
    )

    # Display a warning if the system prompt has changed and there are messages
    if new_system_prompt != st.session_state.system_prompt and len(st.session_state.messages) > 0:
         st.info("System prompt changed. The chat history will be reset on your next message to apply the new prompt.")


    st.markdown("---")

    if st.button("🔄 Reset Chat", type="secondary", use_container_width=True):
        reset_chat()

    st.markdown("---")
    st.subheader("API Key")
    st.markdown("Ensure your Google AI Studio API key is correctly set.")
    st.markdown("[Get your API Key](https://aistudio.google.com/app/apikey)")
    st.markdown("---")
    st.markdown("App built with ❤️ using Streamlit & Google Gemini")
    # Add your GitHub link here if you plan to open source
    # st.markdown("[Source Code on GitHub](https://github.com/your_username/your_repo_name)")


# --- Main Chat Interface ---
st.title("💬 Multimodal Gemini Chat")
st.write("Upload images and enter text prompts to interact with Google's Gemini model.")

# Display previous messages from session state
for message in st.session_state.messages:
    display_message(message)


# --- User Input Area ---
# File Uploader allows multiple image files
uploaded_files = st.file_uploader(
    "🖼️ Upload images (optional)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    key=st.session_state.uploader_key, # Use key to reset the widget
    help="Upload images relevant to your question. Maximum file size per image might apply."
)

# Chat input for text prompt
prompt_text = st.chat_input("Enter your message here...")


# Process user input when either text is entered or files are uploaded
# Note: st.chat_input returns the prompt when submitted.
# If only files are uploaded without text, the code block below won't execute
# unless we add a check for uploaded_files state changing.
# A simpler approach is to require text *or* files for submission.
# Let's make the submission happen when prompt_text is entered, and include files if present.
if prompt_text or uploaded_files: # Allow submitting with just files or just text or both

    # Prepare the user's message parts for display and sending to Gemini
    user_message_parts_for_display = []
    image_parts_for_gemini = [] # Store PIL Images for the model

    # Process uploaded files first
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Open image file using Pillow
                img = Image.open(uploaded_file)
                user_message_parts_for_display.append(img) # Add image to parts for display
                image_parts_for_gemini.append(img) # Add image to parts for Gemini API
            except Exception as e:
                st.error(f"Could not load image {uploaded_file.name}: {e}")
                # Optionally, continue without this image or stop


    # Add the text prompt if available
    if prompt_text:
        user_message_parts_for_display.append(prompt_text) # Add text to parts for display


    # --- Add user message to history and display ---
    # Only add if there's *something* to send (text or images)
    if user_message_parts_for_display:
        # Add the user's message to the session state history
        st.session_state.messages.append({"role": "user", "parts": user_message_parts_for_display})

        # Display the user's message immediately
        display_message({"role": "user", "parts": user_message_parts_for_display})

        # --- Prepare parts for Gemini API call ---
        # The API expects images followed by text if both are present in a turn
        parts_for_gemini = image_parts_for_gemini + ([prompt_text] if prompt_text else [])

        # --- Send message to Gemini and get response ---
        # Use a spinner while waiting for the response
        with st.spinner("Thinking..."):
             # The send_message_to_gemini function handles the chat session state
             model_response_text = send_message_to_gemini(parts_for_gemini, system_instruction=new_system_prompt)


        # --- Add model response to history ---
        if model_response_text is not None: # Only add if the API call was successful
             st.session_state.messages.append({"role": "model", "parts": [model_response_text]})

        # --- Rerun to update UI ---
        # Increment the uploader key to clear uploaded files visually
        st.session_state.uploader_key += 1
        # Force a rerun to clear the uploader and refresh the chat display
        st.rerun()

    elif uploaded_files and not prompt_text:
         # Handle case where files are uploaded but no text prompt is given
         st.warning("Please enter a text prompt along with the images, or click the upload button again after selecting files.")
         # Note: File uploader state is tricky to clear *without* rerun or button click.
         # The rerun above handles clearing it after submission. If no submission, it stays.


# Optional: Add some info at the bottom
st.markdown("---")
st.markdown("This app demonstrates multimodal chat capabilities using Google's Gemini 1.5 Flash model.")
st.markdown("Images uploaded in a turn are sent to the model along with the text prompt for that turn.")
st.markdown("Chat history is maintained within the session.")

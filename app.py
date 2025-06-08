import streamlit as st
import google.genai as genai
from PIL import Image
import io
import os
import time # Optional: for potential future loading indicators

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

# Initialize list to store messages for display (includes images and file text)
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
st.write("Upload images or text files and enter text prompts to interact with Google's Gemini model.")

# Display previous messages from session state
for message in st.session_state.messages:
    display_message(message)


# --- User Input Area ---
# File Uploader allows multiple image and text files
uploaded_files = st.file_uploader(
    "🖼️ Upload images or 📄 text files (optional)",
    type=["png", "jpg", "jpeg", "webp", "txt"], # Added 'txt'
    accept_multiple_files=True,
    key=st.session_state.uploader_key, # Use key to reset the widget
    help="Upload images or text files relevant to your question. Content will be included in the prompt."
)

# Chat input for text prompt
prompt_text = st.chat_input("Enter your message here...")


# Process user input when either text is entered or files are uploaded
if prompt_text or uploaded_files:

    # Prepare the user's message parts for display and sending to Gemini
    user_message_parts_for_display = []
    parts_for_gemini = [] # List to build parts for the API call

    # Process uploaded files first
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type
            file_name = uploaded_file.name

            try:
                if file_type.startswith('image/'):
                    # Handle images
                    img = Image.open(uploaded_file)
                    user_message_parts_for_display.append(img) # Add image for display
                    parts_for_gemini.append(img) # Add PIL Image object for Gemini
                elif file_type == 'text/plain':
                    # Handle text files
                    # Read content (decode using utf-8 is common for text files)
                    file_content = uploaded_file.getvalue().decode('utf-8')
                    # Add formatted text content for display
                    user_message_parts_for_display.append(f"**Content of `{file_name}`:**\n\n```\n{file_content}\n```")
                    # Add raw text content as a string part for Gemini
                    # Optionally add filename/context before content for Gemini too:
                    # parts_for_gemini.append(f"Content from file named {file_name}:\n")
                    parts_for_gemini.append(file_content)

                else:
                    # Handle other potential file types if needed
                    st.warning(f"Skipping unsupported file type: {file_name} ({file_type})")

            except Exception as e:
                st.error(f"Could not process file {file_name}: {e}")
                # Continue processing other files

    # Add the text prompt if available
    if prompt_text:
        user_message_parts_for_display.append(prompt_text) # Add text for display
        parts_for_gemini.append(prompt_text) # Add text for Gemini


    # --- Add user message to history and display ---
    # Only add if there's *something* to send (text, images, or file content)
    if parts_for_gemini: # Check if there's content prepared for Gemini
        # Add the user's message (including processed file content) to the session state history
        st.session_state.messages.append({"role": "user", "parts": user_message_parts_for_display})

        # Display the user's message immediately
        display_message({"role": "user", "parts": user_message_parts_for_display})

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
         # This case is hit if files are uploaded, but the prompt_text was empty when hitting enter.
         # We already processed the files and added to display history if successful,
         # and the user message was added *if* parts_for_gemini wasn't empty.
         # So here, we just need to potentially nudge the user if they didn't enter text,
         # but the file content itself serves as a prompt. Let's rethink this logic slightly.
         # The check `if parts_for_gemini:` before sending covers cases where file processing failed.
         # The main thing is to ensure the user sees the processed file content.
         # The display logic `display_message` handles showing the formatted file content.
         pass # The rest of the block handles adding/displaying if successful


# Optional: Add some info at the bottom
st.markdown("---")
st.markdown("This app demonstrates multimodal chat capabilities using Google's Gemini 1.5 Flash model.")
st.markdown("Images and text files uploaded in a turn are sent to the model along with the text prompt for that turn.")
st.markdown("Chat history is maintained within the session.")
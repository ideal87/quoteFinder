import streamlit as st
import time
import re
import os
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()

# Your assistant ID
ASSISTANT_ID = "asst_4IzrVrEAu46wXeBsmorjLSQC"

def interact_with_assistant(user_message):
    """
    Interact with the OpenAI assistant using beta threads.
    """
    for attempt in range(2):
        thread = None
        try:
            # Create a new thread for each attempt
            thread = client.beta.threads.create()
            st.write(f"Attempt {attempt+1}: Created thread {thread.id}")

            # Add message to the thread
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )

            # Create a run for the thread
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=ASSISTANT_ID
            )
            st.write(f"Run {run.id} created. Waiting...")

            # Increase timeout to 30 seconds and use exponential backoff
            timeout = 10
            start_time = time.time()
            while time.time() - start_time < timeout:
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )

                if run_status.status == "completed":
                    st.write(f"Run completed in {int(time.time()-start_time)}s")
                    messages = client.beta.threads.messages.list(thread.id)
                    for msg in messages.data:
                        if msg.role == "assistant":
                            return msg.content[0].text.value
                    return None

                # Check for terminal states
                if run_status.status in ["failed", "cancelled", "expired"]:
                    st.write(f"Run terminated with status: {run_status.status}")
                    break

                time.sleep(min(2**attempt, 4))  # Exponential backoff

            # Clean up thread if not completed
            if run_status.status != "completed":
                st.write(f"Deleting unfinished thread {thread.id}")
                client.beta.threads.delete(thread.id)
            
        except Exception as e:
            st.write(f"Attempt {attempt+1} error: {str(e)}")
            if thread:
                client.beta.threads.delete(thread.id)
            if attempt == 1:
                return None
        finally:
            if thread and run_status.status == "completed":
                try:
                    client.beta.threads.delete(thread.id)
                except Exception as e:
                    st.write(f"Error cleaning up thread: {str(e)}")

    return None

def process_srt(content):
    """
    Process the SRT content by replacing long quotes in the English lines
    with the assistant's output.
    """
    blocks = re.split(r'\n\n+', content.strip())
    processed_blocks = []
    total_blocks = len(blocks)
    
    for i, block in enumerate(blocks):
        lines = block.split('\n')
        if len(lines) < 3:
            # Skip blocks without at least header, timestamp, and content
            continue

        header, timestamp, *content_lines = lines
        # Assume first content line is Korean and the rest form the English line.
        korean = content_lines[0]
        english_line = ' '.join(content_lines[1:]).strip()

        if not english_line:
            continue

        st.write(f"Processing block {i+1}/{total_blocks}...")
        quotes = re.findall(r'"([^"]*)"', english_line)
        replacement_occurred = False

        for quote in quotes:
            if len(quote) < 40:
                st.write(f"Skipping short quote ({len(quote)} characters): {quote}")
                continue            

            replacement = interact_with_assistant(quote)
            if replacement:
                english_line = english_line.replace(f'"{quote}"', f'"{replacement}"')
                replacement_occurred = True

        if replacement_occurred:
            new_block = f"{header}\n{timestamp}\n{korean}\n{english_line}"
            processed_blocks.append(new_block)
        else:
            st.write(f"No replacements in block {i+1}, skipping.")

    return '\n\n'.join(processed_blocks)

def main():
    st.title("SRT Processor with OpenAI Assistant")
    st.write("Upload an SRT file to process quotes using the OpenAI assistant.")

    uploaded_file = st.file_uploader("Choose an SRT file", type=["srt"])
    
    if uploaded_file is not None:
        # Read file content as a string
        content = uploaded_file.read().decode('utf-8')
        st.text_area("File Content", content, height=300)
        
        if st.button("Process File"):
            with st.spinner("Processing..."):
                processed_content = process_srt(content)
            
            if processed_content:
                st.success("Processing complete!")
                st.download_button(
                    label="Download Processed SRT",
                    data=processed_content,
                    file_name="processed_reference.srt",
                    mime="text/plain"
                )
            else:
                st.error("No replacements were made. The processed file is empty or unchanged.")

if __name__ == "__main__":
    main()

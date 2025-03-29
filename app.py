import streamlit as st
from main import create_agent

# Set up the Streamlit app
st.title("AXA Health Assistant 🩺")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

agent = create_agent()

# Chat input
if prompt := st.chat_input("¿Cómo te sientes hoy?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        output_stream = agent.run(prompt, stream=True)
        msg = []
        for chunk in output_stream:
            st.markdown(chunk.content)
            msg.append(chunk.content)
        msg = ''.join(msg)
        #st.markdown(msg)
        
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": msg})
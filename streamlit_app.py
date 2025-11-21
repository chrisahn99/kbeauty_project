import streamlit as st
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
import os
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# --- Demo product constants ---
PRODUCT_NAME = "Jojoba Tea Tree Cream"
PRODUCT_IMAGE_URL = "https://raw.githubusercontent.com/chrisahn99/kbeauty_project/refs/heads/main/assets/IMG_3273.jpeg"

# --- Helper to display a nice product card ---
def show_product_card(button_key: str):
    with st.container(border=True):
        cols = st.columns([1, 2])
        with cols[0]:
            st.image(PRODUCT_IMAGE_URL, caption=None, use_container_width=True)
        with cols[1]:
            st.markdown(f"**{PRODUCT_NAME}**")
            st.caption("Hydrating cream with jojoba & tea tree — demo product.")
            st.write("Perfect for showcasing how the K-Beauty AI recommends products.")
            if st.button("Buy now", use_container_width=True, key=button_key):
                st.success("🛒 Demo only – checkout flow coming soon!")

# Set page config with title and favicon
st.set_page_config(
    page_title="K-Beauty AI Prototype",
    page_icon="https://raw.githubusercontent.com/chrisahn99/kbeauty_project/refs/heads/main/assets/logo.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

st.title("K-Beauty AI Prototype")
st.info("Check out the full presentation of this app in our homepage", icon="📃")

# Sidebar
st.sidebar.image(
    "https://raw.githubusercontent.com/chrisahn99/kbeauty_project/refs/heads/main/assets/logo.png",
    use_container_width=True
)
st.sidebar.write(
    """
Shop smarter with conversational AI.
Our platform uses an advanced matching engine, backed by vector search technology, to deliver precise product recommendations tailored to your needs.
"""
)

st.sidebar.header("How to use K-Beauty AI assistant")
st.sidebar.write(
    """
Freely engage in any way you want! The assistant is here to listen and assist you in any way possible.
"""
)

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello, how can I help you today?",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="K-Beauty AI assistant will be here shortly - hang tight!"):
        Settings.llm = TogetherLLM(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            api_key=st.secrets.together_key
        )

        Settings.embed_model = TogetherEmbedding(
            model_name="togethercomputer/m2-bert-80M-32k-retrieval",
            api_key=st.secrets.together_key
        )

        milvus_store = MilvusVectorStore(
            uri=st.secrets.zilliz_uri,
            collection_name="kbeauty_mvp_agent",
            token=st.secrets.milvus_key,
            dim=768
        )

        vector_index = VectorStoreIndex.from_vector_store(vector_store=milvus_store)

        return vector_index

index = load_data()
system_prompt = st.secrets.system_prompt

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        similarity_top_k=5,
        chat_mode="condense_plus_context",
        system_prompt=system_prompt,
        verbose=True,
        streaming=True
    )

# User input
if prompt := st.chat_input("Feel free to ask about anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Render chat history
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "assistant":
        with st.chat_message(
            message["role"],
            avatar="https://raw.githubusercontent.com/chrisahn99/kbeauty_project/refs/heads/main/assets/avatar.png"
        ):
            st.write(message["content"])

            # If the assistant message mentions the product, show the product card
            if PRODUCT_NAME in message["content"]:
                show_product_card(button_key=f"history_{i}")

    else:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.spinner("Generating response..."):
        with st.chat_message(
            "assistant",
            avatar="https://raw.githubusercontent.com/chrisahn99/kbeauty_project/refs/heads/main/assets/avatar.png"
        ):
            response_stream = st.session_state.chat_engine.stream_chat(prompt)

            # Stream the text answer
            st.write_stream(response_stream.response_gen)

            # Full text content after streaming
            full_response = response_stream.response

            # Show product card if product name appears
            if PRODUCT_NAME in full_response:
                show_product_card(button_key=f"live_{len(st.session_state.messages)}")

            # Save assistant message to history
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

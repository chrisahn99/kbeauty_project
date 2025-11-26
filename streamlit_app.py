import streamlit as st
import streamlit.components.v1 as components
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
import asyncio

# --- Demo product constants ---
PRODUCT_NAME = "Jojoba Tea Tree Cream"
PRODUCT_IMAGE_URL = "https://raw.githubusercontent.com/chrisahn99/kbeauty_project/refs/heads/main/assets/IMG_3273.jpeg"

# --- Helper to display a nice product card ---
PARENT_BASE_URL = "https://preview--glow-k-beauty-boutique.lovable.app"

def show_product_card(button_key: str):
    # Full absolute URL to the product page on your Lovable preview
    product_url = PARENT_BASE_URL + "/product/jojoba-tea-tree-cream"

    with st.container(border=True):
        cols = st.columns([1, 2])

        with cols[0]:
            st.image(PRODUCT_IMAGE_URL, width="stretch")

        with cols[1]:
            st.markdown(f"**{PRODUCT_NAME}**")
            st.caption("Hydrating cream with jojoba & tea tree — demo product.")
            st.write("Perfect for showcasing how the K-Beauty AI recommends products.")

            # HTML button that opens the Lovable preview product page
            button_html = (
                f'<a href="{product_url}" target="_top" style="text-decoration: none;">'
                '<button style="'
                'width: 100%; '
                'padding: 0.7rem 1.2rem; '
                'border-radius: 8px; '
                'border: 2px solid #ff66b3; '      /* pink outline */
                'background: transparent; '         /* transparent default */
                'color: #ff66b3; '                  /* pink text */
                'font-weight: 600; '
                'cursor: pointer; '
                'transition: all 0.25s ease; '      /* smooth animation */
                '" '
                'onmouseover="this.style.background=\'#ff66b3\'; this.style.color=\'white\';" '
                'onmouseout="this.style.background=\'transparent\'; this.style.color=\'#ff66b3\';"'
                '>'
                'Buy now'
                '</button>'
                '</a>'
            )

            st.markdown(button_html, unsafe_allow_html=True)
                
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
    width="stretch"
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

async def _load_data_async():
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


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="K-Beauty AI assistant will be here shortly - hang tight!"):
        return asyncio.run(_load_data_async())

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

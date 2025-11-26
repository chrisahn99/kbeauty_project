import streamlit as st
import streamlit.components.v1 as components
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
import asyncio

# --- Base URL for Lovable preview ---
PARENT_BASE_URL = "https://preview--glow-k-beauty-boutique.lovable.app"

# --- Product catalog ---
PRODUCTS = [
    {
        "name": "Jojoba Tea Tree Cream",
        "image_url": "https://raw.githubusercontent.com/chrisahn99/kbeauty_project/refs/heads/main/assets/IMG_3273.jpeg",
        "slug": "jojoba-tea-tree-cream",
    },
    {
        "name": "Zero Topia Cream",
        "image_url": "https://raw.githubusercontent.com/chrisahn99/kbeauty_project/refs/heads/main/assets/Screenshot%202025-10-10%20104315.png",
        "slug": "zero-topia-cream",
    },
    {
        "name": "Mooncat Real Green Tea Pore Deep Cleanser",
        "image_url": "https://raw.githubusercontent.com/chrisahn99/kbeauty_project/refs/heads/main/assets/Screenshot%202025-10-10%20104152.png",
        "slug": "mooncat-real-green-tea-pore-deep-cleanser",
    },
    {
        "name": "Quick Glow Bubble Serum",
        "image_url": "https://raw.githubusercontent.com/chrisahn99/kbeauty_project/refs/heads/main/assets/Screenshot%202025-10-10%20104629.png",
        "slug": "quick-glow-bubble-serum",
    },
    {
        "name": "Fluid Calming Pad",
        "image_url": "https://raw.githubusercontent.com/chrisahn99/kbeauty_project/refs/heads/main/assets/Screenshot%202025-10-10%20104650.png",
        "slug": "fluid-calming-pad",
    },
    {
        "name": "Jelly Stick Tint",
        "image_url": "https://raw.githubusercontent.com/chrisahn99/kbeauty_project/refs/heads/main/assets/Screenshot%202025-10-10%20104445.png",
        "slug": "jelly-stick-tint",
    },
]


# --- Helper to display a nice product card ---
def show_product_card(product: dict, button_key: str):
    product_url = f"{PARENT_BASE_URL}/product/{product['slug']}"

    with st.container(border=True):
        cols = st.columns([1, 2])

        with cols[0]:
            st.image(product["image_url"], use_container_width=True)

        with cols[1]:
            st.markdown(f"**{product['name']}**")
            st.caption("Hydrating, skin-loving Korean beauty pick.")
            st.write("Perfect for showcasing how the K-Beauty AI recommends products.")

            # HTML button that opens the Lovable preview product page
            button_html = (
                f'<a href="{product_url}" target="_top" style="text-decoration: none;">'
                '<button style="'
                'width: 100%; '
                'padding: 0.7rem 1.2rem; '
                'border-radius: 8px; '
                'border: 2px solid #ff66b3; '
                'background: transparent; '
                'color: #ff66b3; '
                'font-weight: 600; '
                'cursor: pointer; '
                'transition: all 0.25s ease; '
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
    use_container_width=True,
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

            # Show cards for any products mentioned in this message
            for product in PRODUCTS:
                if product["name"] in message["content"]:
                    show_product_card(product, button_key=f"history_{i}_{product['slug']}")

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

            # Show product cards if their names appear in the response
            for product in PRODUCTS:
                if product["name"] in str(full_response):
                    show_product_card(product, button_key=f"live_{len(st.session_state.messages)}_{product['slug']}")

            # Save assistant message to history
            message = {"role": "assistant", "content": str(full_response)}
            st.session_state.messages.append(message)

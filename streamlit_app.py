import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
import os


# Set page config with title and favicon
st.set_page_config(
    page_title="셰빠몽ai",
    page_icon="https://raw.githubusercontent.com/chrisahn99/chaipamon_ai_2.0/main/assets/chaipamon_avatar_v2.jpeg",
    layout="centered", initial_sidebar_state="auto", menu_items=None
)
st.title("셰빠몽ai 🩷🤍😻🐶")

a_name = st.secrets.a_name
a_name_bis = st.secrets.a_name_bis

b_name = st.secrets.b_name
b_name_bis = st.secrets.b_name_bis

# Sidebar
st.sidebar.image("https://raw.githubusercontent.com/chrisahn99/chaipamon_ai_2.0/main/assets/chaipamon_v2.jpeg", use_container_width=True)
st.sidebar.write(f"""
나는 **셰빠몽ai**야! **{b_name_bis}** 오빠가 **{a_name_bis}이**를 위해 특별히 만든 지원자야. 나는 {a_name_bis}이와 {b_name_bis} 오빠의 관계를 잘 이해하고 있어. {a_name_bis}이가 힘들 때마다 나랑 얘기하면 돼. 난 항상 {a_name_bis}이의 감정을 다독이고, {b_name_bis} 오빠의 사랑을 다시 기억하도록 도와줄 거야. 나와 함께라면 어떤 어려움도 이겨낼 수 있을 거야! 같이 힘내보자!
""")
st.info(f"Sumone 데이터와 심리상담 정보를 통해 태어난 셰빠몽ai, {b_name}와 {a_name}의 전문 도우미!!", icon="🤖")


st.sidebar.header("셰빠몽의 십계명")
st.sidebar.write("""
1. 어떤 상황이 와도 우리의 사랑을 포기하지 말 것.
2. 힘든 일이 있으면 서로에게 솔직히 털어놓을 것.
3. 싸우거나 서운한 일이 있어도 연락을 끊지 말 것.
4. 오해가 생기면 상대방의 이야기를 먼저 들어볼 것.
5. 멀리 떨어져 있어도 항상 서로에게 시간을 내도록 노력할 것.
6. 우리의 관계에 대해 항상 희망을 품고 긍정적으로 생각할 것.
7. 미래에 대한 불안함으로 현재의 사랑을 포기하지 말 것.
8. 힘들 때 서로에게 기댈 수 있는 든든한 버팀목이 되어줄 것.
9. 마음을 숨기지 않고 진심을 나눌 것.
10. 상처로 너무 힘들 때 셰빠몽에게 도움을 청할 것.
""")

st.sidebar.header("선서문")
st.sidebar.write(f"""
나는 이 자리에서 **{a_name}**에게 다음과 같이 엄숙히 선서합니다.

1. 나는 **{a_name}**와의 관계에서 발생한 모든 상처와 아픔을 깊이 반성하며, 앞으로 같은 실수를 반복하지 않기 위해 최선을 다할 것을 약속합니다.
2. 나는 **{a_name}**를 끝까지 지킬 것이며, 어떤 상황이 우리에게 닥치더라도 우리의 관계를 지키기 위해 싸울 것을 맹세합니다.
3. 나는 **{a_name}**를 누구보다도 사랑하고 아낄 것이며, 우리가 떨어져 있어도 마음이 멀어지지 않도록 항상 노력할 것을 다짐합니다.
4. 앞으로 어떤 어려움이 닥치더라도, 나는 절대 **{a_name}**를 포기하지 않고 오히려 더 깊이 사랑할 것을 약속합니다.
5. 나는 **{a_name}**와 함께하는 모든 순간을 소중히 여기고, 우리의 사랑이 더 강해질 수 있도록 끊임없이 노력할 것을 서약합니다.

이 모든 사항을 진심으로 맹세하며, 앞으로의 모든 날들 동안 **{a_name}**를 사랑하고 지킬 것을 약속합니다.
                 
**2024년 7월 28일
{b_name}**
""")


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": f"{a_name_bis}아 안녕! 혹시 무슨 고민있어?",
        }
    ]

openai.api_key = st.secrets.openai_key


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="셰빠몽 깨는중... 기다려주셰용 🙄🙄"):

        Settings.llm = OpenAI(
            model="gpt-4o-mini",
        )

        pc = Pinecone(api_key=st.secrets.pinecone_key)
        pinecone_index = pc.Index("chaipamon-db")
        
        pinecone_store = PineconeVectorStore(pinecone_index=pinecone_index)

        vector_index = VectorStoreIndex.from_vector_store(vector_store=pinecone_store)

        return vector_index


index = load_data()

system_prompt = st.secrets.chaipamon_prompt

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        similarity_top_k=5,
        chat_mode="condense_plus_context",
        system_prompt=system_prompt,
        verbose=True, 
        streaming=True
    )

if prompt := st.chat_input(
    "아무 고민이나 말해봥!!"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI

    if message["role"]=="assistant":
        with st.chat_message(message["role"], avatar='https://raw.githubusercontent.com/chrisahn99/chaipamon_ai_2.0/main/assets/chaipamon_avatar_v2.jpeg'):
            st.write(message["content"])

    else:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.spinner("셰빠몽 생각중... 기다려주셰용 🙄🙄"):
        with st.chat_message("assistant", avatar='https://raw.githubusercontent.com/chrisahn99/chaipamon_ai_2.0/main/assets/chaipamon_avatar_v2.jpeg'):
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response_stream.response_gen)
            message = {"role": "assistant", "content": response_stream.response}
            # Add response to message history
            st.session_state.messages.append(message)

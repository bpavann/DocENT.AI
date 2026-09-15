import streamlit as st
import uuid
from src.deep_agent_docent.dagents import DeepAgent

st.set_page_config(
    page_title="DocENTAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    width: 260px;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

.chat-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    margin-top: 28vh;
    margin-bottom: 10px;
}

.chat-subtitle {
    text-align: center;
    font-size: 1.05rem;
    color: #777;
    max-width: 700px;
    margin: auto;
    line-height: 1.6;
}

.thread-item {
    padding: 8px 10px;
    border-radius: 8px;
    margin-bottom: 4px;
}

.thread-item:hover {
    background-color: #f0f2f6;
}

</style>
""", unsafe_allow_html=True)

if "deep_agent" not in st.session_state:
    st.session_state.deep_agent = DeepAgent()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "threads" not in st.session_state:
    st.session_state.threads = []

with st.sidebar:
    st.markdown("## 🧠 DocENTAI")
    st.caption("Document Intelligence Assistant")
    st.divider()

    st.markdown("### 💬 Chat History")

    if st.session_state.threads:
        for thread in st.session_state.threads:
            col1, col2 = st.columns([5, 1])

            with col1:
                if st.button(
                    thread["title"],
                    key=f"thread_{thread['id']}",
                    use_container_width=True
                ):
                    st.session_state.thread_id = thread["id"]
                    st.session_state.messages = thread["messages"]
                    st.rerun()

            with col2:
                if st.button(
                    "🗑️",
                    key=f"delete_{thread['id']}"
                ):
                    st.session_state.threads = [
                        t for t in st.session_state.threads
                        if t["id"] != thread["id"]
                    ]

                    if st.session_state.thread_id == thread["id"]:
                        st.session_state.thread_id = str(uuid.uuid4())
                        st.session_state.messages = []

                    st.rerun()
    else:
        st.caption("No previous conversations.")

    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        if st.session_state.messages:

            existing_thread = next(
                (
                    t for t in st.session_state.threads
                    if t["id"] == st.session_state.thread_id
                ),
                None
            )

            if existing_thread:
                existing_thread["messages"] = st.session_state.messages
            else:
                st.session_state.threads.append({
                    "id": st.session_state.thread_id,
                    "title": st.session_state.messages[0]["content"][:35],
                    "messages": st.session_state.messages
                })

        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

if not st.session_state.messages:
    st.markdown(
        "<div class='chat-title'>DocENTAI</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class='chat-subtitle'>
        🧠 DocENTAI is a document intelligence assistant that helps you
        understand, analyze, summarize, extract, and generate insights
        from your documents using specialized AI agents and LLM-powered
        workflows.
        <br><br>
        Ask a question about your documents to get started.
        </div>
        """,
        unsafe_allow_html=True
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            documents = message.get("documents", "")

            if documents:
                with st.expander("📚 Retrieved Documents"):
                    st.markdown(documents)

user_query = st.chat_input(
    "Ask DocENTAI anything about your documents..."
)

if user_query:
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.deep_agent.invoke(
                    user_query,
                    st.session_state.thread_id
                )

                print("GRAPH RESULT:")
                print(result)

                agent_result = result.get("agent_result", {})

                response = agent_result.get(
                    "result",
                    "No response was returned."
                )

                st.markdown(response)

                documents = result.get("documents", "")

                if documents:
                    with st.expander("📚 Retrieved Documents"):
                        st.markdown(documents)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "documents": documents
                })

            except Exception as e:

                response = f"❌ Error: {str(e)}"

                st.error(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

    existing_thread = next(
        (
            t for t in st.session_state.threads
            if t["id"] == st.session_state.thread_id
        ),
        None
    )

    if existing_thread:
        existing_thread["messages"] = st.session_state.messages
    else:
        st.session_state.threads.append({
            "id": st.session_state.thread_id,
            "title": user_query[:35],
            "messages": st.session_state.messages
        })

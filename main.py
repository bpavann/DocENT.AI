# web/app.py
import streamlit as st
import uuid
from docent_client import run_query_with_mcp, preprocess_file
from agents.automation_agent import AutomationAgent

st.set_page_config(layout="wide", page_title="DocENT AI (MCP)")
st.title("📄 DocENT AI — MCP-backed")

uploaded_file = st.file_uploader("Upload document (pdf/docx/csv/html/txt/md)", type=["pdf", "docx", "csv", "html", "htm", "txt", "md"])
user_query = st.text_area("Enter your query", height=160)

export_formats = st.multiselect("Export formats", ["docx", "pdf", "csv", "pptx"], default=["docx", "pdf"])

if st.button("Run DocENT AI"):
    if not uploaded_file:
        st.error("Please upload a file first.")
    elif not user_query:
        st.error("Please enter a query.")
    else:
        with st.spinner("Preprocessing locally (ingest → preprocess → chunk → embed)..."):
            chunks, embeddings = preprocess_file(uploaded_file)
            st.success(f"Preprocessing finished — {len(chunks)} chunks created (showing up to 5).")
            for i, c in enumerate(chunks[:5], start=1):
                st.markdown(f"**Chunk {i}:** {c[:400]}")

        with st.spinner("Calling MCP-backed agent..."):
            try:
                result = run_query_with_mcp(uploaded_file, user_query)
                st.subheader("Agent Response")
                # result from create_agent may be a dict with messages
                if isinstance(result, dict) and result.get("messages"):
                    last_msg = result["messages"][-1]
                    content = last_msg.get("content") if isinstance(last_msg, dict) else str(last_msg)
                    st.write(content)
                    text_to_export = content
                else:
                    st.write(result)
                    text_to_export = str(result)

                if export_formats:
                    exporter = AutomationAgent()
                    exporter.export(text_to_export, export_formats)
                    st.success("Exported selected formats to working directory.")
            except Exception as e:
                st.error(f"Error calling agent: {e}")

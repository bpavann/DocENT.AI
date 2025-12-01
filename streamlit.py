from turtle import width
import streamlit as st
from datetime import datetime
import json
import pandas as pd
from typing import Dict, Any
import uuid
from deep_agent import DeepAgent, SubAgentType

st.set_page_config(
    page_title="DocENT.AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FINAL CSS ---
st.markdown("""
<style>

:root {
    --primary-color: #667eea;
    --text-dark: #1a1a1a;
    --light-bg: #f8f9fa;
    --badge-radius: 18px;
}

/* Center alignment for intro text */
.center-text {
    text-align: center;
    font-size: 1.05rem;
    line-height: 1.6;
}

/* Title improvements */
.big-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 900;
    margin-top: -20px;
    margin-bottom: 10px;
}

/* Section header */
.section-header {
    color: var(--primary-color);
    font-size: 1.35em;
    font-weight: bold;
    margin-top: 25px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--primary-color);
}

/* Result container box */
.result-container {
    background-color: var(--light-bg);
    padding: 20px;
    border-radius: 12px;
    margin: 18px 0;
    border-left: 5px solid var(--primary-color);
    box-shadow: 0px 0px 10px rgba(0,0,0,0.03);
}

/* Debug container */
.debug-box {
    background-color: #eef2ff;
    padding: 18px;
    border-radius: 10px;
    border-left: 4px solid #4c51bf;
    margin-top: 20px;
}

/* Status badges */
.status-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: var(--badge-radius);
    font-weight: 600;
    font-size: 0.85em;
    margin: 4px;
}

.status-completed { background: #d4edda; color: #155724; }
.status-skipped   { background: #fff3cd; color: #856404; }
.status-error     { background: #f8d7da; color: #721c24; }

/* Table styling refinement */
table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 12px;
}
table th, table td {
    padding: 8px;
    border: 1px solid #ddd;
}
table th {
    background-color: #ebeeff;
    font-weight: 700;
}

/* Clean bullet spacing */
ul {
    margin: 0 0 10px 0;
    padding-left: 20px;
}

</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "deep_agent" not in st.session_state:
    st.session_state.deep_agent = DeepAgent()

if "current_result" not in st.session_state:
    st.session_state.current_result = None

# --- TITLE & INTRO ---
st.markdown("""
<h1 style='text-align: center;'>🤖 DocENT AI</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 17px;'>
🧠 DocENT.AI helps you explore and understand your Social Network and Big Data Analytics content by analyzing your documents and generating clear, context-aware answers to your queries.
</div>
""", unsafe_allow_html=True)
st.divider()
st.markdown("""
### 🔍 System Output Includes
- **Clear, structured response**
- **Relevant context extracted from your documents**
- **Optional tables or highlighted insights**
- **Execution summary for transparency**

*Debug information (agent output, retrieval logs, sub-agent traces) is available at the end.*
""")
# --- USER QUERY INPUT ---
user_query = st.text_area(
    "Enter your query:",
    placeholder="e.g., Summarize about social network and big data analytics ?",
    height=100,
    key="user_query_input"
)

submit_btn = st.button("🚀 Analyze")

if submit_btn:
    if not user_query.strip():
        st.warning("⚠️ Please enter a query!")
    else:
        with st.spinner("🔄 Analyzing your query..."):
            try:
                thread_id = str(uuid.uuid4())
                result = st.session_state.deep_agent.invoke(user_query, thread_id)
                st.session_state.current_result = result
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# --- DISPLAY RESULTS ---
if st.session_state.current_result:
    result = st.session_state.current_result
    
    st.markdown('<div class="section-header">📊 Query Results</div>', unsafe_allow_html=True)
    
    # Original query
    st.markdown("**📌 Original Query:**")
    st.info(result.get("user_query", "No query recorded"))
    
    st.divider()
    
    # Color-coded badges
    st.markdown("**Status Indicators:**")
    cols = st.columns(len(result.get("approval_status", {})))
    for (agent_name, status), col in zip(result.get("approval_status", {}).items(), cols):
        with col:
            if status == "completed":
                st.markdown(f"<div class='status-badge status-completed'>✅ {agent_name.upper()}</div>", unsafe_allow_html=True)
            elif status == "skipped":
                st.markdown(f"<div class='status-badge status-skipped'>⏭️ {agent_name.upper()}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='status-badge status-error'>❌ {agent_name.upper()}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Subagent results
    st.markdown("**📋 Subagent Results:**")
    subagent_results = result.get("subagent_results", {})
    if subagent_results:
        for agent_name, agent_result in subagent_results.items():
            status = result.get("approval_status", {}).get(agent_name, "unknown")
            
            if status == "skipped":
                with st.expander(f"⏭️ {agent_name.upper()} - Skipped"):
                    st.info("This agent was not required for your query.")
            elif agent_result:
                with st.expander(f"✅ {agent_name.upper()}", expanded=True):
                    if isinstance(agent_result, str):
                        st.write(agent_result)
                    else:
                        st.json(agent_result)
            else:
                with st.expander(f"⚠️ {agent_name.upper()} - No Output"):
                    st.warning("No output generated by this agent.")
    
    st.divider()
    
    # DEBUG / SESSION STATE
    st.markdown('<div class="section-header">🐞 Debug Info</div>', unsafe_allow_html=True)
    st.json({
        "query": result.get("user_query"),
        "required_agents": [
            a.value if hasattr(a, "value") else str(a)
            for a in result.get("required_agents", [])
        ],
        "approval_status": result.get("approval_status", {}),
        "analysis_complete": result.get("analysis_complete", False),
        "subagent_results": result.get("subagent_results", {}),
        "messages": [msg.content if hasattr(msg, "content") else str(msg) for msg in result.get("messages", [])]
    })

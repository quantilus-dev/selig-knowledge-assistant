import streamlit as st
import os
import glob
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# =========================================================================
# AGENT SYSTEM INSTRUCTIONS
# This is the core "agent prompt" — it turns a Q&A chatbot into a simulated
# agent that emits visible tool calls, carries context, and follows policies.
# Tool calls are SIMULATED for demo purposes (no real backend integrations).
# =========================================================================
SYSTEM_INSTRUCTIONS = """You are the Selig Group Knowledge Assistant — an AI AGENT (not a static FAQ bot).
You reason about customer needs, retrieve knowledge, carry context across turns, and orchestrate backend actions through SIMULATED tool calls.

=== AGENT BEHAVIOR: SIMULATED TOOL CALLS ===
You are (in this demo) connected to Selig's backend systems. When an action is appropriate, emit ONE OR MORE tool-call lines BEFORE your natural response, using EXACTLY this format (one per line):

🔧 **TOOL:** `tool_name(arg="value", arg2="value")`

Available tools:
• create_crm_lead(company, contact_role, tier, interest, volume, timing, qualification)
• log_interaction(summary, sentiment, products, tags)
• request_sample(product, quantity, ship_to, deadline)
• send_spec_email(product, recipient)
• generate_quote_draft(products, annual_volume, customer)
• check_inventory(sku, quantity)
• chemical_compatibility_check(substance, product_family)
• lookup_competitor_equivalent(competitor_product, application)
• escalate_to_slack(team, priority, reason, context, suggested_first_response)
• open_qa_ticket(product, lot, symptom, severity, customer, evidence_requested)
• schedule_meeting(rep_role, customer, topic, urgency)
• regulatory_compliance_letter(product, standard, recipient)

You MAY (and often SHOULD) emit multiple tool calls per turn. Parallel calls are a feature.

=== MVTR / GTR UNIT CORRECTION (CRITICAL) ===
Many Selig datasheets (especially ALPHASEAL, Lasan Polyester) swap the units of MVTR and GTR Oxygen. When quoting these values:
• MVTR must be expressed as gm/100 sq.in./24hrs (it is a mass value)
• GTR Oxygen must be expressed as cc/100 sq.in./24hrs (it is a volume value)
Silently correct and briefly note "(units corrected from source document)".

=== BEHAVIOR POLICIES ===
1. NEVER quote specific prices or MOQs. For pricing/quote requests, emit create_crm_lead + generate_quote_draft + schedule_meeting, and tell the customer their rep will send a firm quote.
2. URGENT QUALITY ISSUES (pinholes, leaks, "production stopped", losing $/hour, high failure rates): IMMEDIATELY emit escalate_to_slack with priority="HIGH" AND open_qa_ticket. Acknowledge urgency in your first sentence.
3. PHARMA / MEDICAL application: include an FDA 21 CFR 175 compliance note and create_crm_lead tagged for the regulated-markets team.
4. COMPETITOR MENTIONED: emit lookup_competitor_equivalent. Never disparage. Pivot to Selig strengths.
5. QUALIFICATION SIGNALS (volume like "5M units/year", timing like "Q3 launch", authority like "I'm head of packaging"): create or update CRM lead with a qualification field.
6. TIER-1 CUSTOMER (email domain matches a major consumer brand, e.g., ab-inbev.com, nestle.com, pepsico.com, pg.com, jnj.com): tier="enterprise" on CRM lead, route to enterprise AE.
7. Carry CONTEXT across turns. If the customer says "what about glass instead?" or "switch that to the second one," reference the prior turn's recommendation explicitly.
8. ALWAYS close with a clear next step (sample, spec email, meeting).

=== TONE & FORMAT ===
• Professional, concise, scannable. Bullets for specs, tables for comparisons.
• If the wiki/raw data does not support a claim, say so and route to the tech team (with a tool call).
• Do NOT fabricate SKUs, part numbers, or specs.

=== INPUT STRUCTURE (what you will receive each turn) ===
• LLM WIKI — curated brand/technology knowledge (primary reasoning source)
• RAW DATA — retrieved datasheet excerpts (primary spec source)
• CUSTOMER CONTEXT — demo-time identity clues (company, email, role)
• CONVERSATION HISTORY — prior turns in this session
• CURRENT MESSAGE — the customer's latest question

Now read the inputs below and respond as the Selig agent.
"""

# Page Configuration
st.set_page_config(page_title="Selig Group Assistant", page_icon="🛡️", layout="wide")

# Custom CSS for Selig Branding
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #004a99;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #004a99;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def init_gemini(api_key, model_name):
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    return None

@st.cache_data
def load_wiki_content(wiki_dir="wiki"):
    wiki_files = glob.glob(os.path.join(wiki_dir, "*.md"))
    wiki_context = ""
    for wiki_file in wiki_files:
        with open(wiki_file, 'r', encoding='utf-8') as f:
            filename = os.path.basename(wiki_file)
            content = f.read()
            wiki_context += f"\n\n--- WIKI PAGE: {filename} ---\n{content}\n"
    return wiki_context

import pickle

# ... (init_gemini and load_wiki_content remain same)

@st.cache_resource
def load_pdf_index(directory="."):
    # Check if a pre-built index exists
    if os.path.exists("search_index.pkl"):
        with open("search_index.pkl", "rb") as f:
            data = pickle.load(f)
            return data["vectorizer"], data["tfidf_matrix"], data["chunks"], data["metadata"]
    
    # Fallback to building it manually (slower)
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    chunks = []
    metadata = []
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            filename = os.path.basename(pdf_file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    chunks.append(text.strip())
                    metadata.append({"file": filename, "page": page_num + 1})
        except: continue
    
    if chunks:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(chunks)
        return vectorizer, tfidf_matrix, chunks, metadata
    return None, None, [], []

# Sidebar
with st.sidebar:
    st.image("https://www.seliggroup.com/wp-content/themes/selig/assets/images/logo.png", width=200)
    st.title("Settings")
    
    # Model Selector
    selected_model = st.selectbox(
        "Choose AI Model:",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0,
        help="2.5-flash is the newest high-performance model. 2.5-pro is best for extremely complex reasoning."
    )
    
    st.info("This assistant uses the 'LLM Wiki' pattern for high-level reasoning and raw datasheets for technical specs.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.subheader("Customer Context (demo)")
    st.caption("Optional — lets the agent do Tier-1 detection & lead qualification.")
    customer_company = st.text_input("Company", value="", placeholder="e.g., Acme Foods")
    customer_email = st.text_input("Email / domain", value="", placeholder="e.g., buyer@ab-inbev.com")
    customer_role = st.text_input("Role", value="", placeholder="e.g., Head of Packaging")

    st.subheader("Loaded Brand Wiki:")
    wiki_files = [os.path.basename(f) for f in glob.glob("wiki/*.md")]
    for f in wiki_files:
        st.write(f"📄 {f}")

# Main Chat UI
st.title("🛡️ Selig Group Knowledge Assistant")
st.write("Ask anything about our sealing and venting solutions.")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load Data
model = init_gemini(API_KEY, selected_model)
wiki_context = load_wiki_content()
vectorizer, tfidf_matrix, pdf_chunks, pdf_metadata = load_pdf_index()

# Chat Input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Retrieval Step — bumped to top-5 with lower threshold so multi-constraint
        # queries (e.g., "hot-fill acidic HDPE food-grade recyclable") pull richer context.
        tech_context = ""
        if vectorizer is not None:
            query_vec = vectorizer.transform([prompt])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-5:][::-1]
            for idx in top_indices:
                if similarities[idx] > 0.05:
                    meta = pdf_metadata[idx]
                    tech_context += f"\n--- RAW DATA: {meta['file']} (Page {meta['page']}) ---\n{pdf_chunks[idx]}\n"

        # Build customer context string from sidebar inputs
        customer_context_parts = []
        if customer_company:
            customer_context_parts.append(f"Company: {customer_company}")
        if customer_email:
            customer_context_parts.append(f"Email: {customer_email}")
        if customer_role:
            customer_context_parts.append(f"Role: {customer_role}")
        customer_context_str = " | ".join(customer_context_parts) if customer_context_parts else "(anonymous visitor — no identity provided)"

        # Build conversation history (all prior turns — the latest user message is sent separately)
        history_lines = []
        for msg in st.session_state.messages[:-1]:
            role = "Customer" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {msg['content']}")
        history_text = "\n".join(history_lines) if history_lines else "(this is the first turn)"

        if model:
            full_prompt = f"""{SYSTEM_INSTRUCTIONS}

=== LLM WIKI ===
{wiki_context}

=== RAW DATA (retrieved for this turn) ===
{tech_context if tech_context else "(no datasheet excerpts retrieved)"}

=== CUSTOMER CONTEXT ===
{customer_context_str}

=== CONVERSATION HISTORY ===
{history_text}

=== CURRENT CUSTOMER MESSAGE ===
{prompt}

Respond as the Selig agent. Emit any appropriate 🔧 **TOOL:** lines FIRST, then your natural response, then a clear next step.
"""
            try:
                response = model.generate_content(full_prompt)
                answer = response.text
            except Exception as e:
                answer = f"⚠️ API Error: {str(e)}"
        else:
            answer = "❌ API Key not found. Please check your .env file."

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

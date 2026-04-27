# reddit_lead_dashboard.py
"""
Reddit Lead Intelligence Dashboard
A professional Streamlit application for monitoring Reddit, scoring leads,
drafting Claude-powered responses, and managing a lead CRM.
"""

import streamlit as st
import pandas as pd
import praw
from anthropic import Anthropic, APIStatusError, APIConnectionError
import sqlite3
from datetime import datetime, timezone
import re
import plotly.express as px
from plotly import graph_objects as go
import logging

# ----------------------------------------------------------------------
# Logging & Configuration
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page must be first Streamlit command
st.set_page_config(
    page_title="Reddit Lead Intelligence",
    page_icon="📊",
    layout="wide",
)

# ----------------------------------------------------------------------
# Custom Dark Theme CSS (Enterprise aesthetic, high contrast, Lucide-style)
# ----------------------------------------------------------------------
st.markdown(
    """
<style>
    /* Overall dark background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] * {
        color: #fafafa;
    }
    /* Inputs and widgets */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #21262d;
        color: #fafafa;
        border: 1px solid #30363d;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border: 1px solid #2ea043;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        border-color: #3fb950;
    }
    /* Dataframe and tables */
    .stDataFrame {
        background-color: #161b22;
        border: 1px solid #30363d;
    }
    /* Expander/cards */
    .streamlit-expanderHeader {
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
    /* Badge-like lead score */
    .lead-score-badge {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        background-color: #238636;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Session State Initialization
# ----------------------------------------------------------------------
defaults = {
    "reddit_client_id": "",
    "reddit_client_secret": "",
    "reddit_user_agent": "LeadIntelligenceBot/1.0",
    "target_subreddits": "",
    "negative_keywords": "",
    "anthropic_api_key": "",
    "brand_voice": "",
    "product_value_prop": "",
    "monitor_df": pd.DataFrame(),
    "selected_post_id": None,
    "generated_draft": "",
    "draft_edited": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------------------------------------------------
# Database Functions (SQLite Lead CRM)
# ----------------------------------------------------------------------
DB_PATH = "leads.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT NOT NULL,
            subreddit TEXT,
            title TEXT,
            selftext TEXT,
            lead_score INTEGER,
            status TEXT DEFAULT 'new',
            draft_text TEXT,
            sent_reply TEXT,
            created_utc TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

def upsert_lead(post_id, subreddit, title, selftext, lead_score, status="new", draft_text=None, sent_reply=None, created_utc=None):
    conn = sqlite3.connect(DB_PATH)
    # Check if exists
    cur = conn.execute("SELECT id FROM leads WHERE post_id = ?", (post_id,))
    if cur.fetchone():
        conn.execute(
            """UPDATE leads SET subreddit=?, title=?, selftext=?, lead_score=?, status=?,
               draft_text=?, sent_reply=?, last_updated=CURRENT_TIMESTAMP
               WHERE post_id=?""",
            (subreddit, title, selftext, lead_score, status, draft_text, sent_reply, post_id)
        )
    else:
        conn.execute(
            """INSERT INTO leads (post_id, subreddit, title, selftext, lead_score, status,
               draft_text, sent_reply, created_utc)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (post_id, subreddit, title, selftext, lead_score, status, draft_text, sent_reply, created_utc)
        )
    conn.commit()
    conn.close()

def load_leads_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM leads ORDER BY last_updated DESC", conn)
    conn.close()
    return df

def update_lead_status(post_id, new_status):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE leads SET status=?, last_updated=CURRENT_TIMESTAMP WHERE post_id=?", (new_status, post_id))
    conn.commit()
    conn.close()

# ----------------------------------------------------------------------
# Reddit API Helpers
# ----------------------------------------------------------------------
def get_reddit_instance():
    if not all([st.session_state.reddit_client_id, st.session_state.reddit_client_secret, st.session_state.reddit_user_agent]):
        return None
    try:
        reddit = praw.Reddit(
            client_id=st.session_state.reddit_client_id,
            client_secret=st.session_state.reddit_client_secret,
            user_agent=st.session_state.reddit_user_agent,
        )
        # Quick check
        reddit.user.me()
        return reddit
    except Exception as e:
        st.error(f"Reddit authentication failed: {e}")
        return None

def fetch_subreddit_posts(reddit, subreddits, negative_keywords, limit_per_sub=5):
    all_posts = []
    for sub_name in subreddits:
        try:
            sub = reddit.subreddit(sub_name.strip())
            for post in sub.hot(limit=limit_per_sub):
                # Combine title and selftext for keyword filtering
                content = (post.title + " " + (post.selftext or "")).lower()
                # Check negative keywords
                if any(neg.lower() in content for neg in negative_keywords):
                    continue
                all_posts.append({
                    "post_id": post.id,
                    "subreddit": sub_name.strip(),
                    "title": post.title,
                    "selftext": post.selftext[:500] if post.selftext else "",  # truncate for scoring
                    "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                    "url": f"https://reddit.com{post.permalink}",
                })
        except Exception as e:
            logger.warning(f"Error fetching from r/{sub_name}: {e}")
    return all_posts

# ----------------------------------------------------------------------
# Anthropic / Claude Helpers
# ----------------------------------------------------------------------
def get_anthropic_client():
    if st.session_state.anthropic_api_key:
        return Anthropic(api_key=st.session_state.anthropic_api_key)
    return None

def score_post_with_claude(post_title, post_selftext):
    """Use Claude to score a post's lead potential 0-100."""
    client = get_anthropic_client()
    if not client:
        return 50  # fallback if no API key
    brand_voice = st.session_state.brand_voice
    value_prop = st.session_state.product_value_prop
    prompt = f"""You are an expert lead scoring AI. Evaluate the Reddit post below.
Brand Voice: {brand_voice}
Product Value Proposition: {value_prop}
Reddit Post Title: {post_title}
Post Content: {post_selftext}
Rate how likely this post author is a qualified lead for the product.
Respond with ONLY a single integer between 0 and 100. No other text."""
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            temperature=0,
            system="You are a precise lead scoring AI. Reply only with a number.",
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        # Extract number
        match = re.search(r'\b(\d+)\b', text)
        if match:
            score = int(match.group(1))
            return max(0, min(100, score))
        return 50
    except (APIStatusError, APIConnectionError) as e:
        logger.error(f"Claude scoring error: {e}")
        return 50  # fallback

def generate_draft_reply(post_title, post_selftext):
    """Generate a personalized draft reply using Claude."""
    client = get_anthropic_client()
    if not client:
        return "Error: Anthropic API key not configured."
    brand_voice = st.session_state.brand_voice
    value_prop = st.session_state.product_value_prop
    prompt = f"""You are an AI assistant for a business owner. Your task is to craft a helpful, authentic Reddit reply to a potential lead. Align with the brand voice and product value proposition below.

Brand Voice: {brand_voice}
Product Value Proposition: {value_prop}

The Reddit post you are replying to:
Title: {post_title}
Content: {post_selftext}

Write a friendly, non‑salesy reply that genuinely adds value and naturally introduces the product if appropriate. Keep it concise and relevant. Output only the reply text."""
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            temperature=0.7,
            system="You help business owners write authentic Reddit replies.",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except (APIStatusError, APIConnectionError) as e:
        logger.error(f"Claude draft error: {e}")
        return "Error generating draft. Try again."

# ----------------------------------------------------------------------
# Sidebar: Configuration
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    with st.expander("🔑 Reddit API", expanded=False):
        st.session_state.reddit_client_id = st.text_input("Client ID", value=st.session_state.reddit_client_id, type="default")
        st.session_state.reddit_client_secret = st.text_input("Client Secret", value=st.session_state.reddit_client_secret, type="password")
        st.session_state.reddit_user_agent = st.text_input("User Agent", value=st.session_state.reddit_user_agent)

    st.session_state.target_subreddits = st.text_area(
        "🎯 Target Subreddits (comma‑separated)",
        value=st.session_state.target_subreddits,
        placeholder="e.g. SaaS, smallbusiness, Entrepreneur"
    )

    st.session_state.negative_keywords = st.text_area(
        "🚫 Negative Keywords (comma‑separated)",
        value=st.session_state.negative_keywords,
        placeholder="e.g. spam, hiring, webinar"
    )

    with st.expander("🧠 Anthropic API", expanded=False):
        st.session_state.anthropic_api_key = st.text_input("API Key", value=st.session_state.anthropic_api_key, type="password")

    with st.expander("💬 Brand Voice & Value Prop", expanded=False):
        st.session_state.brand_voice = st.text_area("Brand Voice", value=st.session_state.brand_voice,
                                                     placeholder="e.g., Professional but friendly, no jargon...")
        st.session_state.product_value_prop = st.text_area("Product Value Proposition", value=st.session_state.product_value_prop,
                                                            placeholder="e.g., AI‑powered project management for remote teams...")

    # Quick status indicator
    st.divider()
    st.caption("API Status")
    if st.session_state.anthropic_api_key:
        st.success("Anthropic: configured")
    else:
        st.warning("Anthropic: missing")
    if st.session_state.reddit_client_id and st.session_state.reddit_client_secret:
        reddit = get_reddit_instance()
        if reddit:
            st.success("Reddit: connected")
        else:
            st.warning("Reddit: invalid credentials")
    else:
        st.info("Reddit: not configured")

# ----------------------------------------------------------------------
# Initialize DB
# ----------------------------------------------------------------------
init_db()

# ----------------------------------------------------------------------
# Main Tabs
# ----------------------------------------------------------------------
tabs = st.tabs(["📊 Monitor & Heatmap", "📋 Lead CRM"])

# ----------------------------------------------------------------------
# Tab 1: Monitor, Heatmap & Draft Engine
# ----------------------------------------------------------------------
with tabs[0]:
    st.header("Live Signals")

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔄 Fetch New Signals", use_container_width=True):
            reddit = get_reddit_instance()
            if not reddit:
                st.error("Reddit API credentials are required. Configure them in the sidebar.")
            else:
                subreddits = [s.strip() for s in st.session_state.target_subreddits.split(",") if s.strip()]
                if not subreddits:
                    st.warning("Please enter at least one target subreddit.")
                else:
                    neg_keywords = [k.strip() for k in st.session_state.negative_keywords.split(",") if k.strip()]
                    with st.spinner("Fetching posts and scoring leads with Claude..."):
                        posts = fetch_subreddit_posts(reddit, subreddits, neg_keywords, limit_per_sub=7)
                        if not posts:
                            st.info("No posts found after filtering negative keywords.")
                        else:
                            scores = []
                            for p in posts:
                                score = score_post_with_claude(p["title"], p["selftext"])
                                p["lead_score"] = score
                                scores.append(score)
                                # Save to CRM as 'new' lead (upsert)
                                upsert_lead(
                                    post_id=p["post_id"],
                                    subreddit=p["subreddit"],
                                    title=p["title"],
                                    selftext=p["selftext"],
                                    lead_score=score,
                                    status="new",
                                    created_utc=p["created_utc"].isoformat()
                                )
                            df = pd.DataFrame(posts)
                            st.session_state.monitor_df = df.sort_values("lead_score", ascending=False)
                            st.success(f"Fetched {len(posts)} leads.")
    with col2:
        if st.button("❌ Clear Signals", use_container_width=True):
            st.session_state.monitor_df = pd.DataFrame()
            st.session_state.selected_post_id = None
            st.session_state.generated_draft = ""

    # Display table
    monitor_df = st.session_state.monitor_df
    if not monitor_df.empty:
        # Format the table
        display_df = monitor_df[["subreddit", "title", "lead_score", "post_id"]].copy()
        display_df.rename(columns={"subreddit": "Subreddit", "title": "Post Title", "lead_score": "Lead Score"}, inplace=True)

        # Color‑code score
        def color_score(val):
            if val >= 80:
                return f'background-color: #1b5e20; color: white; border-radius: 4px; padding: 2px 6px;'
            elif val >= 50:
                return f'background-color: #e65100; color: white; border-radius: 4px; padding: 2px 6px;'
            else:
                return f'background-color: #b71c1c; color: white; border-radius: 4px; padding: 2px 6px;'
        styled = display_df.style.applymap(color_score, subset=["Lead Score"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=300)

        # Post selection for draft
        st.subheader("✍️ Draft Reply")
        post_options = {f"{row['Subreddit']} – {row['Post Title'][:60]}...": row["post_id"] for _, row in display_df.iterrows()}
        selected_label = st.selectbox("Select a post to engage", list(post_options.keys()), key="post_selector")
        if selected_label:
            selected_id = post_options[selected_label]
            if st.session_state.selected_post_id != selected_id:
                st.session_state.selected_post_id = selected_id
                st.session_state.generated_draft = ""

        # Show selected post details and draft card
        if st.session_state.selected_post_id:
            sel = monitor_df[monitor_df.post_id == st.session_state.selected_post_id].iloc[0]
            with st.expander(f"📌 {sel['title'][:100]}", expanded=True):
                st.caption(f"r/{sel['subreddit']} | Lead Score: **{sel['lead_score']}**")
                st.markdown(f"[Open on Reddit]({sel['url']})")
                st.text_area("Post Content", value=sel["selftext"], height=100, disabled=True)

                col_a, col_b = st.columns([1, 2])
                with col_a:
                    if st.button("🤖 Generate Draft", use_container_width=True):
                        with st.spinner("Claude is crafting a reply..."):
                            draft = generate_draft_reply(sel["title"], sel["selftext"])
                            st.session_state.generated_draft = draft
                            # Upsert with draft
                            upsert_lead(
                                post_id=sel["post_id"],
                                subreddit=sel["subreddit"],
                                title=sel["title"],
                                selftext=sel["selftext"],
                                lead_score=sel["lead_score"],
                                status="new",
                                draft_text=draft,
                                created_utc=sel["created_utc"].isoformat()
                            )
                with col_b:
                    if st.session_state.generated_draft:
                        st.success("Draft ready for review.")

                if st.session_state.generated_draft:
                    # Human‑in‑the‑Loop Editor (rich‑text approximation)
                    st.markdown("### 📝 Human‑in‑the‑Loop Editor")
                    edited_draft = st.text_area(
                        "Edit your reply below",
                        value=st.session_state.generated_draft,
                        height=200,
                        key="draft_editor"
                    )
                    st.session_state.draft_edited = (edited_draft != st.session_state.generated_draft)

                    send_col, save_col = st.columns(2)
                    with send_col:
                        if st.button("🚀 Send Reply to Reddit", use_container_width=True):
                            reddit = get_reddit_instance()
                            if not reddit:
                                st.error("Reddit API not configured.")
                            else:
                                try:
                                    submission = reddit.submission(id=sel["post_id"])
                                    submission.reply(edited_draft)
                                    st.success("Reply posted successfully!")
                                    # Update CRM status to 'contacted'
                                    update_lead_status(sel["post_id"], "contacted")
                                    upsert_lead(
                                        post_id=sel["post_id"],
                                        subreddit=sel["subreddit"],
                                        title=sel["title"],
                                        selftext=sel["selftext"],
                                        lead_score=sel["lead_score"],
                                        status="contacted",
                                        draft_text=edited_draft,
                                        sent_reply=edited_draft,
                                        created_utc=sel["created_utc"].isoformat()
                                    )
                                    st.balloons()
                                except Exception as e:
                                    st.error(f"Failed to send reply: {e}")
                    with save_col:
                        if st.button("💾 Save Draft Only", use_container_width=True):
                            upsert_lead(
                                post_id=sel["post_id"],
                                subreddit=sel["subreddit"],
                                title=sel["title"],
                                selftext=sel["selftext"],
                                lead_score=sel["lead_score"],
                                status="drafted",
                                draft_text=edited_draft,
                                created_utc=sel["created_utc"].isoformat()
                            )
                            st.success("Draft saved to CRM.")

        # ----------------------------------------------------------------------
        # Heatmap
        # ----------------------------------------------------------------------
        st.subheader("🔥 Activity Heatmap (Hour of Day × Day of Week)")
        if not monitor_df.empty:
            monitor_df["hour"] = pd.to_datetime(monitor_df["created_utc"]).dt.hour
            monitor_df["day_of_week"] = pd.to_datetime(monitor_df["created_utc"]).dt.day_name()
            # Ensure categorical order for days
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            agg = monitor_df.groupby(["day_of_week", "hour"]).size().reset_index(name="count")
            # Pivot for heatmap
            pivot = agg.pivot(index="day_of_week", columns="hour", values="count").reindex(days_order).fillna(0)

            fig = px.imshow(
                pivot,
                labels=dict(x="Hour of Day", y="Day of Week", color="Posts"),
                x=pivot.columns,
                y=pivot.index,
                color_continuous_scale="Blues",
                aspect="auto",
                title="Subreddit Activity Heatmap (from fetched posts)"
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                              font_color="#fafafa")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fetch some signals to see the activity heatmap.")

    else:
        st.info("No signals yet. Click 'Fetch New Signals' to start monitoring.")

# ----------------------------------------------------------------------
# Tab 2: Lead CRM
# ----------------------------------------------------------------------
with tabs[1]:
    st.header("Lead Relationship Management")
    leads_df = load_leads_df()
    if leads_df.empty:
        st.info("No leads in the CRM. Leads will appear after you fetch and score them.")
    else:
        # Editable status drop-down
        st.subheader("Manage Leads")
        edited_df = st.data_editor(
            leads_df[["id", "post_id", "subreddit", "title", "lead_score", "status"]],
            column_config={
                "status": st.column_config.SelectboxColumn(
                    "Status",
                    options=["new", "drafted", "contacted", "replied", "closed"],
                    required=True,
                )
            },
            disabled=["id", "post_id", "subreddit", "title", "lead_score"],
            use_container_width=True,
            num_rows="fixed",
        )
        if st.button("💾 Save Status Changes", use_container_width=True):
            for _, row in edited_df.iterrows():
                update_lead_status(row["post_id"], row["status"])
            st.success("Statuses updated.")
            st.rerun()

        # Quick stats
        st.divider()
        st.metric("Total Leads", len(leads_df))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("New", len(leads_df[leads_df.status == "new"]))
        with col2:
            st.metric("Contacted", len(leads_df[leads_df.status == "contacted"]))
        with col3:
            st.metric("Closed", len(leads_df[leads_df.status == "closed"]))

        # Full lead details
        st.subheader("Lead Details")
        selected_lead_id = st.selectbox("Select a lead to view/edit draft", leads_df["post_id"].tolist())
        if selected_lead_id:
            lead = leads_df[leads_df.post_id == selected_lead_id].iloc[0]
            with st.expander("Post Content", expanded=True):
                st.markdown(f"**r/{lead['subreddit']}** – {lead['title']}")
                st.text_area("Selftext", value=lead["selftext"], height=150, disabled=True)
            if lead["draft_text"]:
                st.markdown("### Draft Reply")
                new_draft = st.text_area("Edit draft", value=lead["draft_text"], height=200, key="crm_draft_editor")
                if st.button("Update Draft in CRM", use_container_width=True):
                    upsert_lead(
                        post_id=lead["post_id"],
                        subreddit=lead["subreddit"],
                        title=lead["title"],
                        selftext=lead["selftext"],
                        lead_score=lead["lead_score"],
                        status=lead["status"],
                        draft_text=new_draft,
                        sent_reply=lead.get("sent_reply"),
                        created_utc=lead["created_utc"]
                    )
                    st.success("Draft updated.")
                    st.rerun()
            else:
                st.info("No draft yet. Generate one from the Monitor tab.")

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.caption("📊 Reddit Lead Intelligence Dashboard · Built with Streamlit, PRAW & Claude")

"""Streamlit frontend for research discovery and uploaded-paper QA."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

from frontend.api_client import ApiError, ResearchApi


st.set_page_config(page_title="Paper Atlas", page_icon="◈", layout="wide")

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
	:root { --ink:#29231d; --muted:#766b5c; --paper:#f5efe3; --surface:#fffdf8; --saffron:#c87921; --gold:#e0a74f; --deep:#4b2e1b; --line:#dfd2bd; --soft:#f0e5d2; }
	.stApp { background:var(--paper); color:var(--ink); }
	[data-testid="stHeader"] { background:rgba(245,239,227,.88); }
	[data-testid="stSidebar"] { background:#fbf8f1; border-right:1px solid var(--line); }
	[data-testid="stSidebar"] * { color:#242a2f; }
	[data-testid="stSidebar"] .stCaption { color:#7c7d78; }
	[data-testid="stSidebar"] .stButton > button { border:0; color:#242a2f; background:transparent; text-align:left; }
	[data-testid="stSidebar"] .stButton > button:hover { background:#e9f1ec; color:#006b5b; }
	h1,h2,h3 { font-family:'Fraunces', Georgia, serif; color:var(--ink); letter-spacing:0; }
	p, label, input, textarea, button { font-family:'DM Sans', sans-serif; }
	.hero { padding:1.1rem 0 1.7rem; border-bottom:1px solid var(--line); margin-bottom:1.7rem; }
	.eyebrow { color:var(--saffron); font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
	.hero h1 { font-size:clamp(2.8rem,6vw,5.8rem); line-height:.94; margin:.35rem 0 .9rem; max-width:12ch; }
	.hero p { color:var(--muted); max-width:41rem; font-size:1.02rem; line-height:1.65; }
	.stForm, [data-testid="stVerticalBlockBorderWrapper"] { background:var(--surface); border:1px solid var(--line); border-radius:8px; }
	[data-testid="stVerticalBlockBorderWrapper"] { padding:1.25rem 1.3rem; }
	.stTextInput input, .stTextArea textarea, .stNumberInput input { background:#fbf7ef; border:1px solid var(--line); color:var(--ink); border-radius:4px; }
	.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus { border-color:var(--gold); box-shadow:0 0 0 1px var(--gold); }
	.stButton > button { min-height:2.7rem; border-radius:4px; border:1px solid var(--deep); color:var(--deep); background:transparent; font-weight:700; }
	.stButton > button:hover { border-color:var(--saffron); color:var(--saffron); }
	.stButton > button[kind="primary"] { background:var(--saffron); border-color:var(--saffron); color:#fffaf2; }
	.stButton > button[kind="primary"]:hover { background:#a96118; border-color:#a96118; color:#fffaf2; }
	.result { border-top:3px solid var(--gold); padding:1rem 0 1.25rem; }
	.result h3 { margin:.2rem 0 .4rem; }
	.meta { color:var(--saffron); font-size:.75rem; font-weight:700; letter-spacing:.08em; }
	.answer { background:#f4e4c8; border-left:4px solid var(--saffron); padding:1rem 1.2rem; margin:.8rem 0 1rem; border-radius:0 5px 5px 0; }
	[data-testid="stExpander"] { background:var(--surface); border:1px solid var(--line); border-radius:5px; }
	[data-testid="stFileUploader"] { background:#fbf7ef; border:1px dashed #c8ab7e; border-radius:5px; padding:.35rem; }
	[data-testid="stMetric"] { background:var(--surface); border-top:2px solid var(--gold); padding:.7rem; }
	.stAlert { border-radius:5px; }
	.header-row { display:grid; grid-template-columns:270px 1fr 220px; gap:1rem; align-items:center; padding:.4rem 0 .8rem; border-bottom:1px solid var(--line); margin-bottom:1.2rem; }
	.brand { display:flex; align-items:center; gap:.65rem; color:#10264a; font-weight:700; line-height:1.05; }
	.brand-mark { display:grid; place-items:center; width:2.4rem; height:2.4rem; border-radius:50%; background:#006b5b; color:#f8e7b3; font-size:1.3rem; }
	.user-chip { display:flex; align-items:center; justify-content:flex-end; gap:.55rem; color:#10264a; font-size:.82rem; }
	.avatar { display:grid; place-items:center; width:2.25rem; height:2.25rem; border-radius:50%; background:#10264a; color:white; font-weight:700; }
	.workflow-title { color:#c9942e; letter-spacing:.06em; font-size:.88rem; font-weight:700; }
	.workflow { position:relative; padding:.35rem 0 .2rem .2rem; }
	.workflow:before { content:""; position:absolute; left:1.05rem; top:2.2rem; bottom:2rem; border-left:1px dotted #c9942e; }
	.step { position:relative; display:flex; gap:.8rem; align-items:flex-start; padding:.75rem 0; }
	.step-mark { z-index:1; display:grid; place-items:center; min-width:2.1rem; height:2.1rem; border-radius:50%; background:#fff8e8; border:1px solid #e6c77a; color:#c9942e; font-weight:700; }
	.step strong { display:block; color:#10264a; font-size:.88rem; } .step small { display:block; color:var(--muted); margin-top:.22rem; }
	.deep-card { margin-top:.75rem; border:1px solid #e6c77a; background:#fff9ed; border-radius:8px; padding:.9rem; }
	.deep-card strong { color:#10264a; } .deep-card p { color:var(--muted); font-size:.78rem; margin:.3rem 0 0; }
	.feature-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin-top:1.1rem; }
	.feature-card { min-height:7.5rem; background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:1rem; }
	.feature-card h3 { font-size:1rem; margin:.4rem 0; } .feature-card p { color:var(--muted); font-size:.78rem; line-height:1.45; margin:0; }
	.feature-icon { color:#006b5b; font-size:1.35rem; } .feature-icon.gold { color:#c9942e; }
	@media (max-width:900px) { .header-row { grid-template-columns:1fr; } .user-chip { justify-content:flex-start; } .feature-grid { grid-template-columns:repeat(2,1fr); } }
	@media (max-width: 700px) { .hero { padding-top:.4rem; } .hero h1 { font-size:3.1rem; } [data-testid="stVerticalBlockBorderWrapper"] { padding:.9rem; } }
	</style>
	""",
	unsafe_allow_html=True,
)


def _local_backend_is_running() -> bool:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
		connection.settimeout(0.2)
		return connection.connect_ex(("127.0.0.1", 8000)) == 0


def _start_local_backend() -> None:
	if _local_backend_is_running():
		return

	process = st.session_state.get("backend_process")
	if process is not None and process.poll() is None:
		return

	project_root = Path(__file__).resolve().parents[1]
	creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
	st.session_state.backend_process = subprocess.Popen(
		[
			sys.executable,
			"-m",
			"uvicorn",
			"backend.main:app",
			"--host",
			"127.0.0.1",
			"--port",
			"8000",
		],
		cwd=project_root,
		creationflags=creation_flags,
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
	)
	for _ in range(100):
		if _local_backend_is_running():
			return
		time.sleep(0.1)


def api() -> ResearchApi:
	base_url = os.getenv("PAPER_API_URL", "http://localhost:8000")
	if base_url.rstrip("/") in {"http://localhost:8000", "http://127.0.0.1:8000"}:
		_start_local_backend()
	return ResearchApi(base_url)


def show_error(error: Exception) -> None:
	st.error(str(error))


if "session_id" not in st.session_state:
	st.session_state.session_id = uuid.uuid4().hex
if "uploaded_papers" not in st.session_state:
	st.session_state.uploaded_papers = {}
if "active_view" not in st.session_state:
	st.session_state.active_view = "Dashboard"
if "recent_questions" not in st.session_state:
	st.session_state.recent_questions = []
if "dark_mode" not in st.session_state:
	st.session_state.dark_mode = False

papers = st.session_state.uploaded_papers

with st.sidebar:
	st.markdown("## AI RESEARCH PAPER ASSISTANT")
	st.caption("Understand papers with more clarity")
	st.markdown("<div class='nav-label'>WORKSPACE</div>", unsafe_allow_html=True)
	for nav_item in ("Dashboard", "Research Discovery", "My Paper", "Saved Papers", "History", "Notebook", "Alerts", "Settings"):
		if st.button(nav_item, key=f"nav_{nav_item}", use_container_width=True, type="primary" if st.session_state.active_view == nav_item else "secondary"):
			st.session_state.active_view = {"Research Discovery": "Discover research", "My Paper": "Upload Document", "Saved Papers": "Saved Answers", "History": "Recent Questions"}.get(nav_item, nav_item)
	st.divider()
	st.caption("CONNECTION")
	st.code(os.getenv("PAPER_API_URL", "http://localhost:8000"), language=None)
	if st.button("Check connection", use_container_width=True):
		try:
			st.success(api().health().get("status", "healthy").title())
		except ApiError as error:
			show_error(error)

st.markdown('<div class="header-row"><div class="brand"><div class="brand-mark">▤</div><span>AI RESEARCH PAPER<br>ASSISTANT</span></div><div></div><div class="user-chip"><span>☼</span><div class="avatar">R</div><span>Researcher⌄</span></div></div>', unsafe_allow_html=True)
global_query = st.text_input("Search papers, topics, authors...", label_visibility="collapsed", placeholder="Search papers, topics, authors...")
if global_query.strip():
	try:
		st.session_state.search = api().search(global_query, 20, 3)
		st.session_state.active_view = "Discover research"
	except ApiError as error:
		show_error(error)

st.markdown('<div class="topbar"><div><div class="eyebrow">AI RESEARCH PAPER ASSISTANT</div><h1>Welcome back. <span>✦</span></h1><p>Explore, understand, and analyze research papers with AI.</p></div><div class="topbar-badge">Evidence-led workspace</div></div>', unsafe_allow_html=True)

uploaded_count = len(st.session_state.uploaded_papers)
question_count = len(st.session_state.recent_questions)
st.markdown(f"""<div class="metrics">
<div class="metric"><div class="metric-icon green">▤</div><div><strong>{uploaded_count}</strong><small>Documents indexed</small><em>↑ Ready to explore</em></div></div>
<div class="metric"><div class="metric-icon gold">◌</div><div><strong>{question_count}</strong><small>Questions asked</small><em>↑ This session</em></div></div>
<div class="metric"><div class="metric-icon navy">◎</div><div><strong>RAG</strong><small>Evidence retrieval</small><em>↑ Source grounded</em></div></div>
<div class="metric"><div class="metric-icon green">▱</div><div><strong>Live</strong><small>Research index</small><em>↑ Backend connected</em></div></div>
</div>""", unsafe_allow_html=True)

active_view = st.session_state.active_view
if active_view in ("Dashboard", "Ask a Question", "Q&A"):
	st.markdown('<div class="section-heading"><div><h2>Ask a question</h2><p>Get accurate answers from your research documents.</p></div><span class="spark">✦</span></div>', unsafe_allow_html=True)
	left, right = st.columns([1.55, 1])
	with left:
		with st.container(border=True):
			options = ["All Documents", *list(papers)]
			selected = st.selectbox("Select document (optional)", options, format_func=lambda item: papers[item]["filename"] if item in papers else item)
			question = st.text_area("What would you like to know about your research papers?", height=120, placeholder="Ask about a method, finding, dataset, or limitation...")
			c1, c2, c3 = st.columns([1, 1, 1.4])
			with c1:
				st.selectbox("Top K results", [3, 5, 10], index=1)
			with c2:
				advanced = st.checkbox("Advanced options")
			with c3:
				st.write("")
				submit_question = st.button("✦ Ask with evidence", type="primary", use_container_width=True, disabled=not question.strip())
			if advanced:
				st.caption("Answers use only indexed source passages and preserve document provenance.")
			if submit_question:
				if selected == "All Documents":
					st.warning("Select an indexed document before asking a paper question.")
				else:
					with st.spinner("Retrieving relevant passages..."):
						try:
							st.session_state.answer = api().ask(selected, question, st.session_state.session_id)
							st.session_state.recent_questions.insert(0, {"question": question, "document": papers[selected]["filename"]})
							st.session_state.recent_questions = st.session_state.recent_questions[:8]
						except ApiError as error:
							show_error(error)
			if st.session_state.get("answer"):
				answer = st.session_state.answer
				st.markdown(f"**{answer.get('status', 'Result').replace('_', ' ').title()}** · {'Grounded in indexed evidence' if answer.get('grounded') else 'Evidence was insufficient'}")
				if answer.get("answer"):
					st.info(answer["answer"])
				for index, evidence in enumerate(answer.get("evidence", []), 1):
					with st.expander(f"Evidence {index} · {evidence.get('section', 'Unknown section')}"):
						st.write(evidence.get("text", ""))
	with right:
		with st.container(border=True):
			st.markdown('<div class="workflow-title">▤ &nbsp; MY PAPER WORKFLOW</div><div class="workflow"><div class="step"><div class="step-mark">1</div><div><strong>Upload PDF</strong><small>Upload your research paper</small></div></div><div class="step"><div class="step-mark">2</div><div><strong>Summary</strong><small>Get an AI-generated summary</small></div></div><div class="step"><div class="step-mark">3</div><div><strong>Q&amp;A</strong><small>Ask anything about your paper</small></div></div><div class="step"><div class="step-mark">4</div><div><strong>Strong / Weak Analysis</strong><small>Discover strengths and weaknesses</small></div></div></div><div class="deep-card"><strong>♔ &nbsp; AI-Powered Deep Analysis</strong><p>Understand your research better with actionable insights.</p></div>', unsafe_allow_html=True)
			st.markdown("### Recent documents")
			if papers:
				for document_id, document in list(papers.items())[-5:]:
					st.markdown(f"**{document['filename']}**  \n<span class='meta'>Indexed · {document_id[:12]}...</span>", unsafe_allow_html=True)
					st.divider()
			else:
				st.caption("Your indexed documents will appear here.")
			if st.button("Upload PDF", key="workflow_upload", use_container_width=True):
				st.session_state.active_view = "Upload Document"
				st.rerun()
			if st.button("Summary", key="workflow_summary", use_container_width=True):
				st.session_state.active_view = "Summary"
				st.rerun()
			if st.button("Q&A", key="workflow_qa", use_container_width=True):
				st.session_state.active_view = "Q&A"
				st.rerun()
			if st.button("Strong / Weak Analysis", key="workflow_analysis", use_container_width=True):
				st.session_state.active_view = "Strong / Weak Analysis"
				st.rerun()
	st.markdown('<div class="section-heading lower"><div><h2>Recent questions</h2><p>Your latest research trail.</p></div></div>', unsafe_allow_html=True)
	with st.container(border=True):
		if st.session_state.recent_questions:
			for item in st.session_state.recent_questions[:5]:
				st.markdown(f"**{item['question']}**  ·  <span class='meta'>{item['document']}</span>", unsafe_allow_html=True)
				st.divider()
		else:
			st.caption("Questions you ask will be saved in this session.")
	st.markdown('<div class="feature-grid"><div class="feature-card"><div class="feature-icon">▤</div><h3>Summary</h3><p>Get a concise summary of papers in seconds.</p></div><div class="feature-card"><div class="feature-icon gold">✣</div><h3>Model</h3><p>Understand models, algorithms and approaches.</p></div><div class="feature-card"><div class="feature-icon">▥</div><h3>Dataset</h3><p>Explore datasets and benchmark information.</p></div><div class="feature-card"><div class="feature-icon gold">▥</div><h3>Insights</h3><p>Methodology, findings, strong and weak points.</p></div></div>', unsafe_allow_html=True)
	st.markdown('<div class="paper-table"><div class="paper-row paper-head"><span>#</span><span>Paper Title</span><span>Authors</span><span>Year</span><span>Relevance</span></div><div class="paper-row"><span>1</span><span>A Survey on Large Language Models</span><span>Touvron, H. et al.</span><span>2023</span><span class="relevance">★★★★★ 98%</span></div><div class="paper-row"><span>2</span><span>Retrieval-Augmented Generation for Knowledge</span><span>Lewis, P. et al.</span><span>2020</span><span class="relevance">★★★★★ 96%</span></div><div class="paper-row"><span>3</span><span>Attention Is All You Need</span><span>Vaswani, A. et al.</span><span>2017</span><span class="relevance">★★★★★ 95%</span></div><div class="paper-row"><span>4</span><span>Graph Neural Networks: A Review</span><span>Zhou, J. et al.</span><span>2020</span><span class="relevance">★★★★☆ 94%</span></div><div class="paper-row"><span>5</span><span>Contrastive Learning: A Comprehensive Survey</span><span>Chen, T. et al.</span><span>2021</span><span class="relevance">★★★★☆ 93%</span></div></div>', unsafe_allow_html=True)
	if st.button("View All Results →", key="view_all_results", use_container_width=True):
		st.session_state.active_view = "Discover research"
		st.rerun()
	feature_actions = {"Summary": "Summary", "Model": "Model", "Dataset": "Dataset", "Insights": "Strong / Weak Analysis"}
	for feature_name, destination in feature_actions.items():
		if st.button(feature_name, key=f"feature_{feature_name}"):
			st.session_state.active_view = destination
			st.rerun()

elif active_view == "Discover research":
	st.subheader("Find a research thread")
	st.caption("Search the indexed literature and open the papers worth reading.")
	with st.form("research_search"):
		query = st.text_input("Research question or topic", placeholder="e.g. transformer-based medical image segmentation")
		col1, col2, col3 = st.columns([1, 1, 2])
		with col1:
			candidate_k = st.number_input("Candidates", min_value=1, max_value=500, value=20)
		with col2:
			final_k = st.number_input("Results", min_value=1, max_value=10, value=3)
		with col3:
			st.write("")
			submitted = st.form_submit_button("✦ Search the index", type="primary", use_container_width=True)
	if submitted:
		if not query.strip():
			st.warning("Enter a topic or question first.")
		elif final_k > candidate_k:
			st.warning("Results must be less than or equal to candidates.")
		else:
			try:
				st.session_state.search = api().search(query, int(candidate_k), int(final_k))
			except ApiError as error:
				show_error(error)
	search = st.session_state.get("search")
	if search:
		st.caption(f"{search.get('returned_count', len(search.get('papers', [])))} papers returned for {search.get('query', '')}")
		for paper in search.get("papers", []):
			with st.container(border=True):
				st.markdown(f'<div class="meta">RANK {paper.get("rank", "-")} · SCORE {float(paper.get("final_score", 0)):.3f}</div>', unsafe_allow_html=True)
				st.subheader(paper.get("title") or "Untitled paper")
				st.write(paper.get("summary") or "No summary was returned for this result.")

elif active_view in ("Model", "Dataset"):
	st.subheader(active_view)
	st.caption(f"Explore the {active_view.lower()} details of an indexed research paper.")
	if papers:
		with st.container(border=True):
			st.markdown(f"### {active_view} analysis")
			st.info("This analysis view is ready for connection to the existing paper-analysis service.")
	else:
		st.info("Upload a paper first to populate this analysis view.")

elif active_view == "Summary":
	st.subheader("Paper summary")
	st.caption("A structured reading view for your selected research paper.")
	if papers:
		st.success("Your paper is indexed and ready for summary generation.")
		with st.container(border=True):
			st.markdown("### Overview")
			st.write("Summary generation is ready to connect to the paper analysis service.")
			st.markdown("### Problem Statement")
			st.caption("The indexed document will provide this section when analysis is enabled.")
			st.markdown("### Methodology · Key Findings · Conclusion")
			st.caption("Use Q&A to interrogate the evidence while structured analysis is unavailable.")
	else:
		st.info("Upload a paper first to create a summary workspace.")

elif active_view == "Strong / Weak Analysis":
	st.subheader("Strong / Weak Analysis")
	st.caption("Review methodology, novelty, datasets, results, and limitations.")
	positive, neutral = st.columns(2)
	with positive:
		with st.container(border=True):
			st.markdown("### Strengths")
			st.write("✓ Methodology")
			st.write("✓ Novelty")
			st.write("✓ Dataset and results")
	with neutral:
		with st.container(border=True):
			st.markdown("### Areas to examine")
			st.write("• Dataset limitations")
			st.write("• Methodology limitations")
			st.write("• Missing evaluation")

elif active_view == "Upload Document":
	st.subheader("Upload document")
	st.caption("Create a searchable evidence desk from a research PDF.")
	upload = st.file_uploader("Add one research paper", type=["pdf"])
	if upload and st.button("✦ Index this paper", type="primary"):
		try:
			result = api().upload(upload.name, upload.getvalue())
			st.session_state.uploaded_papers[result["document_id"]] = {"document_id": result["document_id"], "filename": result.get("filename", upload.name)}
			st.success(f"Indexed {result.get('filename', upload.name)}")
		except ApiError as error:
			show_error(error)

elif active_view in ("My Documents", "All Documents"):
	st.subheader(active_view)
	st.caption("Your paper library and indexed evidence sources.")
	if not st.session_state.uploaded_papers:
		st.info("No documents indexed yet. Open Upload Document to add a PDF.")
	else:
		for document in st.session_state.uploaded_papers.values():
			with st.container(border=True):
				st.markdown(f"### {document['filename']}")
				st.caption(document["document_id"])
				if st.button("Ask this paper", key=f"ask_{document['document_id']}"):
					st.session_state.active_view = "Ask a Question"
					st.rerun()

elif active_view == "Recent Questions":
	st.subheader("Recent questions")
	for item in st.session_state.recent_questions or [{"question": "No questions asked yet.", "document": ""}]:
		with st.container(border=True):
			st.write(item["question"])
			st.caption(item["document"])

elif active_view == "Saved Answers":
	st.subheader("Saved answers")
	st.info("Saved answer controls will appear beside answers as you build your research trail.")

elif active_view == "Analytics":
	st.subheader("Analytics")
	st.caption("A compact view of this session's research activity.")
	col1, col2 = st.columns(2)
	col1.metric("Questions this session", question_count)
	col2.metric("Documents indexed", uploaded_count)

elif active_view == "Settings":
	st.subheader("Settings")
	st.session_state.dark_mode = st.toggle("Dark mode preview", value=st.session_state.dark_mode)
	st.caption("Your session preferences are stored locally while this app is open.")

elif active_view == "Help & Support":
	st.subheader("Help & Support")
	st.info("Upload a PDF, wait for indexing to finish, then ask a specific question. Answers include the source passages used.")

	if False:
		st.markdown('<div class="topbar"><div><div class="eyebrow">VEDAKSH RESEARCH DESK</div><h1>Welcome back. <span>✦</span></h1><p>Your intelligent research companion for understanding papers.</p></div><div class="topbar-badge">Evidence-led workspace</div></div>', unsafe_allow_html=True)

		uploaded_count = len(st.session_state.uploaded_papers)
		question_count = len(st.session_state.recent_questions)
		st.markdown(f"""<div class="metrics">
		<div class="metric"><div class="metric-icon green">▤</div><div><strong>{uploaded_count}</strong><small>Documents indexed</small><em>↑ Ready to explore</em></div></div>
		<div class="metric"><div class="metric-icon gold">◌</div><div><strong>{question_count}</strong><small>Questions asked</small><em>↑ This session</em></div></div>
		<div class="metric"><div class="metric-icon navy">◎</div><div><strong>RAG</strong><small>Evidence retrieval</small><em>↑ Source grounded</em></div></div>
		<div class="metric"><div class="metric-icon green">▱</div><div><strong>Live</strong><small>Research index</small><em>↑ Backend connected</em></div></div>
		</div>""", unsafe_allow_html=True)

		active_view = st.session_state.active_view
		if active_view in ("Dashboard", "Ask a Question"):
			st.markdown('<div class="section-heading"><div><h2>Ask a question</h2><p>Get accurate answers from your research documents.</p></div><span class="spark">✦</span></div>', unsafe_allow_html=True)
			left, right = st.columns([1.55, 1])
			with left:
				with st.container(border=True):
					papers = st.session_state.uploaded_papers
					options = ["All Documents", *list(papers)]
					selected = st.selectbox("Select document (optional)", options, format_func=lambda item: papers[item]["filename"] if item in papers else item)
					question = st.text_area("What would you like to know about your research papers?", height=120, placeholder="Ask about a method, finding, dataset, or limitation...")
					c1, c2, c3 = st.columns([1, 1, 1.4])
					with c1:
						result_limit = st.selectbox("Top K results", [3, 5, 10], index=1)
					with c2:
						advanced = st.checkbox("Advanced options")
					with c3:
						st.write("")
						submit_question = st.button("✦ Ask with evidence", type="primary", use_container_width=True, disabled=not question.strip())
					if advanced:
						st.caption("Answers use only indexed source passages and preserve document provenance.")
					if submit_question:
						if selected == "All Documents":
							st.warning("Select an indexed document before asking a paper question.")
						else:
							with st.spinner("Retrieving relevant passages..."):
								try:
									answer = api().ask(selected, question, st.session_state.session_id)
									st.session_state.answer = answer
									st.session_state.recent_questions.insert(0, {"question": question, "document": papers[selected]["filename"], "answer": answer.get("answer")})
									st.session_state.recent_questions = st.session_state.recent_questions[:8]
								except ApiError as error:
									show_error(error)
					if st.session_state.get("answer"):
							answer = st.session_state.answer
							st.markdown(f"**{answer.get('status', 'Result').replace('_', ' ').title()}** · {'Grounded in indexed evidence' if answer.get('grounded') else 'Evidence was insufficient'}")
							if answer.get("answer"):
								st.info(answer["answer"])
							for index, evidence in enumerate(answer.get("evidence", []), 1):
								with st.expander(f"Evidence {index} · {evidence.get('section', 'Unknown section')}"):
									st.write(evidence.get("text", ""))
			with right:
				with st.container(border=True):
					st.markdown("### Recent documents")
					if st.session_state.uploaded_papers:
						for document_id, document in list(st.session_state.uploaded_papers.items())[-5:]:
							st.markdown(f"**{document['filename']}**  \n<span class='meta'>Indexed · {document_id[:12]}...</span>", unsafe_allow_html=True)
							st.divider()
					else:
						st.caption("Your indexed documents will appear here.")
					if st.button("＋ Upload new document", use_container_width=True):
						st.session_state.active_view = "Upload Document"
						st.rerun()

			st.markdown('<div class="section-heading lower"><div><h2>Recent questions</h2><p>Your latest research trail.</p></div></div>', unsafe_allow_html=True)
			with st.container(border=True):
				if st.session_state.recent_questions:
					for item in st.session_state.recent_questions[:5]:
						st.markdown(f"**{item['question']}**  ·  <span class='meta'>{item['document']}</span>", unsafe_allow_html=True)
						st.divider()
				else:
					st.caption("Questions you ask will be saved in this session.")

		elif active_view == "Upload Document":
			st.subheader("Upload document")
			st.caption("Create a searchable evidence desk from a research PDF.")
			upload = st.file_uploader("Add one research paper", type=["pdf"], help="PDF files only, up to the backend's configured limit.")
			if upload and st.button("✦ Index this paper", type="primary"):
				with st.spinner("Extracting, chunking, and indexing the paper..."):
					try:
						result = api().upload(upload.name, upload.getvalue())
						st.session_state.uploaded_papers[result["document_id"]] = {"document_id": result["document_id"], "filename": result.get("filename", upload.name)}
						st.success(f"Indexed {result.get('filename', upload.name)}")
					except ApiError as error:
						show_error(error)

		elif active_view in ("My Documents", "All Documents"):
			st.subheader(active_view)
			st.caption("Your paper library and indexed evidence sources.")
			if not st.session_state.uploaded_papers:
				st.info("No documents indexed yet. Open Upload Document to add a PDF.")
			else:
				for document in st.session_state.uploaded_papers.values():
					with st.container(border=True):
						st.markdown(f"### {document['filename']}")
						st.caption(document["document_id"])
						if st.button("Ask this paper", key=f"ask_{document['document_id']}"):
							st.session_state.active_view = "Ask a Question"
							st.rerun()

		elif active_view == "Recent Questions":
			st.subheader("Recent questions")
			for item in st.session_state.recent_questions or [{"question": "No questions asked yet.", "document": ""}]:
				with st.container(border=True):
					st.write(item["question"])
					st.caption(item["document"])

		elif active_view == "Saved Answers":
			st.subheader("Saved answers")
			st.info("Save controls will appear beside answers as you build your research trail.")

		elif active_view == "Analytics":
			st.subheader("Analytics")
			st.caption("A compact view of this session's research activity.")
			st.metric("Questions this session", question_count)
			st.metric("Documents indexed", uploaded_count)

		elif active_view == "Settings":
			st.subheader("Settings")
			st.session_state.dark_mode = st.toggle("Dark mode preview", value=st.session_state.dark_mode)
			st.caption("Your session preferences are stored locally while this app is open.")

		elif active_view == "Help & Support":
			st.subheader("Help & Support")
			st.info("Upload a PDF, wait for indexing to finish, then ask a specific question. Answers include the source passages used.")

		elif active_view == "Discover research":
			st.subheader("Find a research thread")
			st.caption("Search the indexed literature and open the papers worth reading.")
			with st.form("research_search"):
				query = st.text_input("Research question or topic", placeholder="e.g. transformer-based medical image segmentation")
				col1, col2, col3 = st.columns([1, 1, 2])
				with col1:
					candidate_k = st.number_input("Candidates", min_value=1, max_value=500, value=20)
				with col2:
					final_k = st.number_input("Results", min_value=1, max_value=10, value=3)
				with col3:
					st.write("")
					submitted = st.form_submit_button("✦ Search the index", type="primary", use_container_width=True)
			if submitted:
				if not query.strip():
					st.warning("Enter a topic or question first.")
				elif final_k > candidate_k:
					st.warning("Results must be less than or equal to candidates.")
				else:
					with st.spinner("Ranking the literature..."):
						try:
							st.session_state.search = api().search(query, int(candidate_k), int(final_k))
						except ApiError as error:
							show_error(error)
			search = st.session_state.get("search")
			if search:
				st.caption(f"{search.get('returned_count', len(search.get('papers', [])))} papers returned for {search.get('query', '')}")
				for paper in search.get("papers", []):
					with st.container(border=True):
						st.markdown(f'<div class="meta">RANK {paper.get("rank", "-")} · SCORE {float(paper.get("final_score", 0)):.3f}</div>', unsafe_allow_html=True)
						st.subheader(paper.get("title") or "Untitled paper")
						st.write(paper.get("summary") or "No summary was returned for this result.")

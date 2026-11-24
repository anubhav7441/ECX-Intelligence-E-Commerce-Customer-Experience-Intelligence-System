"""Builds ECX-Intelligence_Presentation.pptx (modelled on the DermaScan sample layout)."""
from pathlib import Path
from PIL import Image, ImageChops
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

D = Path(__file__).resolve().parent.parent          # docs/
U = D / "diagrams" / "uml"
IMG = Path(__file__).resolve().parent / "img"
NAVY, RED, INK, GREY = RGBColor(0x2E,0x3A,0x8C), RGBColor(0x9B,0x1B,0x30), RGBColor(0x22,0x26,0x3A), RGBColor(0x5A,0x60,0x78)
FONT = "Calibri"

def crop(src, name):
    im = Image.open(src).convert("RGB")
    bg = Image.new("RGB", im.size, (255,255,255))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        pad = 14
        box = (max(0,box[0]-pad), max(0,box[1]-pad), min(im.width,box[2]+pad), min(im.height,box[3]+pad))
        im = im.crop(box)
    out = IMG / name
    im.save(out)
    return out, im.width / im.height

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
state = {"n": 0}

def grad(shape, c1, c2, angle):
    shape.fill.gradient(); shape.fill.gradient_angle = angle
    st = shape.fill.gradient_stops
    st[0].color.rgb, st[0].position = c1, 0
    st[1].color.rgb, st[1].position = c2, 1
    shape.line.fill.background()

def text(slide, x, y, w, h, s, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = s; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = FONT
    return tb

def bullets(slide, x, y, w, h, items, size=17, gap=6):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        b = p.add_run(); b.text = "• "; b.font.color.rgb = RED; b.font.size = Pt(size); b.font.name = FONT
        if isinstance(it, tuple):
            r1 = p.add_run(); r1.text = it[0] + " "; r1.font.bold = True
            r2 = p.add_run(); r2.text = it[1]
            runs = (r1, r2)
        else:
            r = p.add_run(); r.text = it; runs = (r,)
        for r in runs:
            r.font.size = Pt(size); r.font.name = FONT; r.font.color.rgb = INK
    return tb

def base(title, sub=None):
    s = prs.slides.add_slide(BLANK); state["n"] += 1
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = RGBColor(0xFC,0xFB,0xFA)
    strip = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.33), prs.slide_height); grad(strip, NAVY, RED, 270)
    num = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(13.333-0.5), 0, Inches(0.5), Inches(0.5))
    num.fill.solid(); num.fill.fore_color.rgb = NAVY; num.line.fill.background()
    tf = num.text_frame; tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = tf.paragraphs[0].add_run(); r.text = str(state["n"]); r.font.size = Pt(13); r.font.bold = True; r.font.color.rgb = RGBColor(255,255,255); r.font.name = FONT
    text(s, 0.95, 0.5, 11.5, 0.7, title.upper(), 30, True, NAVY)
    a = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.95), Inches(1.18), Inches(2.4), Inches(0.05)); a.fill.solid(); a.fill.fore_color.rgb = NAVY; a.line.fill.background()
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(3.35), Inches(1.18), Inches(2.0), Inches(0.05)); b.fill.solid(); b.fill.fore_color.rgb = RED; b.line.fill.background()
    if sub: text(s, 0.95, 1.28, 11.5, 0.4, sub, 13, False, GREY)
    f = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.95), Inches(7.0), Inches(1.7), Inches(0.02)); f.fill.solid(); f.fill.fore_color.rgb = RED; f.line.fill.background()
    text(s, 0.9, 7.03, 8, 0.3, "ECX-Intelligence  |  Final-year major project", 10, False, RED)
    return s

def card(s, x, y, w, h, head, body, hc=NAVY, size=14, bsize=None):
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    c.adjustments[0] = 0.07; c.fill.solid(); c.fill.fore_color.rgb = RGBColor(255,255,255)
    c.line.color.rgb = RGBColor(0xDD,0xDF,0xF0); c.line.width = Pt(1.25)
    text(s, x+0.15, y+0.12, w-0.3, 0.5, head, size+2, True, hc)
    text(s, x+0.15, y+0.62, w-0.3, h-0.7, body, bsize or size, False, INK)

def pic(s, path, x, y, w, h, ratio):
    """place picture inside box (x,y,w,h) keeping aspect"""
    if w / h > ratio: ph = h; pw = h * ratio
    else: pw = w; ph = w / ratio
    s.shapes.add_picture(str(path), Inches(x + (w-pw)/2), Inches(y + (h-ph)/2), Inches(pw), Inches(ph))

def diagram_slide(title, sub, file, items, wide=False, note=None):
    s = base(title, sub)
    p, ratio = crop(file, Path(file).stem + "_c.png")
    if ratio >= 1.35:      # wide picture: full width, short notes underneath
        pic(s, p, 0.95, 1.65, 11.9, 4.45, ratio)
        bullets(s, 0.95, 6.1, 11.9, 0.9, items[:3], 12, 1)
    else:                  # tall picture: large picture left, notes right
        pic(s, p, 0.95, 1.65, 8.4, 5.3, ratio)
        bullets(s, 9.5, 1.8, 3.5, 5.0, items, 14, 8)
    return s

# ---------------- 1 Title ----------------
s = prs.slides.add_slide(BLANK); state["n"] += 1
bgs = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height); grad(bgs, RGBColor(0x2E,0x3A,0x8C), RGBColor(0x9B,0x1B,0x30), 330)
text(s, 0.9, 0.55, 8, 0.5, "YOUR UNIVERSITY NAME", 20, True, RGBColor(255,255,255))
text(s, 0.9, 2.1, 8.2, 2.4, "ECX-INTELLIGENCE: E-COMMERCE CUSTOMER EXPERIENCE INTELLIGENCE SYSTEM", 38, True, RGBColor(255,255,255))
text(s, 0.9, 4.75, 7.6, 0.9, "A web application that finds why online-shop customers are unhappy, predicts which orders are at risk, and suggests what to do about it.", 16, False, RGBColor(255,255,255))
text(s, 0.9, 6.0, 7, 0.4, "Guide: [Guide name]", 16, True, RGBColor(255,255,255))
text(s, 9.0, 4.7, 4.1, 1.8, "Group Name: [Group name]\nGroup Members:\n1. [Member name]\n2. [Member name]\n3. [Member name]", 15, False, RGBColor(255,255,255))

# ---------------- 2 Abstract ----------------
s = base("Abstract", "What the system does")
bullets(s, 0.95, 1.8, 11.8, 5, [
 ("ECX-Intelligence", "is a web application that studies customer experience in online shopping using the public Olist Brazilian e-commerce dataset (about 95,800 delivered and reviewed orders), and any CSV file a user uploads."),
 "It shows where customers are unhappy: late deliveries, weak sellers and regions, and complaint themes in review text.",
 ("Machine-learning models", "(XGBoost and others) predict the chance of a bad review, using only information that is known at that time. SHAP explains each prediction."),
 ("A team of 5 agents", "investigates a seller or a month, saves evidence for every finding, and proposes actions that a human must approve."),
 "It also has a what-if simulator, a product recommender, a grounded assistant and plain-English advice cards.",
 "Built with Python, Flask, SQLite, pandas, scikit-learn, XGBoost and SHAP. It is a research prototype, not a production system.",
], 17, 9)

# ---------------- 3 Introduction ----------------
s = base("Introduction")
card(s, 0.95, 1.7, 3.8, 2.7, "LATE ORDERS HURT", "In the Olist data, 62.4% of late orders got a bad review, against 9.3% of on-time orders.")
card(s, 4.95, 1.7, 3.8, 2.7, "DATA IS SCATTERED", "Orders, reviews, sellers and products sit in separate tables. Managers cannot see the cause easily.")
card(s, 8.95, 1.7, 3.8, 2.7, "AI CAN GIVE EARLY WARNING", "A trained model scores each order, explains the reasons and points to what can be fixed.")
text(s, 0.95, 4.75, 6, 0.4, "Core idea", 20, True, NAVY)
text(s, 0.95, 5.3, 11.8, 1.3, "A shop owner should not have to guess why customers leave bad reviews. The system reads the data, tests the causes with statistics, explains the answer in plain words and suggests actions, while keeping a person in charge of every decision.", 17)

# ---------------- 4 Problem & objectives ----------------
s = base("Problem statement & objectives")
text(s, 0.95, 1.55, 6, 0.4, "Problem statement", 20, True, NAVY)
text(s, 0.95, 2.0, 11.8, 1.0, "Customer experience data is spread over many tables and is hard to read. Managers see unhappy customers only after the bad review is written, and most tools give charts without a clear cause or next step.", 16)
text(s, 0.95, 3.2, 6, 0.4, "Objectives", 20, True, NAVY)
bullets(s, 0.95, 3.7, 11.8, 3.2, [
 "Measure how delivery delay, sellers, regions and products relate to customer satisfaction.",
 "Predict dissatisfaction without data leakage (time-based split; checkout-time and delivery-time models).",
 "Explain every model with SHAP and show honest limits (accuracy, confidence).",
 "Investigate sellers and months with agents; store evidence; require human approval for actions.",
 "Work with any user CSV: map columns, check what can be analysed, produce a PDF report.",
 "Keep users and data safe: login, roles, per-user data, rate limits and an audit log.",
], 16, 5)

# ---------------- 5 Solution domain ----------------
s = base("Solution domain", "Domain: data science, machine learning and web engineering for e-commerce")
bullets(s, 0.95, 1.8, 11.8, 5, [
 ("Statistics:", "Mann-Whitney, chi-square, Cliff's delta, odds ratios and false-discovery-rate control test each claim."),
 ("Machine learning:", "XGBoost, logistic regression and others with a time-based split; SHAP for explanations."),
 ("Text analysis:", "Review-text complaint themes and sentiment for aspect-level insight."),
 ("Anomaly detection:", "Isolation Forest plus statistical tests flag unusual sellers and months."),
 ("Recommender systems:", "popularity, item-based and user-based CF, ALS, content-based, association rules and a hybrid."),
 ("Multi-agent design:", "a deterministic Supervisor with 5 tool-calling agents (no LLM needed)."),
 ("Web application:", "Flask blueprints, SQLite, Jinja2, Bootstrap and Chart.js, secured with CSRF, CSP and hashed passwords."),
], 16, 7)

# ---------------- 6 Requirements ----------------
s = base("Requirement gathering")
text(s, 0.95, 1.55, 5.8, 0.4, "Functional requirements", 20, True, NAVY); text(s, 0.95, 1.95, 5.8, 0.3, "What the system must do", 12, False, GREY)
bullets(s, 0.95, 2.35, 5.8, 4.6, ["Register, log in and log out; guest demo.", "Upload a CSV and confirm the column mapping.", "Run only the analyses the data supports.", "Predict the risk of a bad review for an order.", "Run what-if scenarios on delivery changes.", "Recommend products to a customer.", "Investigate a seller or month; store evidence.", "Approve or reject proposed actions.", "Admin: manage users and view the audit log.", "Download a 14-section PDF report."], 15, 4)
text(s, 7.0, 1.55, 5.8, 0.4, "Non-functional requirements", 20, True, NAVY); text(s, 7.0, 1.95, 5.8, 0.3, "How well it must work", 12, False, GREY)
bullets(s, 7.0, 2.35, 5.8, 4.6, ["Security: hashed passwords, CSRF, CSP, lockout, rate limits.", "Privacy: each user sees only their own data.", "Honesty: no leakage; limits and confidence shown.", "Runs on a normal laptop, no GPU, no internet needed.", "Repeatable: fixed seeds, saved models, 155 automatic tests passing (22 skipped).", "Works on Windows and Linux (portable model files).", "Simple language for non-technical users."], 15, 4)

# ---------------- 7 Feasibility ----------------
s = base("Feasibility")
for i, (h, b) in enumerate([("Technical feasibility", "Python, Flask, SQLite, pandas, scikit-learn, XGBoost and SHAP are free and run on a laptop. Models are small and train in minutes."),
                            ("Operational feasibility", "The user opens a web page, uploads a file and reads plain-English advice. No setup beyond running the app."),
                            ("Economic feasibility", "All tools and the Olist dataset are free for study. No extra cost beyond a laptop.")]):
    card(s, 0.95 + i*4.0, 1.7, 3.8, 3.4, h, b, size=15)
text(s, 0.95, 5.5, 11.8, 1.2, "Conclusion: the project is feasible as a working research prototype, because the data was split by time, the tests were honest and the limits are reported.", 17, True, NAVY)

# ---------------- 8 Methodology ----------------
s = base("Project planning & methodology", "Step-by-step development plan")
steps = [("Decide the requirements", "Questions, users, roles, inputs and outputs"),
         ("Prepare the data", "Clean Olist tables, build one order-level table, avoid leakage"),
         ("Analyse and model", "Statistics, XGBoost models, SHAP, anomaly and recommender models"),
         ("Build the application", "Flask blueprints, agents, security, uploads, reports, UI"),
         ("Test and improve", "Automatic tests, honest evaluation and a written list of limits")]
for i, (h, b) in enumerate(steps):
    y = 1.85 + i*1.02
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.95), Inches(y), Inches(0.62), Inches(0.62)); o.fill.solid(); o.fill.fore_color.rgb = NAVY if i % 2 == 0 else RED; o.line.fill.background()
    tf = o.text_frame; tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = tf.paragraphs[0].add_run(); r.text = f"0{i+1}"; r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = RGBColor(255,255,255)
    text(s, 1.8, y+0.05, 4.2, 0.5, h, 19, True, INK)
    text(s, 6.2, y+0.05, 6.6, 0.6, b, 15, False, GREY)

# ---------------- 9 UML overview ----------------
s = base("UML diagrams covered", "All 14 diagram types of UML 2.x, plus 2 extra software-engineering diagrams")
rows = [("Structure", "Class (2) · Object · Component · Composite structure · Package · Deployment · Profile"),
        ("Behaviour", "Use case (2) · Activity · State machine"),
        ("Interaction", "Sequence (2) · Communication · Timing · Interaction overview"),
        ("Extra (not UML)", "Data flow diagram (level 0 and 1) · Entity-relationship diagram (Olist and app)")]
for i, (h, b) in enumerate(rows):
    y = 1.85 + i*0.95
    card(s, 0.95, y, 3.0, 0.8, h, "", size=15)
    text(s, 4.15, y+0.15, 8.7, 0.6, b, 16, False, INK)
text(s, 0.95, 5.8, 11.8, 1.0, "Every diagram was drawn from the real code: Flask blueprints, the ecx2 agent classes, the SQLite tables and the Action status rules in ecx2/store.py. Source files (.puml) are in docs/diagrams/uml.", 14, False, GREY)

# ---------------- 10 Architecture ----------------
s = base("System design - architecture", "The main parts of the system and how they connect")
p, r = crop(D / "diagrams" / "architecture.png", "arch_c.png")
pic(s, p, 0.95, 1.7, 7.9, 5.2, r)
bullets(s, 9.05, 1.8, 3.95, 5, [("Browser:", "pages built with Jinja2 and Chart.js."), ("Security layer:", "login guard, CSRF, CSP, rate limits."), ("Blueprints:", "pages, api, v2, md, auth, demo, insights, admin, rec."), ("Engines:", "ML, agents, simulator, multi-dataset, recommender."), ("Storage:", "SQLite files and parquet / model files.")], 15, 8)

# ---------------- UML diagrams ----------------
diagram_slide("UML - use case diagram (1/2)", "Access, data and administration", U/"01a_usecase_access.png", [
  ("Actors:", "Guest, Registered User and Administrator (Administrator inherits from Registered User)."),
  ("Include:", "Upload and Run analyses always need Log in."),
  ("Extend:", "Download PDF extends Run analyses; Try guest demo can lead to Log in."),
  "Guest demo is limited to a 500-row sample."])
diagram_slide("UML - use case diagram (2/2)", "Analytics and decision support", U/"01b_usecase_analytics.png", [
  ("Registered User:", "predicts risk, runs what-if, sees recommendations, investigates and approves actions."),
  ("Include:", "Investigate always records evidence."),
  "Every decision is made by a person, never by the agents."])
diagram_slide("UML - class diagram (1/2)", "Accounts, datasets and analysis runs", U/"02_class_data.png", [
  ("User:", "role is an enumeration (user, admin)."),
  ("Dataset:", "owned by one user, has exactly one Mapping (composition)."),
  ("AnalysisRun:", "one run may produce one Report (aggregation)."),
  "Multiplicities (1, 0..*) and visibility (-, +) are shown on every link."])
diagram_slide("UML - class diagram (2/2)", "Agent system and evidence", U/"03_class_agents.png", [
  ("Generalisation:", "6 agent classes inherit from the abstract class Base."),
  ("Composition:", "a report owns its agent reports, findings and tool calls."),
  ("Dependency:", "Supervisor creates the InvestigationReport."),
  "Evidence backs each Finding; a ProposedAction becomes an Action with an ActionStatus."], wide=True)
diagram_slide("UML - object diagram", "One real-looking snapshot of the running system", U/"04_object.png", [
  ("Instances:", "each box is an object written as name : Class with attribute values."),
  "The snapshot shows an analyst who owns a dataset, starts a run and gets an investigation with one finding, its evidence and a proposed action.",
  "Values are sample values for illustration."])
diagram_slide("UML - sequence diagram (1/2)", "Upload a dataset and run analyses", U/"08_sequence.png", [
  ("Lifelines:", "browser, security guard, blueprint, engine and store."),
  ("Fragments:", "alt (valid session or not), opt (user selects analyses) and loop (each analysis)."),
  "Solid arrows are calls; dashed arrows are replies; numbers show the order."])
diagram_slide("UML - sequence diagram (2/2)", "Investigate a seller and approve an action", U/"08b_sequence_investigation.png", [
  ("Supervisor:", "plans, calls the agents in order and gathers their reports."),
  "Every finding is saved as evidence before the next step.",
  ("Human step:", "an action stays proposed until a named person approves it with a note.")], wide=True)
diagram_slide("UML - communication diagram", "Predict the risk of one order (same message flow, shown as links)", U/"09_communication.png", [
  ("Objects and links:", "the lines show who talks to whom."),
  ("Numbering:", "1, 2, 3 give the order; 3.1, 3.2, 3.3 are nested calls inside message 3."),
  "It shows the same interaction as a sequence diagram but stresses structure."], wide=True)
diagram_slide("UML - activity diagram", "From uploading a file to reading the advice (with swim-lanes)", U/"10_activity.png", [
  ("Swim-lanes:", "User and System."),
  ("Control nodes:", "initial node, decisions, fork and join (three checks run in parallel), final nodes."),
  "An invalid file ends the flow early with an error message."])
diagram_slide("UML - state machine diagram", "Life of a proposed action (rules from ecx2/store.py)", U/"11_state.png", [
  ("States:", "proposed, approved, in_progress, rejected, completed."),
  ("Guards:", "a note is required for rejection and completion."),
  "rejected and completed are final: no further change is allowed.",
  "Every change is written to action_history."])
diagram_slide("UML - timing diagram", "Login lockout: 5 failed attempts in 5 minutes", U/"12_timing.png", [
  ("Lifelines:", "the client, the auth service state and the failure count against a time axis."),
  ("Rule:", "after 5 failed logins in 5 minutes the account key is locked for 5 minutes."),
  "Time constraints are shown with the double arrows."], wide=True)
diagram_slide("UML - component diagram", "Main software components and how they use each other", U/"05_component.png", [
  ("Tiers:", "presentation, application (blueprints) and intelligence."),
  ("Sockets:", "the half-circle shows a required interface."),
  ("Dependencies:", "dashed arrows show which engine or store is used."),
  "Two data stores: SQLite and file storage."])
diagram_slide("UML - package diagram", "Python packages and their dependencies", U/"06_package.png", [
  ("Packages:", "app, ml, ecx2, multids, recsys and research."),
  ("«import» / «access»:", "show how one package uses another."),
  "The routes layer depends on the engines; the engines never import the routes."])
diagram_slide("UML - deployment diagram", "Where the software runs", U/"07_deployment.png", [
  ("Nodes:", "the client device with a browser and the application host with the Python runtime."),
  ("Artifacts:", "run.py, the packages, SQLite files, model files and datasets."),
  ("Link:", "HTTP on localhost:5000."),
  "No cloud service is needed."], wide=True)
diagram_slide("UML - composite structure diagram", "Inside the investigation engine", U/"13_composite.png", [
  ("Parts:", "the Supervisor and five agent parts that exist during one investigation."),
  ("Ports:", "request in; report and evidence out."),
  "Agents talk only to the Supervisor and to the evidence port, never to each other."], wide=True)
diagram_slide("UML - interaction overview diagram", "How the individual sequences fit together in one analyst session", U/"14_interaction_overview.png", [
  ("Frame iod:", "an activity-style flow whose nodes are interaction references (ref)."),
  ("Control:", "decision on login result, fork for three parallel features, decision on action approval."),
  "Each ref points to one sequence diagram."])
diagram_slide("UML - profile diagram", "Small UML profile for this project", U/"15_profile.png", [
  ("Stereotypes:", "Agent, Blueprint, ModelArtifact and HumanApproval extend UML metaclasses."),
  ("Tagged values:", "uses_llm = false, required = true."),
  "A profile lets the project use its own words inside standard UML."], wide=True)
diagram_slide("Data flow diagram - level 0 (context)", "Extra diagram (not part of UML): the system as one process", D/"diagrams"/"uml"/"16a_dfd0.png", [
  ("External entities:", "users, administrator and the datasets."),
  "Arrows are data flows; no data stores are shown at this level."])
diagram_slide("Data flow diagram - level 1", "Extra diagram (not part of UML): the five main processes", U/"16b_dfd1.png", [
  ("Processes:", "1.0 login, 2.0 ingest, 3.0 analyse, 4.0 predict and recommend, 5.0 investigate."),
  ("Stores:", "D1 to D4 are open-ended boxes; every flow touches a process.")], wide=True)
diagram_slide("ER diagram - Olist database", "Extra diagram (not part of UML): crow's-foot notation", U/"17_er.png", [
  ("Keys:", "PK and FK are marked on every table."),
  ("Links:", "one customer has many orders; an order has many items, reviews and payments."),
  "Product and seller link to order items."])
diagram_slide("ER diagram - application database", "Extra diagram (not part of UML): users, runs, evidence and actions", U/"18_er_app.png", [
  ("Auth store:", "users own analysis runs; the audit log can refer to a user."),
  ("Evidence store:", "actions have a history; evidence is cited by actions."),
  "Created with SQLite; foreign keys are enabled."], wide=True)

# ---------------- Tech stack ----------------
s = base("Technology stack & implementation", "Tools used in the project")
tech = [("Python 3.12", "Main language for the web app, analysis and models"), ("Flask + Jinja2", "App factory, 9 blueprints, secure pages"),
        ("pandas, SciPy, statsmodels", "Cleaning and statistical tests"), ("scikit-learn + XGBoost", "Prediction models and anomaly detection"),
        ("SHAP", "Explains each prediction"), ("SQLite + SQLAlchemy", "Users, evidence, actions and Olist data"),
        ("Bootstrap 5 + Chart.js", "Dashboard and charts, bundled offline"), ("Pydantic", "Typed reports from the agents"),
        ("Git, pytest", "155 tests passing, 22 skipped on the author's Windows run")]
for i, (h, b) in enumerate(tech):
    card(s, 0.95 + (i % 3)*4.0, 1.75 + (i // 3)*1.7, 3.8, 1.55, h, b, size=14, bsize=13)

# ---------------- Data & training ----------------
s = base("Datasets & modelling", "Where the data came from and how the models learned")
card(s, 0.95, 1.7, 5.9, 2.6, "Olist dataset", "95,824 delivered and reviewed orders after cleaning. Nine tables joined into one order-level table (95,446 rows, 76 columns). Split by time, not at random.", size=15)
card(s, 7.0, 1.7, 5.8, 2.6, "Other data", "Any user CSV can be uploaded (guest demo uses a 500-row sample). The recommender uses 101,521 customer-product rows.", size=15)
text(s, 0.95, 4.55, 6, 0.4, "How we modelled", 20, True, NAVY)
bullets(s, 0.95, 5.05, 11.8, 2, ["S0 model uses checkout-time information only; S1 adds delivery-time information.", "Class weights handle the small share of unhappy customers.", "Predictions are tested on later orders that were never used for training."], 16, 5)

# ---------------- Results ----------------
s = base("Results", "Measured on later orders that the models had not seen")
kp = [("62.4%", "late orders with a bad review", NAVY), ("9.3%", "on-time orders with a bad review", NAVY), ("0.57-0.61", "ROC-AUC, checkout-time model (S0)", RED), ("0.69-0.71", "ROC-AUC, delivery-time model (S1)", NAVY)]
for i, (v, l, c) in enumerate(kp):
    cd = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.95 + i*3.0), Inches(1.7), Inches(2.8), Inches(2.1)); cd.adjustments[0] = 0.07
    cd.fill.solid(); cd.fill.fore_color.rgb = RGBColor(255,255,255); cd.line.color.rgb = RGBColor(0xDD,0xDF,0xF0)
    text(s, 0.95 + i*3.0, 1.95, 2.8, 0.8, v, 32, True, c, PP_ALIGN.CENTER)
    text(s, 1.05 + i*3.0, 2.85, 2.6, 0.9, l, 14, False, INK, PP_ALIGN.CENTER)
bullets(s, 0.95, 4.1, 11.8, 2.8, [
 "Late delivery is the strongest effect: Cliff's delta -0.64 and odds ratio 16.3. In SHAP, delay days is the top feature.",
 "Checkout-time prediction is weak (AUC about 0.6): most unhappiness shows only after delivery.",
 "Recommender accuracy is very low: only 5.7% of customers bought two or more products, and the hybrid did not beat popularity.",
 "155 automatic tests pass (22 skipped); users cannot see each other's data."], 15, 6)

# ---------------- Demo ----------------
s = base("Live demonstration", "Demo plan (get this ready before the presentation)")
demo = [("Open the website", "Run run.py and open the landing page; try the guest demo"), ("Log in and upload", "Upload samples/01_good_shop_orders.csv and confirm the mapping"),
        ("Read the advice", "Show the Overview dashboard and the advice cards"), ("Predict and simulate", "Order Risk Check, What-If simulator and Recommendations"),
        ("Investigate and approve", "Investigate a seller, then approve one proposed action")]
for i, (h, b) in enumerate(demo):
    y = 1.85 + i*1.02
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.95), Inches(y), Inches(0.62), Inches(0.62)); o.fill.solid(); o.fill.fore_color.rgb = NAVY if i % 2 == 0 else RED; o.line.fill.background()
    tf = o.text_frame; tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = tf.paragraphs[0].add_run(); r.text = f"0{i+1}"; r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = RGBColor(255,255,255)
    text(s, 1.8, y+0.05, 4.2, 0.5, h, 19, True, INK); text(s, 6.2, y+0.05, 6.6, 0.6, b, 15, False, GREY)

# ---------------- Screens ----------------
s = base("Application screens", "Screenshots from the running application")
for i, (f, cap) in enumerate([("01_landing.png", "Landing page"), ("03_guest_demo.png", "Guest demo dashboard"), ("12_recommendations.png", "Product recommendations")]):
    p, r = crop(D / "screenshots" / f, f"shot{i}.png")
    pic(s, p, 0.95 + i*4.0, 1.9, 3.8, 3.8, r); text(s, 0.95 + i*4.0, 5.55, 3.8, 0.4, cap, 14, True, NAVY, PP_ALIGN.CENTER)

# ---------------- Testing & limitations ----------------
s = base("Testing & limitations")
text(s, 0.95, 1.55, 5.8, 0.4, "Testing", 20, True, NAVY)
bullets(s, 0.95, 2.1, 5.8, 4.8, ["155 automatic tests passed, 22 skipped, 0 failed (author's Windows run).", "Tests cover login, roles, CSRF, uploads, privacy, models and the recommender.", "Time-based split and leakage checks on the models.", "Bad files, tiny files and non-order files were tried.", "Windows model-loading problem fixed with portable model files."], 15, 6)
text(s, 7.0, 1.55, 5.8, 0.4, "Current limitations", 20, True, RED)
bullets(s, 7.0, 2.1, 5.8, 4.8, ["Olist is old (2016-2018) Brazilian data; results are associations, not proof of cause.", "Checkout-time prediction is weak (AUC about 0.6).", "Recommender accuracy is very low because most customers bought once.", "No usability study with real users; no live local-LLM test.", "Advice labels depend mainly on sample size.", "It is a research prototype, not a production system."], 15, 6)

# ---------------- Future / conclusion / thanks ----------------
s = base("Future scope")
bullets(s, 0.95, 1.8, 11.8, 5, ["Test on more real shop data and other countries.", "Add significance testing to the advice confidence labels.", "Flag out-of-range or invalid values in the data-quality check.", "Run a usability study with real shop managers.", "Add Hindi and other languages for review-text analysis.", "Deploy with HTTPS and a production database.", "Use richer recommender data (more repeat buyers, ratings)."], 18, 9)
s = base("Conclusion and Q&A")
bullets(s, 0.95, 1.8, 11.8, 5, ["ECX-Intelligence shows how data, statistics and machine learning can explain customer unhappiness and suggest actions.", "Agents collect evidence; a human approves every action; the whole system works offline on a laptop.", "Honest testing matters more than a large accuracy number: the system reports weak results and limits openly.", "UML models (14 diagram types) document the structure and behaviour of the system.", "It is a research prototype and supports decisions; it does not replace human judgement."], 18, 9)
s = base("Thank you!")
text(s, 0.95, 1.9, 11, 0.5, "Group Name: [Group name]", 20); text(s, 0.95, 2.5, 11, 0.5, "Guide: [Guide name]", 20); text(s, 0.95, 3.1, 11, 0.5, "Members: [Member 1], [Member 2], [Member 3]", 20)
text(s, 0.95, 4.6, 11, 0.5, "For any query or suggestion, please mail us at [your.email@example.com]", 16, False, GREY)

out = Path(__file__).resolve().parent / "ECX-Intelligence_Presentation.pptx"
prs.save(out); print("saved", out, state["n"], "slides")

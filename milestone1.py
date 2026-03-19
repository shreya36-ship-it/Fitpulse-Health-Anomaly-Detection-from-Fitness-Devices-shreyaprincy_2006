import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_con
g(

page_title="Fitness Health Data — Pro Pipeline",
layout="wide",
page_icon="💙"
)

# ==================================================
# SESSION STATE
# ==================================================
if "theme" not in st.session_state:
st.session_state.theme = "dark"

if "df" not in st.session_state:
st.session_state.df = None

if "cleaned_df" not in st.session_state:
st.session_state.cleaned_df = None

# ==================================================
# THEME TOGGLE FUNCTION
# ==================================================
def toggle_theme():
st.session_state.theme = (
"light" if st.session_state.theme == "dark" else "dark"

)

# ==================================================
# THEME COLORS
# ==================================================
if st.session_state.theme == "dark":
main_bg = "#0a0f1c"
right_bg = "#111827"
text_color = "white"
else:
main_bg = "#f5f7fb"
right_bg = "#dbeafe"
text_color = "black"

# ==================================================
# APPLY CSS
# ==================================================
st.markdown(f"""
<style>
.stApp {{
background-color: {main_bg};
color: {text_color};
}}
.right-panel {{
background-color: {right_bg};
padding: 20px;
border-radius: 12px;
height: 100%;
}}
.pipeline-step {{
padding: 12px;
margin-bottom: 10px;
border-radius: 8px;

background-color: rgba(255,255,255,0.05);
}}
h1, h2, h3 {{
color: #1f77
;

}}
.stButton>button {{
background: linear-gradient(90deg,#1f77

,#004aad);

color: white;
border-radius: 8px;
font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LAYOUT
# ==================================================
main_col, right_col = st.columns([4, 1])

# ==================================================
# RIGHT SIDE PANEL
# ==================================================
with right_col:
st.markdown("<div class='right-panel'>", unsafe_allow_html=True)

st.button("🌗 Toggle Dark / Light", on_click=toggle_theme)

st.markdown("###🚀 Pipeline Status")

def status_icon(condition):
return "✅" if condition else "⏳"

st.markdown(

f"<div class='pipeline-step'>{status_icon(st.session_state.df is not None)}
📂 Upload</div>",
unsafe_allow_html=True
)
st.markdown(
f"<div class='pipeline-step'>{status_icon(st.session_state.df is not None)}
🔍 Null Check</div>",
unsafe_allow_html=True
)
st.markdown(
f"<div class='pipeline-step'>{status_icon(st.session_state.cleaned_df is not
None)} ⚙ Preprocess</div>",
unsafe_allow_html=True
)
st.markdown(
f"<div class='pipeline-step'>{status_icon(st.session_state.cleaned_df is not
None)} 👁 Preview</div>",
unsafe_allow_html=True
)
st.markdown(
f"<div class='pipeline-step'>{status_icon(st.session_state.cleaned_df is not
None)} 📊 EDA</div>",
unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# MAIN CONTENT
# ==================================================
with main_col:

st.title("💙 Fitness Health Data — Pro Pipeline")

# ---------------- STEP 1 ----------------
st.header("📂 Step 1 · Upload Dataset")
uploaded_
le = st.

le_uploader("Upload CSV File", type=["csv"])

if uploaded_
le:

df = pd.read_csv(uploaded_
le)

st.session_state.df = df

rows, cols = df.shape
total_nulls = df.isnull().sum().sum()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", rows)
c2.metric("Columns", cols)
c3.metric("Total Nulls", total_nulls)

st.success("Dataset Loaded Successfully!")

# ---------------- STEP 2 ----------------
if st.session_state.df is not None:
st.header("🔍 Step 2 · Check Null Values")

df = st.session_state.df
null_counts = df.isnull().sum()
st.dataframe(null_counts)

# NULL PERCENTAGE GRAPH
st.subheader("📊 Missing Data Percentage")

null_percent = (null_counts / len(df)) * 100

plt.
gure(
gsize=(8,4))

null_percent.sort_values(ascending=False).plot(kind="bar")
plt.ylabel("Missing Percentage (%)")
plt.xticks(rotation=45)
st.pyplot(plt)

# ---------------- STEP 3 ----------------
if st.session_state.df is not None:
st.header("⚙ Step 3 · Preprocess Data")

if st.button("Run Preprocessing"):

df = st.session_state.df.copy()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce", day

rst=True)

numeric_cols = [
"Hours_Slept",
"Water_Intake (Liters)",
"Active_Minutes",
"Heart_Rate (bpm)"
]

df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
lambda x: x.interpolate(method="linear")
)

df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
lambda x: x.
ll().b
ll()

)

df["Workout_Type"] = df["Workout_Type"].

llna("No Workout")

st.session_state.cleaned_df = df

st.success("Preprocessing Completed Successfully!")

# ---------------- STEP 4 ----------------
if st.session_state.cleaned_df is not None:
st.header("👁 Step 4 · Preview Cleaned Data")
df = st.session_state.cleaned_df
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
st.dataframe(df.head(20))

# ---------------- STEP 5 ----------------
if st.session_state.cleaned_df is not None:
st.header("📊 Step 5 · EDA")

if st.button("Run EDA"):

numeric_cols = [
"Steps_Taken",
"Calories_Burned",
"Hours_Slept",
"Active_Minutes",
"Heart_Rate (bpm)",
"Stress_Level (1
10)"

]

g, axes = plt.subplots(3,2,

gsize=(12,8))

axes = axes.

atten()

fori, col in enumerate(numeric_cols):
sns.histplot(df[col], kde=True, ax=axes[i])
axes[i].set_title(col)

plt.tight_layout()
st.pyplot(
g)
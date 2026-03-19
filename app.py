import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from prophet import Prophet

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



# -------------------------------------------------------
# PAGE CONFIG (ONLY ONCE)
# -------------------------------------------------------

st.set_page_config(
    page_title="FitPulse ML Dashboard",
    page_icon="💓",
    layout="wide"
)
# -------------------------------------------------------
# SIDEBAR MILESTONE SELECTOR
# -------------------------------------------------------

st.sidebar.title("💓 FitPulse Dashboard")

milestone = st.sidebar.radio(
    "🚀 Select Project Milestone",
    (
        "📊 Milestone 1 — Data Processing Pipeline",
        "🤖 Milestone 2 — ML Analytics Pipeline"
    )
)

st.sidebar.markdown("---")


# -------------------------------------------------------
# GLOBAL DARK CORPORATE UI
# -------------------------------------------------------

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f172a,#020617);
color:white;
}

/* Sidebar */

[data-testid="stSidebar"]{
background:#020617;
}

[data-testid="stSidebar"] *{
color:white !important;
}

/* Buttons */

button{
background:#2563eb !important;
color:white !important;
border-radius:8px;
}

/* Headers */

h1,h2,h3{
color:#60a5fa;
}

</style>
""", unsafe_allow_html=True)



# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

# -------------------------------------------------------
# SIDEBAR EXECUTED FEATURES
# -------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Pipeline Features")

st.sidebar.write("📂 **Data Upload**")
st.sidebar.caption("Upload Fitbit and health datasets for analysis")

st.sidebar.write("🔍 **Missing Value Analysis**")
st.sidebar.caption("Detect null values and visualize missing data %")

st.sidebar.write("🧹 **Data Preprocessing**")
st.sidebar.caption("Interpolation, forward fill and cleaning of records")

st.sidebar.write("👀 **Clean Data Preview**")
st.sidebar.caption("View processed dataset after preprocessing")

st.sidebar.write("📊 **Exploratory Data Analysis**")
st.sidebar.caption("Distribution plots for health metrics")

st.sidebar.write("❤️ **Heart Rate Processing**")
st.sidebar.caption("Resampling second-level HR data to minute level")

st.sidebar.write("🔬 **TSFresh Feature Engineering**")
st.sidebar.caption("Automatic time-series feature extraction")

st.sidebar.write("🔥 **Feature Heatmap Visualization**")
st.sidebar.caption("Normalized feature matrix heatmap")

st.sidebar.write("📈 **Heart Rate Forecasting**")
st.sidebar.caption("Future HR prediction using Prophet")

st.sidebar.write("🤖 **User Clustering**")
st.sidebar.caption("KMeans clustering based on activity metrics")

st.sidebar.write("📉 **PCA Visualization**")
st.sidebar.caption("2D dimensionality reduction of clusters")

st.sidebar.write("🧠 **t-SNE Projection**")
st.sidebar.caption("Advanced visualization of user clusters")

st.sidebar.markdown("---")

st.sidebar.success("🚀 ML Pipeline Active")


# =====================================================
# MILESTONE 1 FUNCTION
# =====================================================

def milestone1():

    if "df" not in st.session_state:
        st.session_state.df = None

    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None

    st.title("💙 Fitness Health Data — Pro Pipeline")

    # STEP 1
    st.header("📂 Step 1 · Upload Dataset")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        rows, cols = df.shape
        total_nulls = df.isnull().sum().sum()

        c1,c2,c3 = st.columns(3)

        c1.metric("Rows", rows)
        c2.metric("Columns", cols)
        c3.metric("Total Nulls", total_nulls)

        st.success("Dataset Loaded Successfully!")

    # STEP 2
    if st.session_state.df is not None:

        st.header("🔍 Step 2 · Check Null Values")

        df = st.session_state.df

        null_counts = df.isnull().sum()

        st.dataframe(null_counts)

        st.subheader("📊 Missing Data Percentage")

        null_percent = (null_counts/len(df))*100

        fig,ax = plt.subplots()

        null_percent.sort_values(ascending=False).plot(kind="bar",ax=ax)

        ax.set_ylabel("Missing %")

        st.pyplot(fig)

    # STEP 3
    if st.session_state.df is not None:

        st.header("⚙ Step 3 · Preprocess Data")

        if st.button("Run Preprocessing"):

            df = st.session_state.df.copy()

            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"],errors="coerce")

            numeric_cols = [
            "Hours_Slept",
            "Water_Intake (Liters)",
            "Active_Minutes",
            "Heart_Rate (bpm)"
            ]

            numeric_cols = [c for c in numeric_cols if c in df.columns]

            if "User_ID" in df.columns and numeric_cols:

                df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
                lambda x: x.interpolate()
                )

                df[numeric_cols] = df.groupby("User_ID")[numeric_cols].transform(
                lambda x: x.ffill().bfill()
                )

            if "Workout_Type" in df.columns:
                df["Workout_Type"] = df["Workout_Type"].fillna("No Workout")

            st.session_state.cleaned_df = df

            st.success("Preprocessing Completed")

    # STEP 4
    if st.session_state.cleaned_df is not None:

        st.header("👁 Step 4 · Preview Cleaned Data")

        df = st.session_state.cleaned_df

        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

        st.dataframe(df.head(20))

    # STEP 5
    if st.session_state.cleaned_df is not None:

        st.header("📊 Step 5 · EDA")

        if st.button("Run EDA"):

            df = st.session_state.cleaned_df

            numeric_cols = [
            "Steps_Taken",
            "Calories_Burned",
            "Hours_Slept",
            "Active_Minutes",
            "Heart_Rate (bpm)",
            "Stress_Level (1-10)"
            ]

            numeric_cols=[c for c in numeric_cols if c in df.columns]

            fig,axes = plt.subplots(3,2,figsize=(12,8))

            axes = axes.flatten()

            for i,col in enumerate(numeric_cols):

                sns.histplot(df[col],kde=True,ax=axes[i])

                axes[i].set_title(col)

            plt.tight_layout()

            st.pyplot(fig)



# =====================================================
# MILESTONE 2 FUNCTION
# =====================================================
def milestone2():

    st.title("💓 FitPulse ML Dashboard")
    st.markdown("### 📂 Upload Fitbit Dataset Files")

    # FILE UPLOAD (HORIZONTAL)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        daily_file = st.file_uploader("dailyActivity", type=["csv"])
    with col2:
        steps_file = st.file_uploader("hourlySteps", type=["csv"])
    with col3:
        int_file = st.file_uploader("hourlyIntensities", type=["csv"])
    with col4:
        sleep_file = st.file_uploader("minuteSleep", type=["csv"])
    with col5:
        hr_file = st.file_uploader("heartrate", type=["csv"])

    if daily_file and steps_file and int_file and sleep_file and hr_file:

        st.success("✅ All files uploaded")

        # LOAD
        daily = pd.read_csv(daily_file)
        sleep = pd.read_csv(sleep_file)
        hr = pd.read_csv(hr_file)

        # DATE
        daily["ActivityDate"] = pd.to_datetime(daily["ActivityDate"])
        sleep["date"] = pd.to_datetime(sleep["date"])
        hr["Time"] = pd.to_datetime(hr["Time"])

        # HEART RATE PROCESS
        hr_minute = (
            hr.set_index("Time")
            .groupby("Id")["Value"]
            .resample("1min")
            .mean()
            .reset_index()
        )

        hr_minute.columns = ["Id", "Time", "HeartRate"]
        hr_minute.dropna(inplace=True)
        hr_minute["Date"] = hr_minute["Time"].dt.date

        # SLEEP
        sleep["Date"] = sleep["date"].dt.date

        sleep_daily = (
            sleep.groupby(["Id", "Date"])
            .agg(TotalSleepMinutes=("value", "count"))
            .reset_index()
        )

        # MASTER DATASET
        master = daily.copy()
        master["Date"] = master["ActivityDate"].dt.date

        master = master.merge(sleep_daily, on=["Id", "Date"], how="left")
        master.fillna(0, inplace=True)

        st.subheader("📊 Master Dataset")
        st.dataframe(master.head())

        # CLUSTERING
        cluster_cols = [
            "TotalSteps", "Calories",
            "VeryActiveMinutes",
            "SedentaryMinutes",
            "TotalSleepMinutes"
        ]

        cluster_features = master.groupby("Id")[cluster_cols].mean()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_features)

        # ELBOW CURVE
        st.subheader("📊 Elbow Curve")

        inertias = []
        K_range = range(2, 8)

        for k in K_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        fig1, ax1 = plt.subplots(figsize=(4, 2.5))

        ax1.plot(K_range, inertias, "o-", markersize=5, linewidth=1.5)

        ax1.set_title("Elbow Curve", fontsize=10)
        ax1.set_xlabel("Number of Clusters (K)", fontsize=8)
        ax1.set_ylabel("Inertia", fontsize=8)
        ax1.tick_params(axis='both', labelsize=8)
        ax1.grid(alpha=0.3)

        plt.tight_layout()

        colA, colB = st.columns([1, 1])
        with colA:
            st.pyplot(fig1, use_container_width=False)

        st.info("👉 Choose the K where the curve bends (elbow point)")

        # KMEANS
        OPTIMAL_K = 3
        kmeans = KMeans(n_clusters=OPTIMAL_K)
        labels = kmeans.fit_predict(X_scaled)

        cluster_features["Cluster"] = labels

        # PCA
        st.subheader("📉 PCA Clusters")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig2, ax2 = plt.subplots(figsize=(4, 2.5))

        ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=30)

        ax2.set_title("PCA Cluster Projection", fontsize=10)
        ax2.set_xlabel("PC1", fontsize=8)
        ax2.set_ylabel("PC2", fontsize=8)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        colA, colB = st.columns([1, 1])
        with colA:
            st.pyplot(fig2, use_container_width=False)

        # TSNE
        st.subheader("📉 t-SNE Visualization")

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)

        fig3, ax3 = plt.subplots(figsize=(4, 2.5))

        ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=30, alpha=0.8)

        ax3.set_title("t-SNE Clustering", fontsize=10)
        ax3.set_xlabel("Component 1", fontsize=8)
        ax3.set_ylabel("Component 2", fontsize=8)
        ax3.tick_params(axis='both', labelsize=8)
        ax3.grid(alpha=0.3)

        plt.tight_layout()

        colA, colB = st.columns([1, 1])
        with colA:
            st.pyplot(fig3, use_container_width=False)

        # TSFRESH
        st.subheader("🧪 TSFresh Features")

        ts_hr = hr_minute.rename(columns={
            "Id": "id",
            "Time": "time",
            "HeartRate": "value"
        })

        features = extract_features(
            ts_hr,
            column_id="id",
            column_sort="time",
            column_value="value",
            default_fc_parameters=MinimalFCParameters()
        )

        features.dropna(axis=1, how="all", inplace=True)

        scaler = MinMaxScaler()
        features_norm = pd.DataFrame(
            scaler.fit_transform(features),
            columns=features.columns
        )

        fig4, ax4 = plt.subplots(figsize=(5, 3))

        sns.heatmap(
            features_norm,
            ax=ax4,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
            linewidths=0.3,
            cbar=True
        )

        ax4.set_title("TSFresh Feature Matrix (Normalized 0–1)", fontsize=9)
        ax4.tick_params(axis='x', labelsize=6, rotation=90)
        ax4.tick_params(axis='y', labelsize=6)

        plt.tight_layout()
        st.pyplot(fig4, use_container_width=False)

        # PROPHET
        st.subheader("❤️ Heart Rate Forecast")

        hr_daily_plot = (
            hr_minute.groupby("Date")["HeartRate"]
            .mean()
            .reset_index()
        )

        prophet_hr = hr_daily_plot.rename(columns={"Date": "ds", "HeartRate": "y"})
        prophet_hr["ds"] = pd.to_datetime(prophet_hr["ds"])

        model_hr = Prophet()
        model_hr.fit(prophet_hr)

        future_hr = model_hr.make_future_dataframe(periods=30)
        forecast_hr = model_hr.predict(future_hr)

        fig_hr, ax_hr = plt.subplots(figsize=(7, 3))

        ax_hr.scatter(prophet_hr["ds"], prophet_hr["y"], s=15, alpha=0.7)
        ax_hr.plot(forecast_hr["ds"], forecast_hr["yhat"], linewidth=2)

        ax_hr.fill_between(
            forecast_hr["ds"],
            forecast_hr["yhat_lower"],
            forecast_hr["yhat_upper"],
            alpha=0.2
        )

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_hr)
                # ---------------- CLUSTER PROFILE ----------------
        st.subheader("📊 Cluster Profiles")

        profile = cluster_features.groupby("Cluster")[cluster_cols].mean().round(2)
        st.dataframe(profile)

        fig_profile, ax_profile = plt.subplots(figsize=(9, 4))

        profile.plot(kind="bar", ax=ax_profile, width=0.7)

        plt.tight_layout()
        st.pyplot(fig_profile)
        
                
                # ---------------- PROPHET COMPONENTS ----------------
        st.subheader("📊 Prophet Components")

        # Generate components plot
        fig_comp = model_hr.plot_components(forecast_hr)

        # Resize to compact
        fig_comp.set_size_inches(4,2.5)

        # Add title
        fig_comp.suptitle(
            "Heart Rate Components",
            fontsize=7,
            y=1.02
        )

        plt.tight_layout()

        # DISPLAY
        col1, col2 = st.columns([1, 1])
        with col1:
            st.pyplot(fig_comp, use_container_width=False)


        # ---------------------------------------------------
        # t-SNE (KMeans vs DBSCAN)
        # ---------------------------------------------------
        st.subheader("📉 t-SNE Cluster Comparison")

        st.info("⏳ Running t-SNE... (may take ~20–30 seconds)")

        # Run t-SNE
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, len(X_scaled) - 1),
            max_iter=1000
        )

        X_tsne = tsne.fit_transform(X_scaled)

        # Figure
        fig, axes = plt.subplots(1, 2, figsize=(9,4))

        # ---------------- KMeans ----------------
        OPTIMAL_K = 3

        palette = [
            "#4FD1C5",
            "#63B3ED",
            "#F6AD55",
            "#FC8181",
            "#B794F4",
            "#68D391"
        ]

        kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)

        for cluster_id in sorted(set(kmeans_labels)):
            mask = kmeans_labels == cluster_id

            axes[0].scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                c=palette[cluster_id % len(palette)],
                label=f"C{cluster_id}",
                s=40,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5
            )

        axes[0].set_title(f"KMeans (K={OPTIMAL_K})", fontsize=10)
        axes[0].set_xlabel("Dim 1", fontsize=8)
        axes[0].set_ylabel("Dim 2", fontsize=8)
        axes[0].tick_params(axis='both', labelsize=7)
        axes[0].grid(alpha=0.3)

        # ---------------- DBSCAN ----------------
        from sklearn.cluster import DBSCAN

        EPS = 0.5

        dbscan = DBSCAN(eps=EPS, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)

        for label in sorted(set(dbscan_labels)):
            mask = dbscan_labels == label

            if label == -1:
                axes[1].scatter(
                    X_tsne[mask, 0],
                    X_tsne[mask, 1],
                    c="red",
                    marker="x",
                    s=50,
                    label="Noise",
                    alpha=0.9
                )
            else:
                axes[1].scatter(
                    X_tsne[mask, 0],
                    X_tsne[mask, 1],
                    c=palette[label % len(palette)],
                    label=f"C{label}",
                    s=40,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.5
                )

        axes[1].set_title(f"DBSCAN (eps={EPS})", fontsize=10)
        axes[1].set_xlabel("Dim 1", fontsize=8)
        axes[1].set_ylabel("Dim 2", fontsize=8)
        axes[1].tick_params(axis='both', labelsize=7)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        # DISPLAY
        col1, col2 = st.columns([1, 1])
        with col1:
            st.pyplot(fig, use_container_width=False)
        #------------------------------------------------------------
                # ---------------- CLUSTER INTERPRETATION ----------------
        st.subheader("🧠 Cluster Interpretation")

        for i in range(len(profile)):

            row = profile.loc[i]
            steps = row["TotalSteps"]

            st.markdown(f"### Cluster {i}")
            st.write(f"Steps: {steps:,.0f}")

            if steps > 10000:
                st.success("🏃 Highly Active")
            elif steps > 5000:
                st.info("🚶 Moderate")
            else:
                st.warning("🛋️ Sedentary")

        # ---------------- FINAL SUMMARY ----------------
        st.markdown("---")
        st.subheader("📊 Milestone 2 Summary — Real Fitbit Data")

        st.markdown("### ✅ Dataset Info")
        st.write(f"Users: {cluster_features.shape[0]}")
        st.write("Days: 31 (March–April 2016)")

        st.markdown("### 🧪 TSFresh Features")
        st.write(f"Extracted Features: {features.shape[1]}")
        st.write("Source: Real minute-level heart rate data")

        st.markdown("### 📈 Prophet Models")
        st.write("Heart Rate — 30 day forecast (80% CI)")
        st.write("Steps — 30 day forecast (80% CI)")
        st.write("Sleep — 30 day forecast (80% CI)")

        st.markdown("### 🤖 KMeans Clustering")
        st.write(f"Clusters Identified: {OPTIMAL_K}")

        cluster_dist = dict(cluster_features["Cluster"].value_counts().sort_index())
        st.write(f"Distribution: {cluster_dist}")

        # ---------------- DBSCAN INFO ----------------
        st.markdown("### ⚙️ DBSCAN Clustering")

        try:
            n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            n_noise = list(dbscan_labels).count(-1)

            st.write(f"Clusters: {n_clusters}")
            st.write(f"Noise Points: {n_noise}")
            st.write(f"Noise %: {n_noise/len(dbscan_labels)*100:.1f}%")

        except:
            st.warning("DBSCAN is applied in this run")

        st.success("✅ Milestone 2 Completed Successfully")


    else:
        st.warning("⬆️ Upload all files")



# =====================================================
# APP ROUTER
# =====================================================

# -------------------------------------------------------
# APP ROUTER
# -------------------------------------------------------

if milestone == "📊 Milestone 1 — Data Processing Pipeline":
    milestone1()

elif milestone == "🤖 Milestone 2 — ML Analytics Pipeline":
    milestone2()
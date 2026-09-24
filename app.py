import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from classes import CLASS_NAMES

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Traffic Vision AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# WHITE / LIGHT UI STYLE (upgraded)
# ============================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f6f8fc 0%, #ffffff 260px);
    color: #111827;
}

/* Header */
[data-testid="stHeader"] {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(6px);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: none;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12);
}

section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] p {
    color: #94a3b8 !important;
}

/* Hero title */
.hero-title {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    text-align: center;
    font-size: 50px;
    font-weight: 900;
    margin-top: 10px;
    margin-bottom: 6px;
    letter-spacing: -1px;
    line-height: 1;
}

.hero-title .title-text {
    background: linear-gradient(90deg, #2563eb, #7c3aed 60%, #db2777);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-title .traffic-emoji {
    font-size: 48px;
    line-height: 1;
}

/* Traffic-light EMOJI blink: cycles the emoji's own colors red -> yellow -> green */
.traffic-emoji {
    display: inline-block;
    animation: trafficEmojiBlink 3s infinite;
}

@keyframes trafficEmojiBlink {
    0%, 27%   { filter: hue-rotate(0deg)   saturate(3) brightness(1.1) drop-shadow(0 0 10px rgba(220, 38, 38, 0.7)); }
    33%, 60%  { filter: hue-rotate(140deg) saturate(3) brightness(1.2) drop-shadow(0 0 10px rgba(245, 158, 11, 0.7)); }
    66%, 94%  { filter: hue-rotate(260deg) saturate(3) brightness(1.1) drop-shadow(0 0 10px rgba(22, 163, 74, 0.7)); }
    100%      { filter: hue-rotate(0deg)   saturate(3) brightness(1.1) drop-shadow(0 0 10px rgba(220, 38, 38, 0.7)); }
}

/* Hero subtitle */
.hero-subtitle {
    text-align: center;
    color: #64748b;
    font-size: 17px;
    margin-bottom: 32px;
    font-weight: 500;
}

/* Section title */
.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 4px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #eef1f6;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.10);
}

/* Metric text */
div[data-testid="stMetric"] label {
    color: #64748b !important;
    font-weight: 600 !important;
}

div[data-testid="stMetric"] div {
    color: #0f172a;
    font-weight: 800;
}

/* Buttons */
.stButton > button {
    border-radius: 14px;
    min-height: 50px;
    font-weight: 700;
    letter-spacing: 0.3px;

    background: linear-gradient(90deg, #2563eb, #4f46e5);
    color: white;

    border: none;
    transition: all 0.25s ease;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #4338ca);
    color: white;
    box-shadow: 0 8px 22px rgba(37, 99, 235, 0.35);
    transform: translateY(-2px);
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background: #f8fafc;
    border: 1.5px dashed #93c5fd;
    border-radius: 18px;
    padding: 14px;
    transition: border-color 0.2s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #2563eb;
}

/* File uploader text */
div[data-testid="stFileUploader"] * {
    color: #334155;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #64748b;
    font-weight: 600;
    font-size: 15px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563eb;
    font-weight: 800;
}

div[data-baseweb="tab-highlight"] {
    background-color: #2563eb;
    height: 3px;
    border-radius: 3px;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 16px;
    border: 1px solid #eef1f6;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #eef1f6;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
}

/* Captions */
.stCaption {
    color: #64748b;
}

/* Divider spacing */
hr {
    margin: 22px 0 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    padding: 34px 0 12px 0;
    font-size: 13px;
    line-height: 1.8;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD SAME MODEL
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "Model/traffic_sign_cnn.keras"
    )

model = load_model()


def run_prediction(pil_image):
    """Shared prediction helper used by both single and batch flows."""
    processed = pil_image.resize((32, 32))
    arr = np.array(processed).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    pred_class = int(np.argmax(preds))
    conf = float(preds[pred_class] * 100)

    return processed, arr, preds, pred_class, conf


# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

if "nav_tab" not in st.session_state:
    st.session_state.nav_tab = "🚦 Recognition"

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(
    '<h1 style="display:flex; align-items:center; gap:10px; '
    'font-size:26px; font-weight:800; margin-bottom:0;">'
    '<span class="traffic-emoji" style="font-size:26px;">🚦</span>'
    '<span>Traffic Vision AI</span>'
    '</h1>',
    unsafe_allow_html=True
)

st.sidebar.caption(
    "Advanced CNN Traffic Sign Recognition"
)

st.sidebar.divider()

st.sidebar.subheader("🧠 Model")

model_info = [
    ("Architecture", "CNN", "A Convolutional Neural Network with Conv2D, BatchNorm, MaxPooling and Dense layers — designed to learn visual patterns such as edges, shapes, and colors."),
    ("Dataset", "GTSRB", "German Traffic Sign Recognition Benchmark — a dataset of 50,000+ real traffic sign images across 43 different German traffic sign categories."),
    ("Classes", "43", "The model can recognize 43 different types of traffic signs, including speed limits, stop, yield, no entry, and warning signs."),
    ("Input Size", "32 × 32", "Every image is resized to 32×32 pixels before prediction, matching the model's training input size."),
    ("Framework", "TensorFlow / Keras", "The model was built and trained using TensorFlow's Keras API, and is saved in the .keras format."),
]

for label, value, detail in model_info:
    with st.sidebar.expander(f"**{label}:** {value}"):
        st.caption(detail)
        if st.button("➡️ Go here", key=f"goto_model_{label}", use_container_width=True):
            st.session_state.nav_tab = "ℹ️ Project"
            st.rerun()

st.sidebar.divider()

st.sidebar.subheader("⚡ Features")

feature_info = [
    ("🖼️ Image Recognition", "Upload any traffic sign image and the model will instantly predict its class.", "🚦 Recognition"),
    ("📷 Camera Input", "Capture a photo directly with your camera — go to the Recognition tab and turn on the 'Enable Camera' switch.", "🚦 Recognition"),
    ("🎯 CNN Classification", "The deep learning CNN model analyzes the visual patterns in the image to decide the sign's class.", "🚦 Recognition"),
    ("📈 Confidence Analysis", "Every prediction comes with a confidence score (%) that shows how sure the model is about its answer.", "🚦 Recognition"),
    ("🏆 Top-5 Predictions", "See not just the best guess but the top 5 possible matches, each with its own confidence %.", "🚦 Recognition"),
    ("📊 Probability Analysis", "View the probability distribution across all 43 classes in a chart and table.", "📊 Analytics"),
    ("🕘 Prediction History", "All your past predictions are saved in the History tab for as long as the session is active.", "🕘 History"),
    ("📦 Batch Prediction", "Upload multiple images at once and get predictions for all of them in a grid and summary table.", "📦 Batch Predict"),
]

for title, detail, target_tab in feature_info:
    with st.sidebar.expander(title):
        st.caption(detail)
        if st.button("➡️ Go here", key=f"goto_{title}", use_container_width=True):
            st.session_state.nav_tab = target_tab
            st.rerun()

st.sidebar.divider()

st.sidebar.caption(
    "📷 Camera access is optional. "
    "Use the Enable Camera switch when you need it."
)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="hero-title">'
    '<span class="traffic-emoji">🚦</span>'
    '<span class="title-text">Traffic Vision AI</span>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Intelligent Traffic Sign Recognition using '
    'Convolutional Neural Networks'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# DASHBOARD
# ============================================================
m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "🧠 AI Architecture",
    "CNN"
)

m2.metric(
    "🚦 Sign Classes",
    "43"
)

m3.metric(
    "🖼️ Input Size",
    "32 × 32"
)

m4.metric(
    "🔍 Predictions",
    len(st.session_state.history)
)

st.write("")

# ============================================================
# NAVIGATION (session-state driven so sidebar buttons can jump here)
# ============================================================
nav_options = [
    "🚦 Recognition",
    "📦 Batch Predict",
    "📊 Analytics",
    "🕘 History",
    "ℹ️ Project"
]

st.markdown(
    """
    <style>
    div[role="radiogroup"] {
        gap: 6px;
    }
    div[role="radiogroup"] label {
        background: #f8fafc;
        border: 1px solid #eef1f6;
        padding: 8px 18px;
        border-radius: 999px;
        margin-right: 4px;
    }
    div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        border-color: transparent;
    }
    div[role="radiogroup"] label:has(input:checked) p {
        color: white !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

active_tab = st.radio(
    "Navigation",
    nav_options,
    horizontal=True,
    label_visibility="collapsed",
    key="nav_tab"
)

# ============================================================
# RECOGNITION TAB
# ============================================================
if active_tab == "🚦 Recognition":

    st.markdown(
        '<div class="section-title">'
        '📤 Traffic Sign Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Upload a traffic sign image or capture one "
        "using your camera."
    )

    upload_col, camera_col = st.columns(2)

    # ========================================================
    # UPLOAD IMAGE
    # ========================================================
    with upload_col:

        uploaded_file = st.file_uploader(
            "📁 Upload Image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

    # ========================================================
    # CAMERA
    # ========================================================
    with camera_col:

        camera_enabled = st.toggle(
            "📷 Enable Camera",
            value=False,
            help=(
                "Turn this ON only when you want "
                "to use the camera."
            )
        )

        if camera_enabled:

            st.caption(
                "Camera is ON. Capture an image below."
            )

            camera_file = st.camera_input(
                "📷 Camera Capture",
                key="traffic_camera"
            )

        else:

            camera_file = None

            st.info(
                "📷 Camera is OFF"
            )

    # ========================================================
    # SELECT IMAGE SOURCE
    # ========================================================
    source = (
        uploaded_file
        if uploaded_file is not None
        else camera_file
    )

    if source is not None:

        image = Image.open(
            source
        ).convert("RGB")

        processed_image = image.resize(
            (32, 32)
        )

        st.divider()

        image_col, process_col = st.columns(2)

        # ====================================================
        # ORIGINAL IMAGE
        # ====================================================
        with image_col:

            st.subheader(
                "📷 Original Image"
            )

            st.image(
                image,
                use_container_width=True
            )

        # ====================================================
        # PREPROCESSED IMAGE
        # ====================================================
        with process_col:

            st.subheader(
                "⚙️ CNN Preprocessed Image"
            )

            st.image(
                processed_image,
                width=220
            )

            st.caption(
                "Resized from original image "
                "to 32 × 32 pixels."
            )

        st.divider()

        # ====================================================
        # PROCESSING PIPELINE
        # ====================================================
        st.info(
            "🧠 Processing Pipeline: "
            "Image → Resize → Normalize → CNN → Classification"
        )

        # ====================================================
        # ANALYZE BUTTON
        # ====================================================
        if st.button(
            "🚀 ANALYZE TRAFFIC SIGN",
            use_container_width=True
        ):

            with st.spinner(
                "🧠 CNN model is analyzing the image..."
            ):

                img_array = np.array(
                    processed_image
                )

                img_array = img_array.astype(
                    "float32"
                ) / 255.0

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                predictions = model.predict(
                    img_array,
                    verbose=0
                )[0]

                predicted_class = int(
                    np.argmax(predictions)
                )

                confidence = float(
                    predictions[predicted_class] * 100
                )

            # =================================================
            # SAVE RESULT
            # =================================================
            st.session_state.result = {
                "class_id":
                    predicted_class,

                "name":
                    CLASS_NAMES[predicted_class],

                "confidence":
                    confidence,

                "predictions":
                    predictions
            }

            # =================================================
            # SAVE HISTORY
            # =================================================
            st.session_state.history.append({

                "Traffic Sign":
                    CLASS_NAMES[predicted_class],

                "Class ID":
                    predicted_class,

                "Confidence":
                    f"{confidence:.2f}%"
            })

        # ====================================================
        # RESULT
        # ====================================================
        if st.session_state.result is not None:

            result = st.session_state.result

            st.divider()

            st.subheader(
                "🎯 AI Recognition Result"
            )

            result_col, confidence_col = st.columns(2)

            # =================================================
            # PREDICTED SIGN
            # =================================================
            with result_col:

                st.markdown(
                    f'<div data-testid="stAlert" style="background:#dcfce7; '
                    f'border:1px solid #86efac; border-radius:16px; padding:14px 18px; '
                    f'color:#166534; font-weight:600; margin-bottom:8px;">'
                    f'<span class="traffic-emoji">🚦</span> {result["name"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.metric(
                    "Class ID",
                    result["class_id"]
                )

            # =================================================
            # CONFIDENCE
            # =================================================
            with confidence_col:

                st.metric(
                    "AI Confidence",
                    f"{result['confidence']:.2f}%"
                )

                st.progress(
                    result["confidence"] / 100
                )

                if result["confidence"] >= 80:

                    st.success(
                        "✅ High confidence prediction"
                    )

                elif result["confidence"] >= 50:

                    st.warning(
                        "⚠️ Medium confidence prediction"
                    )

                else:

                    st.error(
                        "❌ Low confidence prediction"
                    )

            # =================================================
            # TOP 5 PREDICTIONS
            # =================================================
            st.divider()

            st.subheader(
                "🏆 Top 5 Predictions"
            )

            top_5 = np.argsort(
                result["predictions"]
            )[-5:][::-1]

            top_data = []

            for rank, class_id in enumerate(
                top_5,
                1
            ):

                top_data.append({

                    "Rank":
                        f"#{rank}",

                    "Class ID":
                        int(class_id),

                    "Traffic Sign":
                        CLASS_NAMES[class_id],

                    "Confidence":
                        (
                            f"{result['predictions'][class_id] * 100:.2f}%"
                        )
                })

            st.dataframe(
                pd.DataFrame(top_data),
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # PROBABILITY DISTRIBUTION
            # =================================================
            st.divider()

            st.subheader(
                "📊 Probability Distribution"
            )

            probability_df = pd.DataFrame({

                "Traffic Sign":
                    CLASS_NAMES,

                "Probability":
                    result["predictions"] * 100
            })

            probability_df = probability_df.sort_values(
                "Probability",
                ascending=False
            ).head(10)

            st.bar_chart(
                probability_df.set_index(
                    "Traffic Sign"
                )
            )

# ============================================================
# BATCH PREDICT TAB (Advanced Feature)
# ============================================================
if active_tab == "📦 Batch Predict":

    st.markdown(
        '<div class="section-title">📦 Batch Prediction</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Upload multiple traffic sign images at once and get "
        "predictions for all of them together."
    )

    batch_files = st.file_uploader(
        "📁 Upload Multiple Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if batch_files:

        st.info(f"📸 {len(batch_files)} image(s) selected.")

        if st.button("🚀 ANALYZE ALL IMAGES", use_container_width=True, key="batch_analyze_btn"):

            batch_results = []

            progress_bar = st.progress(0)

            for i, file in enumerate(batch_files):

                try:
                    img = Image.open(file).convert("RGB")
                    _, _, preds, pred_class, conf = run_prediction(img)

                    batch_results.append({
                        "thumbnail": img,
                        "File Name": file.name,
                        "Traffic Sign": CLASS_NAMES[pred_class],
                        "Class ID": pred_class,
                        "Confidence": conf
                    })

                except Exception as e:
                    batch_results.append({
                        "thumbnail": None,
                        "File Name": file.name,
                        "Traffic Sign": f"Error: {e}",
                        "Class ID": "-",
                        "Confidence": 0.0
                    })

                progress_bar.progress((i + 1) / len(batch_files))

            st.session_state.batch_results = batch_results
            st.session_state.history.extend([
                {
                    "Traffic Sign": r["Traffic Sign"],
                    "Class ID": r["Class ID"],
                    "Confidence": f"{r['Confidence']:.2f}%"
                }
                for r in batch_results if r["thumbnail"] is not None
            ])

        if "batch_results" in st.session_state and st.session_state.batch_results:

            st.divider()
            st.subheader("🎯 Batch Results")

            results = st.session_state.batch_results

            avg_conf = np.mean([r["Confidence"] for r in results if r["thumbnail"] is not None]) if results else 0
            c1, c2 = st.columns(2)
            c1.metric("Images Processed", len(results))
            c2.metric("Average Confidence", f"{avg_conf:.2f}%")

            st.divider()

            cols_per_row = 4
            for row_start in range(0, len(results), cols_per_row):
                row_items = results[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)

                for col, item in zip(cols, row_items):
                    with col:
                        if item["thumbnail"] is not None:
                            st.image(item["thumbnail"], use_container_width=True)
                            st.markdown(f"**{item['Traffic Sign']}**")
                            st.caption(f"Confidence: {item['Confidence']:.2f}%")
                        else:
                            st.error(item["Traffic Sign"])
                        st.caption(item["File Name"])

            st.divider()
            st.subheader("📋 Summary Table")

            summary_df = pd.DataFrame([
                {
                    "File Name": r["File Name"],
                    "Traffic Sign": r["Traffic Sign"],
                    "Class ID": r["Class ID"],
                    "Confidence": f"{r['Confidence']:.2f}%"
                }
                for r in results
            ])

            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    else:
        st.markdown(
            '<div class="card" style="text-align:center; padding: 30px; '
            'background:#f8fafc; border-radius:16px; border:1px solid #eef1f6;">'
            '<h4>👆 Upload 2 or more images to run batch prediction</h4>'
            '</div>',
            unsafe_allow_html=True
        )

# ============================================================
# ANALYTICS TAB
# ============================================================
if active_tab == "📊 Analytics":

    st.subheader(
        "📊 AI Analytics Dashboard"
    )

    if st.session_state.result is None:

        st.info(
            "Upload and analyze a traffic sign first "
            "to view detailed analytics."
        )

    else:

        result = st.session_state.result

        a, b, c = st.columns(3)

        a.metric(
            "Predicted Class",
            result["class_id"]
        )

        b.metric(
            "Confidence",
            f"{result['confidence']:.2f}%"
        )

        c.metric(
            "Total Classes",
            "43"
        )

        st.divider()

        analytics_df = pd.DataFrame({

            "Class ID":
                range(43),

            "Traffic Sign":
                CLASS_NAMES,

            "Probability":
                result["predictions"] * 100
        })

        analytics_df = analytics_df.sort_values(
            "Probability",
            ascending=False
        )

        st.subheader(
            "All 43 Class Probabilities"
        )

        st.dataframe(
            analytics_df,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# HISTORY TAB
# ============================================================
if active_tab == "🕘 History":

    st.subheader(
        "🕘 Prediction History"
    )

    if len(st.session_state.history) == 0:

        st.info(
            "No prediction history available yet."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button(
            "🗑️ Clear Prediction History"
        ):

            st.session_state.history = []

            st.session_state.result = None

            st.rerun()

# ============================================================
# PROJECT TAB
# ============================================================
if active_tab == "ℹ️ Project":

    st.subheader(
        "ℹ️ Project Information"
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.info(
            "**🧠 Deep Learning**\n\n"
            "Convolutional Neural Network based "
            "traffic sign classification."
        )

    with p2:

        st.info(
            "**📦 Dataset**\n\n"
            "GTSRB dataset with 43 traffic sign categories."
        )

    with p3:

        st.info(
            "**⚡ Technologies**\n\n"
            "Python, TensorFlow, Keras, NumPy, "
            "Pandas and Streamlit."
        )

    st.divider()

    st.subheader(
        "🔬 AI Processing Pipeline"
    )

    p1, p2, p3, p4 = st.columns(4)

    p1.info(
        "🖼️ **Input Image**"
    )

    p2.info(
        "⚙️ **Preprocessing**"
    )

    p3.info(
        "🧠 **CNN Model**"
    )

    p4.info(
        "🎯 **Prediction**"
    )

# ============================================================
# FOOTER
# ============================================================
st.divider()

st.markdown(
    '<div class="footer">'
    '<span class="traffic-emoji">🚦</span> Traffic Vision AI | CNN | TensorFlow | Keras | GTSRB'
    '<br>'
    'AI Traffic Sign Recognition Project'
    '</div>',
    unsafe_allow_html=True
)
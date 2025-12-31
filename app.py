import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import pandas as pd
from datetime import datetime

# Optional: webcam live video (pip install streamlit-webrtc av opencv-python)
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    _HAS_WEBRTC = True
except Exception:
    _HAS_WEBRTC = False

# -------------------------------
# ✅ PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="BRUCEBOT-Underwater Trash Detection", layout="wide")

# -------------------------------
# ✅ PATHS / STORAGE
# -------------------------------
COUNTER_FILE = "visitor_count.txt"
DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
EXCEL_FILE = os.path.join(DATA_DIR, "submissions.xlsx")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------
# ✅ VISITOR COUNTER (increment per new session/visitor)
# -------------------------------
if "visitor_id" not in st.session_state:
    if not os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "w") as f:
            f.write("0")
    with open(COUNTER_FILE, "r") as f:
        count = int((f.read() or "0").strip())

    count += 1

    with open(COUNTER_FILE, "w") as f:
        f.write(str(count))

    st.session_state["visitor_id"] = count

visitor_id = st.session_state["visitor_id"]

# -------------------------------
# ✅ EXCEL INIT / HELPERS
# -------------------------------
def init_excel():
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame(
            columns=[
                "VisitorID",
                "Timestamp",
                "Name",
                "Country",
                "MediaType",               # image / video / realtime_snapshot / realtime_video
                "ImageFilename",           # only set for images/snapshots
                "ConsentToUseForTraining", # Yes / No
                "Feedback",                # Yay / Nay
            ]
        )
        with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Submissions")

def read_submissions():
    init_excel()
    return pd.read_excel(EXCEL_FILE, sheet_name="Submissions")

def write_submissions(df: pd.DataFrame):
    os.makedirs(os.path.dirname(EXCEL_FILE), exist_ok=True)
    with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, index=False, sheet_name="Submissions")

def upsert_row(visitor_id: int, updates: dict):
    df = read_submissions()

    if (df["VisitorID"] == visitor_id).any():
        idx = df.index[df["VisitorID"] == visitor_id][0]
        for k, v in updates.items():
            df.at[idx, k] = v
    else:
        base = {
            "VisitorID": visitor_id,
            "Timestamp": datetime.now().isoformat(timespec="seconds"),
            "Name": "",
            "Country": "",
            "MediaType": "",
            "ImageFilename": "",
            "ConsentToUseForTraining": "",
            "Feedback": "",
        }
        base.update(updates)
        df.loc[len(df)] = base

    write_submissions(df)

# -------------------------------
# ✅ CSS STYLING
# -------------------------------
st.markdown(
    """
<style>
body { background: linear-gradient(to right, #1fa2ff, #12d8fa, #a6ffcb); }
.title-box {
    background: linear-gradient(135deg, #ff512f, #dd2476);
    padding: 25px; border-radius: 15px; color: white; text-align: center;
    font-size: 32px; font-weight: bold;
}
.visitor-box {
    background: #000; color: #00ffcc; padding: 10px; border-radius: 10px;
    font-size: 20px; text-align: center; margin-top: 10px;
}
.image-box { border: 5px solid #ff9800; padding: 10px; border-radius: 15px; background-color: white; }
.subtitle { font-size: 20px; font-weight: bold; color: #0047ab; text-align: center; }
.small-note { font-size: 14px; color: #0b3557; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------
# ✅ HEADER
# -------------------------------
st.markdown('<div class="title-box">Ashwita Ramanavelan Science Fair Project</div>', unsafe_allow_html=True)
st.markdown(f'<div class="visitor-box">👥 Total Visitors: {visitor_id}</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter your details and upload an underwater image/video, or use real-time camera</p>', unsafe_allow_html=True)

# -------------------------------
# ✅ SESSION STATE
# -------------------------------
st.session_state.setdefault("name", "")
st.session_state.setdefault("country", "")
st.session_state.setdefault("consent", "")               # Yes / No
st.session_state.setdefault("image_saved", False)        # only for uploaded image
st.session_state.setdefault("saved_filename", "")
st.session_state.setdefault("feedback_saved", False)
st.session_state.setdefault("session_closed", False)     # Close & thank you
st.session_state.setdefault("last_mode", "Upload Image") # preserve mode on rerun

# -------------------------------
# ✅ CLOSE SCREEN
# -------------------------------
if st.session_state["session_closed"]:
    st.markdown(
        """
        <div style="
            background:#0b3557;
            color:white;
            padding:25px;
            border-radius:15px;
            text-align:center;
            font-size:22px;">
            🙏 Thank you for participating!<br><br>
            Your contribution helps improve underwater trash detection 🌊🤖
            <br><br>
            — Project by <strong>Ashwita Ramanavelan</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# -------------------------------
# ✅ USER FORM (Name + Country + Consent)
# -------------------------------
with st.form("user_form"):
    name = st.text_input("👤 Enter Your Name", value=st.session_state["name"])
    country = st.text_input("🌎 Enter Your Country", value=st.session_state["country"])

    st.markdown(
        '<div class="small-note">📌 Permission: Can we use your uploaded picture to improve this model in the future (training)?</div>',
        unsafe_allow_html=True,
    )
    consent = st.radio(
        "Consent",
        options=["Yes", "No"],
        index=0 if st.session_state["consent"] in ["", "Yes"] else 1,
        horizontal=True,
        label_visibility="collapsed",
    )

    submitted = st.form_submit_button("Continue")

if submitted:
    if name and country and consent in ["Yes", "No"]:
        st.session_state["name"] = name.strip()
        st.session_state["country"] = country.strip()
        st.session_state["consent"] = consent

        upsert_row(
            visitor_id,
            {
                "Timestamp": datetime.now().isoformat(timespec="seconds"),
                "Name": st.session_state["name"],
                "Country": st.session_state["country"],
                "ConsentToUseForTraining": st.session_state["consent"],
            },
        )

        st.success(f"Welcome {st.session_state['name']} from {st.session_state['country']} ✅")
    else:
        st.warning("Please enter Name, Country, and choose Consent (Yes/No).")

# -------------------------------
# ✅ LOAD YOLO MODEL
# ---------------train13----------------
MODEL_PATH = ./train13/weights/best.pt"
model = YOLO(MODEL_PATH)

# -------------------------------
# ✅ CHOOSE INPUT MODE
# -------------------------------
st.markdown("## 🎥 Choose Input Type")
modes = ["Upload Image", "Upload Video", "Real-time Camera (image snapshots)", "Real-time Video (live detection)"]
default_index = modes.index(st.session_state["last_mode"]) if st.session_state["last_mode"] in modes else 0

mode = st.radio("Input", modes, index=default_index)
st.session_state["last_mode"] = mode

# -------------------------------
# ✅ MAIN MEDIA HANDLING
# -------------------------------
detection_shown = False  # used to decide when to show retry/close + feedback

# ---- Upload Image (SAVED)
if mode == "Upload Image":
    uploaded_img = st.file_uploader("📸 Upload Underwater Image", type=["jpg", "jpeg", "png"])

    if uploaded_img and st.session_state["name"] and st.session_state["country"] and st.session_state["consent"]:
        # Save image once (per visitor session)
        if not st.session_state["image_saved"]:
            ext = os.path.splitext(uploaded_img.name)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png"]:
                ext = ".jpg"

            safe_name = "".join([c for c in st.session_state["name"] if c.isalnum() or c in (" ", "_", "-")]).strip()
            safe_country = "".join([c for c in st.session_state["country"] if c.isalnum() or c in (" ", "_", "-")]).strip()

            filename = f"{visitor_id:06d}_{safe_name}_{safe_country}{ext}".replace(" ", "_")
            save_path = os.path.join(UPLOAD_DIR, filename)

            with open(save_path, "wb") as f:
                f.write(uploaded_img.getvalue())

            st.session_state["image_saved"] = True
            st.session_state["saved_filename"] = filename

            upsert_row(
                visitor_id,
                {
                    "Timestamp": datetime.now().isoformat(timespec="seconds"),
                    "MediaType": "image",
                    "ImageFilename": filename,
                },
            )

        img = Image.open(uploaded_img)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            temp.write(uploaded_img.getvalue())
            temp_path = temp.name

        results = model(temp_path)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="image-box">', unsafe_allow_html=True)
            st.image(img, caption="🌊 Original Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            for r in results:
                output = r.plot()
                st.markdown('<div class="image-box">', unsafe_allow_html=True)
                st.image(output, caption="🤖 AI Detection Result", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        detection_shown = True

    elif uploaded_img:
        st.warning("⚠️ Please enter Name, Country, and Consent first!")

# ---- Upload Video (NOT SAVED)
elif mode == "Upload Video":
    uploaded_vid = st.file_uploader("📼 Upload Underwater Video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_vid and st.session_state["name"] and st.session_state["country"] and st.session_state["consent"]:
        upsert_row(
            visitor_id,
            {
                "Timestamp": datetime.now().isoformat(timespec="seconds"),
                "MediaType": "video",
                "ImageFilename": "",
            },
        )

        st.video(uploaded_vid)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tempv:
            tempv.write(uploaded_vid.getvalue())
            video_path = tempv.name

        st.info("Running detection on video (showing ~10 sampled frames for speed).")

        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            sample_every = max(1, frame_count // 10) if frame_count > 0 else 30

            shown = 0
            while cap.isOpened() and shown < 10:
                idx = shown * sample_every
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = model(frame_rgb)
                plotted = res[0].plot()

                st.image(plotted, caption=f"🤖 Detection sample frame #{shown + 1}", use_container_width=True)
                shown += 1

            cap.release()
            detection_shown = True
        except Exception as e:
            st.warning(f"Video sampling requires OpenCV. Install: pip install opencv-python\n\nError: {e}")

    elif uploaded_vid:
        st.warning("⚠️ Please enter Name, Country, and Consent first!")

# ---- Real-time Camera snapshots (SAVED ONLY WHEN SNAPSHOT TAKEN)
elif mode == "Real-time Camera (image snapshots)":
    if not (st.session_state["name"] and st.session_state["country"] and st.session_state["consent"]):
        st.warning("⚠️ Please enter Name, Country, and Consent first!")
    else:
        upsert_row(
            visitor_id,
            {
                "Timestamp": datetime.now().isoformat(timespec="seconds"),
                "MediaType": "realtime_snapshot",
                "ImageFilename": "",
            },
        )

        st.info("Use the camera below and take a snapshot. Only snapshots (images) are saved; live feed is not saved.")
        cam_img = st.camera_input("📷 Take a snapshot")

        if cam_img:
            # Save snapshot
            ext = ".jpg"
            safe_name = "".join([c for c in st.session_state["name"] if c.isalnum() or c in (" ", "_", "-")]).strip()
            safe_country = "".join([c for c in st.session_state["country"] if c.isalnum() or c in (" ", "_", "-")]).strip()

            filename = f"{visitor_id:06d}_{safe_name}_{safe_country}_snapshot{ext}".replace(" ", "_")
            save_path = os.path.join(UPLOAD_DIR, filename)

            with open(save_path, "wb") as f:
                f.write(cam_img.getvalue())

            upsert_row(
                visitor_id,
                {
                    "Timestamp": datetime.now().isoformat(timespec="seconds"),
                    "MediaType": "realtime_snapshot",
                    "ImageFilename": filename,
                },
            )

            # Run detection
            img = Image.open(cam_img)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                temp.write(cam_img.getvalue())
                temp_path = temp.name

            results = model(temp_path)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="image-box">', unsafe_allow_html=True)
                st.image(img, caption="📷 Snapshot", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                for r in results:
                    output = r.plot()
                    st.markdown('<div class="image-box">', unsafe_allow_html=True)
                    st.image(output, caption="🤖 AI Detection Result", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            detection_shown = True

# ---- Real-time Video (LIVE, NOT SAVED)
elif mode == "Real-time Video (live detection)":
    if not (st.session_state["name"] and st.session_state["country"] and st.session_state["consent"]):
        st.warning("⚠️ Please enter Name, Country, and Consent first!")
    else:
        upsert_row(
            visitor_id,
            {
                "Timestamp": datetime.now().isoformat(timespec="seconds"),
                "MediaType": "realtime_video",
                "ImageFilename": "",
            },
        )

        if not _HAS_WEBRTC:
            st.error(
                "Real-time live video requires streamlit-webrtc.\n\n"
                "Install:\n"
                "pip install streamlit-webrtc av opencv-python\n\n"
                "Then rerun the app."
            )
        else:
            st.info("Live camera feed with detection. Nothing is saved.")

            class YOLOVideoTransformer(VideoTransformerBase):
                def __init__(self):
                    self.model = model

                def transform(self, frame):
                    img_bgr = frame.to_ndarray(format="bgr24")
                    img_rgb = img_bgr[:, :, ::-1]
                    res = self.model(img_rgb)
                    out = res[0].plot()
                    return out

            webrtc_streamer(
                key="yolo-live",
                video_transformer_factory=YOLOVideoTransformer,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            detection_shown = True

# -------------------------------
# ✅ FEEDBACK + RETRY/CLOSE (only after detection shown)
# -------------------------------
if detection_shown:
    st.markdown("### ✅ Was this detection accurate?")

    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("👍 Yay", disabled=st.session_state["feedback_saved"]):
            upsert_row(visitor_id, {"Feedback": "Yay", "Timestamp": datetime.now().isoformat(timespec="seconds")})
            st.session_state["feedback_saved"] = True
            st.success("🎉 Thank you for the positive feedback!")

    with col_no:
        if st.button("👎 Nay", disabled=st.session_state["feedback_saved"]):
            upsert_row(visitor_id, {"Feedback": "Nay", "Timestamp": datetime.now().isoformat(timespec="seconds")})
            st.session_state["feedback_saved"] = True
            st.warning("💡 Thanks! We’ll use this to improve.")

    st.markdown("## 🔁 Next Steps")
    col_retry, col_close = st.columns(2)

    with col_retry:
        if st.button("🔄 Retry Another Image / Video"):
            # reset ONLY media + feedback state (keep name/country/consent and same visitor_id)
            st.session_state["image_saved"] = False
            st.session_state["saved_filename"] = ""
            st.session_state["feedback_saved"] = False
            # also clear uploader/camera widget state by rerun
            st.rerun()

    with col_close:
        if st.button("✅ Close & Finish"):
            st.session_state["session_closed"] = True
            st.rerun()

# -------------------------------
# ✅ GLOBAL USER DISTRIBUTION + FEEDBACK SUMMARY (from Excel)
# -------------------------------
st.markdown("## 🌍 Global User Distribution")
df_all = read_submissions()
df_users = df_all.dropna(subset=["Country"])
df_users = df_users[df_users["Country"].astype(str).str.strip() != ""]

if not df_users.empty:
    st.bar_chart(df_users["Country"].value_counts())
else:
    st.info("No user data yet.")

st.markdown("## 📊 Feedback Summary")
df_fb = df_all.dropna(subset=["Feedback"])
df_fb = df_fb[df_fb["Feedback"].astype(str).str.strip() != ""]

if not df_fb.empty:
    st.bar_chart(df_fb["Feedback"].value_counts())
else:
    st.info("No feedback yet.")

# -------------------------------
# ✅ FOOTER
# -------------------------------
st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size:16px; color:#003366;">
        🌟 Thank you for supporting student-led AI research 🌟<br>
        Project by <strong>Ashwita Ramanavelan</strong>
    </div>
    """,
    unsafe_allow_html=True,
)

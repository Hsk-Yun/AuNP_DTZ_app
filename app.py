import os
import io
import hashlib
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from skimage.color import rgb2lab
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from streamlit_image_coordinates import streamlit_image_coordinates


# =========================================================
# Basic settings
# =========================================================
st.set_page_config(page_title="AuNP@DTZ Heavy Metal Ion Detection App", layout="wide")

DATA_PATH = "training_data.csv"
AUG_PATH = "training_data_augmented.csv"

FEATURES = ["L", "a", "b", "deltaL", "deltaa", "deltab", "deltaE"]

PPM_MIN = 0
PPM_MAX = 40

DISPLAY_WIDTH = 900
ROI_RADIUS_DISP = 14
NO_METAL_THRESHOLD = 8.0
KEEP_PERCENT = 65
CONFIDENCE_WARNING_THRESHOLD = 0.50

GROUP_K = 5
METAL_K = 3
PPM_K = 3

GROUPED_LABELS = ["Ag-Zn", "Cd-Mn"]

st.title("AuNP@DTZ Heavy Metal Ion Detection App")
st.markdown("""
<style>
.workflow {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin: 20px 0 35px 0;
    padding: 16px;
    background-color: #f8fafc;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

.step {
    flex: 1;
    text-align: center;
    padding: 12px 8px;
    background-color: white;
    border-radius: 10px;
    border: 1px solid #d1d5db;
    font-size: 14px;
    font-weight: 600;
}

.arrow {
    font-size: 22px;
    font-weight: bold;
    color: #9ca3af;
}
</style>

<div class="workflow">
    <div class="step">1. Image Upload</div>
    <div class="arrow">→</div>
    <div class="step">2. ROI Selection</div>
    <div class="arrow">→</div>
    <div class="step">3. RGB Extraction</div>
    <div class="arrow">→</div>
    <div class="step">4. CIE Lab Conversion</div>
    <div class="arrow">→</div>
    <div class="step">5. ML Prediction</div>
    <div class="arrow">→</div>
    <div class="step">6. Result Output</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Image loading
# =========================================================
def load_uploaded_image(uploaded_file):
    data = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    # RAW file handling
    if filename.endswith(".raw"):
        try:
            import rawpy
        except ImportError:
            st.error("To process RAW files, add rawpy to requirements.txt.")
            st.stop()

        suffix = os.path.splitext(filename)[1] if os.path.splitext(filename)[1] else ".raw"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            temp_path = tmp.name

        try:
            with rawpy.imread(temp_path) as raw:
                rgb = raw.postprocess()
            image = Image.fromarray(rgb).convert("RGB")
        except Exception:
            st.error("Failed to read the RAW file. This RAW format may not be supported.")
            st.stop()
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

        return image, data

    # Standard image handling
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return image, data
    except Exception:
        st.error("Failed to open the image file.")
        st.stop()


# =========================================================
# Training data loading
# =========================================================
@st.cache_data
def load_training_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}

    if "dE" in df.columns and "deltaE" not in df.columns:
        rename_map["dE"] = "deltaE"
    if "DeltaE" in df.columns and "deltaE" not in df.columns:
        rename_map["DeltaE"] = "deltaE"
    if "DeltaL" in df.columns and "deltaL" not in df.columns:
        rename_map["DeltaL"] = "deltaL"
    if "Deltaa" in df.columns and "deltaa" not in df.columns:
        rename_map["Deltaa"] = "deltaa"
    if "Deltab" in df.columns and "deltab" not in df.columns:
        rename_map["Deltab"] = "deltab"
    if "heavy metal" in df.columns and "Metal" not in df.columns:
        rename_map["heavy metal"] = "Metal"

    df = df.rename(columns=rename_map)

    required_cols = ["Metal", "Group", "ppm"] + FEATURES
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        st.stop()

    for c in FEATURES + ["ppm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Metal"] = df["Metal"].astype(str).str.strip()
    df["Group"] = df["Group"].astype(str).str.strip()

    df = df[~df["Metal"].isin(["", "nan", "None"])]
    df = df[~df["Group"].isin(["", "nan", "None"])]

    df = df[df["ppm"].between(PPM_MIN, PPM_MAX)]
    df = df[df["ppm"] > 0].copy()
    df = df.dropna(subset=["Metal", "Group", "ppm"] + FEATURES)

    if os.path.exists(AUG_PATH):
        aug = pd.read_csv(AUG_PATH, encoding="utf-8-sig")
        aug.columns = [str(c).strip() for c in aug.columns]

        rename_aug = {}

        if "dE" in aug.columns and "deltaE" not in aug.columns:
            rename_aug["dE"] = "deltaE"
        if "DeltaE" in aug.columns and "deltaE" not in aug.columns:
            rename_aug["DeltaE"] = "deltaE"
        if "DeltaL" in aug.columns and "deltaL" not in aug.columns:
            rename_aug["DeltaL"] = "deltaL"
        if "Deltaa" in aug.columns and "deltaa" not in aug.columns:
            rename_aug["Deltaa"] = "deltaa"
        if "Deltab" in aug.columns and "deltab" not in aug.columns:
            rename_aug["Deltab"] = "deltab"
        if "heavy metal" in aug.columns and "Metal" not in aug.columns:
            rename_aug["heavy metal"] = "Metal"

        aug = aug.rename(columns=rename_aug)

        if all(c in aug.columns for c in ["Metal", "Group", "ppm"] + FEATURES):
            for c in FEATURES + ["ppm"]:
                aug[c] = pd.to_numeric(aug[c], errors="coerce")

            aug["Metal"] = aug["Metal"].astype(str).str.strip()
            aug["Group"] = aug["Group"].astype(str).str.strip()

            aug = aug[~aug["Metal"].isin(["", "nan", "None"])]
            aug = aug[~aug["Group"].isin(["", "nan", "None"])]

            aug = aug[aug["ppm"].between(PPM_MIN, PPM_MAX)]
            aug = aug[aug["ppm"] > 0].copy()
            aug = aug.dropna(subset=["Metal", "Group", "ppm"] + FEATURES)

            df = pd.concat([df, aug], ignore_index=True)

    return df


# =========================================================
# Model building
# =========================================================
@st.cache_resource
def build_group_model(df):
    X = df[FEATURES]
    y = df["Group"]

    k = min(GROUP_K, len(df))

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=k,
            metric="euclidean",
            weights="distance"
        )
    )
    model.fit(X, y)
    return model


def build_metal_model(group_df, n_neighbors=METAL_K):
    k = min(n_neighbors, len(group_df))

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=k,
            metric="euclidean",
            weights="distance"
        )
    )

    X = group_df[FEATURES]
    y = group_df["Metal"]
    model.fit(X, y)
    return model


def build_ppm_model(metal_df, n_neighbors=PPM_K):
    k = min(n_neighbors, len(metal_df))

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=k,
            metric="euclidean",
            weights="distance"
        )
    )

    X = metal_df[FEATURES]
    y = metal_df["ppm"].astype(int)
    model.fit(X, y)
    return model


# =========================================================
# Color calculation functions
# =========================================================
def rgb_to_lab_value(rgb):
    arr = np.array(rgb, dtype=np.float32).reshape(1, 1, 3) / 255.0
    lab = rgb2lab(arr)[0, 0]
    return lab


def delta_e(lab1, lab2):
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def robust_rgb_from_circle(image, x, y, radius, keep_percent=65):
    """
    Inside the circular ROI:
    1) Remove overly bright or dark pixels
    2) Remove pixels that deviate too far from the center color
    => Reduce reflection, shadow, and background effects
    """
    arr = np.array(image).astype(np.float32)
    h, w, _ = arr.shape

    x = int(round(x))
    y = int(round(y))
    radius = max(2, int(round(radius)))

    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    crop = arr[y1:y2, x1:x2, :]

    if crop.size == 0:
        return np.array([0, 0, 0], dtype=np.float32)

    yy, xx = np.mgrid[y1:y2, x1:x2]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    mask_circle = dist <= radius

    pixels = crop[mask_circle]
    if len(pixels) == 0:
        return np.array([0, 0, 0], dtype=np.float32)

    center_r = max(2, radius // 3)
    mask_center = dist <= center_r
    center_pixels = crop[mask_center]

    if len(center_pixels) == 0:
        ref_rgb = np.median(pixels, axis=0)
    else:
        ref_rgb = np.median(center_pixels, axis=0)

    brightness = pixels.mean(axis=1)
    low_b, high_b = np.percentile(brightness, [2, 98])
    brightness_mask = (brightness >= low_b) & (brightness <= high_b)

    color_dist = np.linalg.norm(pixels - ref_rgb, axis=1)
    valid_d = color_dist[brightness_mask] if brightness_mask.sum() >= 10 else color_dist

    if len(valid_d) >= 10:
        threshold = np.percentile(valid_d, keep_percent)
        final_mask = brightness_mask & (color_dist <= threshold)

        if final_mask.sum() < 10:
            final_mask = brightness_mask
    else:
        final_mask = brightness_mask if brightness_mask.sum() > 0 else np.ones(len(pixels), dtype=bool)

    selected = pixels[final_mask]
    if len(selected) == 0:
        selected = pixels

    robust_rgb = np.median(selected, axis=0)
    return robust_rgb


# =========================================================
# Draw circles on the preview image
# =========================================================
def draw_circle_preview(display_image, blank_orig, sample_orig_list, scale, radius_disp):
    img = display_image.copy()
    draw = ImageDraw.Draw(img)

    def draw_one(orig_point, color, label):
        if orig_point is None:
            return

        x_disp = int(round(orig_point[0] * scale))
        y_disp = int(round(orig_point[1] * scale))
        r = int(radius_disp)

        draw.ellipse(
            (x_disp - r, y_disp - r, x_disp + r, y_disp + r),
            outline=color,
            width=4
        )
        draw.ellipse(
            (x_disp - 4, y_disp - 4, x_disp + 4, y_disp + 4),
            fill=color
        )
        draw.text((x_disp + r + 4, y_disp + 2), label, fill=color)

    draw_one(blank_orig, "blue", "Blank")

    for idx, sample_orig in enumerate(sample_orig_list, start=1):
        draw_one(sample_orig, "red", f"Sample {idx}")

    return img


# =========================================================
# Data preparation
# =========================================================
df = load_training_data()
group_model = build_group_model(df)
group_knn = group_model.named_steps["kneighborsclassifier"]
group_classes = group_knn.classes_


# =========================================================
# Session state initialization
# =========================================================
if "blank_point_orig" not in st.session_state:
    st.session_state.blank_point_orig = None

if "sample_points_orig" not in st.session_state:
    st.session_state.sample_points_orig = []

if "selection_mode" not in st.session_state:
    st.session_state.selection_mode = "blank"

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "image_hash" not in st.session_state:
    st.session_state.image_hash = None


# =========================================================
# 1. Image upload
# =========================================================
st.subheader("1. Upload Image")

input_method = st.radio(
    "",
    ["Upload Photo", "Take Photo"],
    horizontal=True,
    label_visibility="collapsed"
)

uploaded_file = None

if input_method == "Upload Photo":
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png", "raw"],
        label_visibility="collapsed"
    )
else:
    uploaded_file = st.camera_input("")

if uploaded_file is None:
    st.stop()

image, image_bytes = load_uploaded_image(uploaded_file)

current_hash = hashlib.md5(image_bytes).hexdigest()

if st.session_state.image_hash != current_hash:
    st.session_state.image_hash = current_hash
    st.session_state.blank_point_orig = None
    st.session_state.sample_points_orig = []
    st.session_state.analysis_result = None

orig_w, orig_h = image.size
disp_w = min(DISPLAY_WIDTH, orig_w)
scale = disp_w / orig_w
disp_h = int(orig_h * scale)

display_image = image.resize((disp_w, disp_h))

st.write(f"Original image size: {orig_w} × {orig_h}")


# =========================================================
# 2. ROI selection
# =========================================================
st.subheader("2. Select ROI")

btn1, btn2, btn3 = st.columns([1, 1, 1])

with btn1:
    if st.button("Select Blank", use_container_width=True):
        st.session_state.selection_mode = "blank"

with btn2:
    if st.button("Select Sample", use_container_width=True):
        st.session_state.selection_mode = "sample"

with btn3:
    if st.button("Reset ROI", use_container_width=True):
        st.session_state.blank_point_orig = None
        st.session_state.sample_points_orig = []
        st.session_state.analysis_result = None
        st.rerun()

if st.session_state.selection_mode == "blank":
    st.markdown(
        "<p style='color:#1f77b4; font-weight:700; font-size:20px;'>Select the Blank position on the image.</p>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<p style='color:#d62728; font-weight:700; font-size:20px;'>Select the Sample position on the image.</p>",
        unsafe_allow_html=True
    )

preview = draw_circle_preview(
    display_image=display_image,
    blank_orig=st.session_state.blank_point_orig,
    sample_orig_list=st.session_state.sample_points_orig,
    scale=scale,
    radius_disp=ROI_RADIUS_DISP
)

click = streamlit_image_coordinates(
    preview,
    key=f"coord_{st.session_state.image_hash}_{st.session_state.selection_mode}"
)

if click is not None and ("x" in click) and ("y" in click):
    x_disp = int(click["x"])
    y_disp = int(click["y"])

    x_orig = int(round(x_disp / scale))
    y_orig = int(round(y_disp / scale))

    x_orig = max(0, min(orig_w - 1, x_orig))
    y_orig = max(0, min(orig_h - 1, y_orig))

    if st.session_state.selection_mode == "blank":
        if st.session_state.blank_point_orig != (x_orig, y_orig):
            st.session_state.blank_point_orig = (x_orig, y_orig)
            st.session_state.analysis_result = None
            st.rerun()

    else:
        new_point = (x_orig, y_orig)

        if new_point not in st.session_state.sample_points_orig:
            if len(st.session_state.sample_points_orig) < 3:
                st.session_state.sample_points_orig.append(new_point)
            else:
                st.session_state.sample_points_orig[-1] = new_point

            st.session_state.analysis_result = None
            st.rerun()

pos1, pos2 = st.columns(2)
with pos1:
    st.write(f"Blank position: {st.session_state.blank_point_orig}")
with pos2:
    st.write(f"Sample position: {st.session_state.sample_points_orig}")

if st.button("Confirm ROI and Run Analysis", type="primary", use_container_width=True):
    if st.session_state.blank_point_orig is None or len(st.session_state.sample_points_orig) < 1:
        st.warning("Select one Blank position and at least one Sample position.")
    else:
        blank_x, blank_y = st.session_state.blank_point_orig

        roi_radius_orig = max(2, int(round(ROI_RADIUS_DISP / scale)))

        blank_rgb = robust_rgb_from_circle(
            image=image,
            x=blank_x,
            y=blank_y,
            radius=roi_radius_orig,
            keep_percent=KEEP_PERCENT
        )

        sample_rgbs = []
        for sample_x, sample_y in st.session_state.sample_points_orig:
            sample_rgb_each = robust_rgb_from_circle(
                image=image,
                x=sample_x,
                y=sample_y,
                radius=roi_radius_orig,
                keep_percent=KEEP_PERCENT
            )
            sample_rgbs.append(sample_rgb_each)

        sample_rgb = np.mean(np.vstack(sample_rgbs), axis=0)

        blank_lab = rgb_to_lab_value(blank_rgb)
        sample_lab = rgb_to_lab_value(sample_rgb)

        delta_l = float(sample_lab[0] - blank_lab[0])
        delta_a = float(sample_lab[1] - blank_lab[1])
        delta_b = float(sample_lab[2] - blank_lab[2])
        delta_e_val = delta_e(sample_lab, blank_lab)

        input_df = pd.DataFrame(
            [[
                sample_lab[0],
                sample_lab[1],
                sample_lab[2],
                delta_l,
                delta_a,
                delta_b,
                delta_e_val
            ]],
            columns=FEATURES
        )

        result = {
            "blank_rgb": blank_rgb,
            "sample_rgb": sample_rgb,
            "sample_rgbs": sample_rgbs,
            "blank_lab": blank_lab,
            "sample_lab": sample_lab,
            "deltaL": delta_l,
            "deltaa": delta_a,
            "deltab": delta_b,
            "deltaE": delta_e_val,
        }

        if delta_e_val < NO_METAL_THRESHOLD:
            result["predicted_group"] = "No_Metal_or_Below_Threshold"
            result["group_confidence"] = None
            result["group_top"] = []
            result["predicted_metal"] = None
            result["metal_confidence"] = None
            result["metal_top"] = []
            result["needs_metal_detail"] = False
            result["predicted_ppm"] = 0
            result["ppm_confidence"] = None
            result["ppm_top"] = []
            result["is_no_metal"] = True

        else:
            # =================================================
            # Step 1: Heavy metal group classification based on Group labels
            # =================================================
            group_pred = group_model.predict(input_df)[0]
            group_proba = group_model.predict_proba(input_df)[0]

            top_idx = np.argsort(group_proba)[::-1][:3]
            top_groups = [(group_classes[i], float(group_proba[i])) for i in top_idx]

            predicted_group = top_groups[0][0]
            predicted_conf = top_groups[0][1]

            group_df = df[df["Group"] == predicted_group].copy()

            # =================================================
            # Step 2: Detailed metal classification only for grouped labels such as Ag-Zn and Cd-Mn
            # Detailed heavy metal classification based on the heavy metal label
            # =================================================
            predicted_metal = predicted_group
            metal_conf = None
            metal_top = []
            needs_metal_detail = predicted_group in GROUPED_LABELS

            if needs_metal_detail and len(group_df) >= 2 and group_df["Metal"].nunique() >= 2:
                metal_model = build_metal_model(group_df, n_neighbors=METAL_K)
                metal_pred = metal_model.predict(input_df)[0]

                metal_knn = metal_model.named_steps["kneighborsclassifier"]
                metal_classes = metal_knn.classes_
                metal_proba = metal_model.predict_proba(input_df)[0]

                metal_idx = np.argsort(metal_proba)[::-1][:2]
                metal_top = [(metal_classes[i], float(metal_proba[i])) for i in metal_idx]

                predicted_metal = metal_top[0][0]
                metal_conf = metal_top[0][1]

            elif not needs_metal_detail:
                predicted_metal = predicted_group

            # =================================================
            # Step 3: Concentration classification based on the final predicted metal
            # =================================================
            ppm_top = []
            ppm_pred = None
            ppm_conf = None

            metal_df = df[df["Metal"] == predicted_metal].copy()

            if len(metal_df) < 1:
                metal_df = group_df.copy()

            if len(metal_df) >= 1:
                ppm_model = build_ppm_model(metal_df, n_neighbors=PPM_K)
                ppm_pred = int(ppm_model.predict(input_df)[0])

                ppm_knn = ppm_model.named_steps["kneighborsclassifier"]
                ppm_classes = ppm_knn.classes_
                ppm_proba = ppm_model.predict_proba(input_df)[0]

                ppm_idx = np.argsort(ppm_proba)[::-1][:3]
                ppm_top = [(int(ppm_classes[i]), float(ppm_proba[i])) for i in ppm_idx]

                if len(ppm_top) > 0:
                    ppm_conf = ppm_top[0][1]

            result["predicted_group"] = predicted_group
            result["group_confidence"] = predicted_conf
            result["group_top"] = top_groups
            result["predicted_metal"] = predicted_metal
            result["metal_confidence"] = metal_conf
            result["metal_top"] = metal_top
            result["needs_metal_detail"] = needs_metal_detail
            result["predicted_ppm"] = ppm_pred
            result["ppm_confidence"] = ppm_conf
            result["ppm_top"] = ppm_top
            result["is_no_metal"] = False

        st.session_state.analysis_result = result
        st.rerun()


# =========================================================
# 3. RGB extraction and CIE Lab conversion
# =========================================================
if st.session_state.analysis_result is not None:
    res = st.session_state.analysis_result

    st.subheader("3. RGB Extraction and CIE Lab Conversion")

    rgb1, rgb2 = st.columns(2)
    with rgb1:
        st.markdown("#### Blank")
        st.markdown(f"R: {round(float(res['blank_rgb'][0]), 3)}")
        st.markdown(f"G: {round(float(res['blank_rgb'][1]), 3)}")
        st.markdown(f"B: {round(float(res['blank_rgb'][2]), 3)}")

    with rgb2:
        st.markdown("#### Mean Sample RGB")
        st.markdown(f"R: {round(float(res['sample_rgb'][0]), 3)}")
        st.markdown(f"G: {round(float(res['sample_rgb'][1]), 3)}")
        st.markdown(f"B: {round(float(res['sample_rgb'][2]), 3)}")

        if len(res.get("sample_rgbs", [])) > 1:
            st.markdown("#### Individual Sample RGB")
            for idx, sample_rgb_each in enumerate(res["sample_rgbs"], start=1):
                st.markdown(
                    f"- Sample {idx}: "
                    f"R {round(float(sample_rgb_each[0]), 3)}, "
                    f"G {round(float(sample_rgb_each[1]), 3)}, "
                    f"B {round(float(sample_rgb_each[2]), 3)}"
                )

    cie1, cie2 = st.columns(2)
    with cie1:
        st.markdown("#### CIE Lab")
        st.markdown(f"L: {round(float(res['sample_lab'][0]), 3)}")
        st.markdown(f"a: {round(float(res['sample_lab'][1]), 3)}")
        st.markdown(f"b: {round(float(res['sample_lab'][2]), 3)}")

    with cie2:
        st.markdown("#### Color Difference")
        st.markdown(f"△L: {round(float(res['deltaL']), 3)}")
        st.markdown(f"△a: {round(float(res['deltaa']), 3)}")
        st.markdown(f"△b: {round(float(res['deltab']), 3)}")
        st.markdown(f"△E: {round(float(res['deltaE']), 3)}")

    # =====================================================
    # 4. Prediction results
    # =====================================================
    st.subheader("4. Prediction Results")

    if res["is_no_metal"]:
        st.warning("No heavy metal detected or the colorimetric response is weak.")
        st.write(f"△E: {res['deltaE']:.2f}")
    else:
        pred1, pred2 = st.columns(2)

        with pred1:
            st.success(f"Predicted Heavy Metal: {res['predicted_metal']}")

            if res.get("needs_metal_detail") and res.get("metal_confidence") is not None:
                st.write(f"Heavy Metal Confidence: {res['metal_confidence'] * 100:.1f}%")
            else:
                st.write(f"Heavy Metal Confidence: {res['group_confidence'] * 100:.1f}%")

        with pred2:
            st.success(f"Predicted Concentration: {res['predicted_ppm']} ppm")
            if res["ppm_confidence"] is not None:
                st.write(f"Concentration Confidence: {res['ppm_confidence'] * 100:.1f}%")

        cand1, cand2 = st.columns(2)

        with cand1:
            st.markdown("#### Similar Heavy Metal Candidates")

            if res.get("needs_metal_detail") and len(res.get("metal_top", [])) > 0:
                for m, p in res["metal_top"]:
                    st.markdown(f"- {m}: {p * 100:.1f}%")
            elif len(res["group_top"]) > 0:
                for g, p in res["group_top"]:
                    st.markdown(f"- {g}: {p * 100:.1f}%")

        with cand2:
            st.markdown("#### Similar Concentration Candidates")
            if len(res["ppm_top"]) > 0:
                for ppm_val, p in res["ppm_top"]:
                    st.markdown(f"- {ppm_val} ppm: {p * 100:.1f}%")

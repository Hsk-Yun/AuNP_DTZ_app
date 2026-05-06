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
# 기본 설정
# =========================================================
st.set_page_config(page_title="AD 전용 중금속 판독앱", layout="wide")

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

st.title("AD 전용 중금속 판독앱")


# =========================================================
# 이미지 로드
# =========================================================
def load_uploaded_image(uploaded_file):
    data = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    # RAW 파일 처리
    if filename.endswith(".raw"):
        try:
            import rawpy
        except ImportError:
            st.error("RAW 파일 처리를 위해 requirements.txt에 rawpy를 추가해야 합니다.")
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
            st.error("RAW 파일을 읽는 데 실패했습니다. 해당 RAW 포맷이 지원되지 않을 수 있습니다.")
            st.stop()
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

        return image, data

    # 일반 이미지 처리
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return image, data
    except Exception:
        st.error("이미지 파일을 여는 데 실패했습니다.")
        st.stop()


# =========================================================
# 학습 데이터 로드
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

    required_cols = ["Group", "ppm"] + FEATURES
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"CSV에 필요한 컬럼이 없습니다: {missing}")
        st.stop()

    for c in FEATURES + ["ppm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Group"] = df["Group"].astype(str).str.strip()
    df = df[~df["Group"].isin(["", "nan", "None"])]

    if "Metal" in df.columns:
        df["Metal"] = df["Metal"].astype(str).str.strip()
        df = df[~df["Metal"].str.lower().isin(["ag", "ag+", "silver", "은"])]

    df = df[~df["Group"].str.lower().isin(["ag", "ag+", "silver", "은"])]

    df = df[df["ppm"].between(PPM_MIN, PPM_MAX)]
    df = df[df["ppm"] > 0].copy()
    df = df.dropna(subset=["Group", "ppm"] + FEATURES)

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

        if all(c in aug.columns for c in ["Group", "ppm"] + FEATURES):
            for c in FEATURES + ["ppm"]:
                aug[c] = pd.to_numeric(aug[c], errors="coerce")

            aug["Group"] = aug["Group"].astype(str).str.strip()
            aug = aug[~aug["Group"].isin(["", "nan", "None"])]

            if "Metal" in aug.columns:
                aug["Metal"] = aug["Metal"].astype(str).str.strip()
                aug = aug[~aug["Metal"].str.lower().isin(["ag", "ag+", "silver", "은"])]

            aug = aug[~aug["Group"].str.lower().isin(["ag", "ag+", "silver", "은"])]
            aug = aug[aug["ppm"].between(PPM_MIN, PPM_MAX)]
            aug = aug[aug["ppm"] > 0].copy()
            aug = aug.dropna(subset=["Group", "ppm"] + FEATURES)

            df = pd.concat([df, aug], ignore_index=True)

    return df


# =========================================================
# 모델 생성
# =========================================================
@st.cache_resource
def build_group_model(df):
    X = df[FEATURES]
    y = df["Group"]

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=4,
            metric="euclidean",
            weights="distance"
        )
    )
    model.fit(X, y)
    return model


def build_ppm_model(group_df, n_neighbors=3):
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
    y = group_df["ppm"].astype(int)
    model.fit(X, y)
    return model


# =========================================================
# 색 계산 함수
# =========================================================
def rgb_to_lab_value(rgb):
    arr = np.array(rgb, dtype=np.float32).reshape(1, 1, 3) / 255.0
    lab = rgb2lab(arr)[0, 0]
    return lab


def delta_e(lab1, lab2):
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def robust_rgb_from_circle(image, x, y, radius, keep_percent=65):
    """
    원형 ROI 내부에서:
    1) 너무 밝거나 어두운 픽셀 제거
    2) 중심색에서 너무 멀리 벗어난 픽셀 제거
    => 반사광 / 그림자 / 배경 영향 완화
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
# 미리보기 이미지에 원 그리기
# =========================================================
def draw_circle_preview(display_image, blank_orig, sample_orig, scale, radius_disp):
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
    draw_one(sample_orig, "red", "Sample")

    return img


# =========================================================
# 데이터 준비
# =========================================================
df = load_training_data()
group_model = build_group_model(df)
group_knn = group_model.named_steps["kneighborsclassifier"]
group_classes = group_knn.classes_


# =========================================================
# 세션 상태 초기화
# =========================================================
if "blank_point_orig" not in st.session_state:
    st.session_state.blank_point_orig = None

if "sample_point_orig" not in st.session_state:
    st.session_state.sample_point_orig = None

if "selection_mode" not in st.session_state:
    st.session_state.selection_mode = "blank"

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "image_hash" not in st.session_state:
    st.session_state.image_hash = None


# =========================================================
# 1. 이미지 업로드
# =========================================================
st.subheader("1. 이미지를 업로드하세요")

input_method = st.radio(
    "",
    ["사진 업로드", "카메라 촬영"],
    horizontal=True,
    label_visibility="collapsed"
)

uploaded_file = None

if input_method == "사진 업로드":
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
    st.session_state.sample_point_orig = None
    st.session_state.analysis_result = None

orig_w, orig_h = image.size
disp_w = min(DISPLAY_WIDTH, orig_w)
scale = disp_w / orig_w
disp_h = int(orig_h * scale)

display_image = image.resize((disp_w, disp_h))

st.write(f"원본 이미지 크기: {orig_w} × {orig_h}")


# =========================================================
# 2. 영역 선택
# =========================================================
st.subheader("2. 영역 선택")

btn1, btn2, btn3 = st.columns([1, 1, 1])

with btn1:
    if st.button("Blank 선택", use_container_width=True):
        st.session_state.selection_mode = "blank"

with btn2:
    if st.button("Sample 선택", use_container_width=True):
        st.session_state.selection_mode = "sample"

with btn3:
    if st.button("영역 초기화", use_container_width=True):
        st.session_state.blank_point_orig = None
        st.session_state.sample_point_orig = None
        st.session_state.analysis_result = None
        st.rerun()

if st.session_state.selection_mode == "blank":
    st.markdown(
        "<p style='color:#1f77b4; font-weight:700; font-size:20px;'>이미지에서 Blank 위치를 선택하세요.</p>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<p style='color:#d62728; font-weight:700; font-size:20px;'>이미지에서 Sample 위치를 선택하세요.</p>",
        unsafe_allow_html=True
    )

preview = draw_circle_preview(
    display_image=display_image,
    blank_orig=st.session_state.blank_point_orig,
    sample_orig=st.session_state.sample_point_orig,
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
        if st.session_state.sample_point_orig != (x_orig, y_orig):
            st.session_state.sample_point_orig = (x_orig, y_orig)
            st.session_state.analysis_result = None
            st.rerun()

pos1, pos2 = st.columns(2)
with pos1:
    st.write(f"Blank 위치: {st.session_state.blank_point_orig}")
with pos2:
    st.write(f"Sample 위치: {st.session_state.sample_point_orig}")

if st.button("위치 확정 및 분석 실행", type="primary", use_container_width=True):
    if st.session_state.blank_point_orig is None or st.session_state.sample_point_orig is None:
        st.warning("Blank와 Sample 위치를 모두 선택하세요.")
    else:
        blank_x, blank_y = st.session_state.blank_point_orig
        sample_x, sample_y = st.session_state.sample_point_orig

        roi_radius_orig = max(2, int(round(ROI_RADIUS_DISP / scale)))

        blank_rgb = robust_rgb_from_circle(
            image=image,
            x=blank_x,
            y=blank_y,
            radius=roi_radius_orig,
            keep_percent=KEEP_PERCENT
        )

        sample_rgb = robust_rgb_from_circle(
            image=image,
            x=sample_x,
            y=sample_y,
            radius=roi_radius_orig,
            keep_percent=KEEP_PERCENT
        )

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
            result["predicted_ppm"] = 0
            result["ppm_confidence"] = None
            result["ppm_top"] = []
            result["is_no_metal"] = True

        else:
            group_pred = group_model.predict(input_df)[0]
            group_proba = group_model.predict_proba(input_df)[0]

            top_idx = np.argsort(group_proba)[::-1][:3]
            top_groups = [(group_classes[i], float(group_proba[i])) for i in top_idx]

            predicted_group = top_groups[0][0]
            predicted_conf = top_groups[0][1]

            group_df = df[df["Group"] == predicted_group].copy()

            ppm_top = []
            ppm_pred = None
            ppm_conf = None

            if len(group_df) >= 1:
                ppm_model = build_ppm_model(group_df, n_neighbors=3)
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
            result["predicted_ppm"] = ppm_pred
            result["ppm_confidence"] = ppm_conf
            result["ppm_top"] = ppm_top
            result["is_no_metal"] = False

        st.session_state.analysis_result = result
        st.rerun()


# =========================================================
# 3. RGB 추출 및 CIE 변환
# =========================================================
if st.session_state.analysis_result is not None:
    res = st.session_state.analysis_result

    st.subheader("3. RGB 추출 및 CIE 변환")

    rgb1, rgb2 = st.columns(2)
    with rgb1:
        st.markdown("#### Blank")
        st.markdown(f"R: {round(float(res['blank_rgb'][0]), 3)}")
        st.markdown(f"G: {round(float(res['blank_rgb'][1]), 3)}")
        st.markdown(f"B: {round(float(res['blank_rgb'][2]), 3)}")

    with rgb2:
        st.markdown("#### Sample")
        st.markdown(f"R: {round(float(res['sample_rgb'][0]), 3)}")
        st.markdown(f"G: {round(float(res['sample_rgb'][1]), 3)}")
        st.markdown(f"B: {round(float(res['sample_rgb'][2]), 3)}")

    cie1, cie2 = st.columns(2)
    with cie1:
        st.markdown("#### CIE Lab")
        st.markdown(f"L: {round(float(res['sample_lab'][0]), 3)}")
        st.markdown(f"a: {round(float(res['sample_lab'][1]), 3)}")
        st.markdown(f"b: {round(float(res['sample_lab'][2]), 3)}")

    with cie2:
        st.markdown("#### 변화량")
        st.markdown(f"△L: {round(float(res['deltaL']), 3)}")
        st.markdown(f"△a: {round(float(res['deltaa']), 3)}")
        st.markdown(f"△b: {round(float(res['deltab']), 3)}")
        st.markdown(f"△E: {round(float(res['deltaE']), 3)}")


    # =====================================================
    # 4. 예측 결과
    # =====================================================
    st.subheader("4. 예측 결과")

    if res["is_no_metal"]:
        st.warning("중금속 미검출 또는 반응이 약합니다.")
        st.write(f"△E: {res['deltaE']:.2f}")
    else:
        pred1, pred2 = st.columns(2)

        with pred1:
            st.success(f"예상 중금속군: {res['predicted_group']}")
            st.write(f"중금속 신뢰도: {res['group_confidence'] * 100:.1f}%")

        with pred2:
            st.success(f"예상 농도: {res['predicted_ppm']} ppm")
            if res["ppm_confidence"] is not None:
                st.write(f"농도 신뢰도: {res['ppm_confidence'] * 100:.1f}%")

        cand1, cand2 = st.columns(2)

        with cand1:
            st.markdown("#### 유사 중금속군 후보")
            if len(res["group_top"]) > 0:
                for g, p in res["group_top"]:
                    st.markdown(f"- {g}: {p * 100:.1f}%")

        with cand2:
            st.markdown("#### 유사 농도 후보")
            if len(res["ppm_top"]) > 0:
                for ppm_val, p in res["ppm_top"]:
                    st.markdown(f"- {ppm_val} ppm: {p * 100:.1f}%")

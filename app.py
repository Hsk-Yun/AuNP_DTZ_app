import os
import io
import hashlib

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2lab
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from streamlit_image_coordinates import streamlit_image_coordinates


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(page_title="중금속 이미지 색센싱 웹앱", layout="wide")

DATA_PATH = "training_data.csv"
AUG_PATH = "training_data_augmented.csv"
FEATURES = ["L", "a", "b", "deltaE"]


# =========================================================
# 데이터 로드
# =========================================================
@st.cache_data
def load_training_data():
    if not os.path.exists(DATA_PATH):
        st.error("training_data.csv 파일이 app.py와 같은 폴더에 없습니다.")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    # 열 이름 보정
    if "dE" in df.columns and "deltaE" not in df.columns:
        df = df.rename(columns={"dE": "deltaE"})

    required_cols = ["Group", "ppm"] + FEATURES
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"CSV에 필요한 열이 없습니다: {missing}")
        st.stop()

    # 수치형 변환
    for c in FEATURES + ["ppm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 그룹 정리
    df["Group"] = df["Group"].astype(str).str.strip()
    df = df[~df["Group"].isin(["", "nan", "None"])]

    # 학습에 필요한 값만
    df = df.dropna(subset=["Group", "ppm"] + FEATURES)

    # 0ppm 제거
    df = df[df["ppm"] > 0].copy()

    # 검증 저장 데이터가 있으면 합치기
    if os.path.exists(AUG_PATH):
        aug = pd.read_csv(AUG_PATH)

        if "dE" in aug.columns and "deltaE" not in aug.columns:
            aug = aug.rename(columns={"dE": "deltaE"})

        for c in FEATURES + ["ppm"]:
            if c in aug.columns:
                aug[c] = pd.to_numeric(aug[c], errors="coerce")

        if "Group" in aug.columns:
            aug["Group"] = aug["Group"].astype(str).str.strip()
            aug = aug[~aug["Group"].isin(["", "nan", "None"])]
            aug = aug.dropna(subset=["Group", "ppm"] + FEATURES)
            aug = aug[aug["ppm"] > 0].copy()

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
def rgb_to_lab(rgb):
    arr = np.array(rgb, dtype=np.float32).reshape(1, 1, 3) / 255.0
    lab = rgb2lab(arr)[0, 0]
    return lab


def delta_e(lab1, lab2):
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def robust_rgb_from_circle(image, x, y, radius, keep_percent=65):
    """
    선택한 점 중심의 원형 ROI 내부 픽셀을 사용.
    반사광/그림자/배경 영향을 줄이기 위해 튀는 픽셀 일부 제거.
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
        return np.array([0, 0, 0], dtype=np.float32), 0.0

    yy, xx = np.mgrid[y1:y2, x1:x2]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    mask_circle = dist <= radius

    pixels = crop[mask_circle]
    if len(pixels) == 0:
        return np.array([0, 0, 0], dtype=np.float32), 0.0

    # 중심부 패치 기준색
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

    if brightness_mask.sum() >= 10:
        valid_d = color_dist[brightness_mask]
    else:
        valid_d = color_dist

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
    kept_ratio = len(selected) / len(pixels)

    return robust_rgb, kept_ratio


# =========================================================
# 미리보기 그리기
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

        # 원만 그리기
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
# 메인 로직
# =========================================================
df = load_training_data()
group_model = build_group_model(df)
group_knn = group_model.named_steps["kneighborsclassifier"]
group_classes = group_knn.classes_

st.title("중금속 이미지 색센싱 웹앱")
st.caption("CIE Lab 기반 distance-weighted kNN 분류")

left_col, main_col = st.columns([1, 3], gap="large")

with left_col:
    st.markdown("### Model Setting")
    st.write("**Features:** L*, a*, b*, ΔE")
    st.write("**1차 분류:** kNN 8 Group")
    st.write("**2차 분류:** 예측 Group 내 ppm 분류")
    st.write("**k (Group):** 4")
    st.write("**k (ppm):** 3")
    st.write("**Metric:** Euclidean")
    st.write("**Weight:** By Distance")
    st.write(f"**Training samples:** {len(df)}")

    st.markdown("---")

    display_width = st.slider(
        "표시 이미지 너비(px)",
        min_value=280,
        max_value=1000,
        value=700,
        step=20
    )

    no_metal_threshold = st.slider(
        "No Metal 판단 ΔE 기준",
        min_value=0.0,
        max_value=30.0,
        value=8.0,
        step=0.5
    )

    keep_percent = st.slider(
        "노이즈 제거 강도",
        min_value=40,
        max_value=90,
        value=65,
        step=5
    )

with main_col:
    st.subheader("1. 이미지 입력")

    input_method = st.radio(
        "이미지 입력 방식",
        ["사진 업로드", "카메라 촬영"],
        horizontal=True
    )

    uploaded_file = None
    if input_method == "사진 업로드":
        uploaded_file = st.file_uploader("키트 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("키트를 촬영하세요")

    if uploaded_file is None:
        st.info("사진을 업로드하거나 카메라로 촬영하세요.")
        st.stop()

    image_bytes = uploaded_file.getvalue()
    current_hash = hashlib.md5(image_bytes).hexdigest()

    if st.session_state.image_hash != current_hash:
        st.session_state.image_hash = current_hash
        st.session_state.blank_point_orig = None
        st.session_state.sample_point_orig = None
        st.session_state.analysis_result = None

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = image.size

    disp_w = min(display_width, orig_w)
    scale = disp_w / orig_w
    disp_h = int(orig_h * scale)

    display_image = image.resize((disp_w, disp_h))

    st.write(f"원본 이미지 크기: {orig_w} × {orig_h}")

    roi_radius_disp = st.slider(
        "점 주변 RGB 평균 영역 크기(화면 표시 기준 px)",
        min_value=6,
        max_value=80,
        value=24,
        step=2
    )

    st.subheader("2. ROI 점 선택")

    btn1, btn2, btn3 = st.columns([1, 1, 1])

    with btn1:
        if st.button("Blank 선택", use_container_width=True):
            st.session_state.selection_mode = "blank"

    with btn2:
        if st.button("Sample 선택", use_container_width=True):
            st.session_state.selection_mode = "sample"

    with btn3:
        if st.button("ROI 초기화", use_container_width=True):
            st.session_state.blank_point_orig = None
            st.session_state.sample_point_orig = None
            st.session_state.analysis_result = None
            st.rerun()

    if st.session_state.selection_mode == "blank":
        st.info("현재 모드: Blank 선택. 이미지에서 Blank 위치를 클릭하세요.")
    else:
        st.info("현재 모드: Sample 선택. 이미지에서 Sample 위치를 클릭하세요.")

    preview = draw_circle_preview(
        display_image=display_image,
        blank_orig=st.session_state.blank_point_orig,
        sample_orig=st.session_state.sample_point_orig,
        scale=scale,
        radius_disp=roi_radius_disp
    )

    click = streamlit_image_coordinates(
        preview,
        key=f"coord_{st.session_state.image_hash}_{st.session_state.selection_mode}"
    )

    # 클릭 좌표 저장
    if click is not None and ("x" in click) and ("y" in click):
        x_disp = int(click["x"])
        y_disp = int(click["y"])

        x_orig = int(round(x_disp / scale))
        y_orig = int(round(y_disp / scale))

        # 이미지 범위 보정
        x_orig = max(0, min(orig_w - 1, x_orig))
        y_orig = max(0, min(orig_h - 1, y_orig))

        if st.session_state.selection_mode == "blank":
            if st.session_state.blank_point_orig != (x_orig, y_orig):
                st.session_state.blank_point_orig = (x_orig, y_orig)
                st.session_state.analysis_result = None
                st.rerun()

        elif st.session_state.selection_mode == "sample":
            if st.session_state.sample_point_orig != (x_orig, y_orig):
                st.session_state.sample_point_orig = (x_orig, y_orig)
                st.session_state.analysis_result = None
                st.rerun()

    pos1, pos2 = st.columns(2)
    with pos1:
        st.write(f"Blank 위치: {st.session_state.blank_point_orig}")
    with pos2:
        st.write(f"Sample 위치: {st.session_state.sample_point_orig}")

    if st.session_state.blank_point_orig is None or st.session_state.sample_point_orig is None:
        st.info("Blank 점과 Sample 점을 각각 선택해야 분석할 수 있습니다.")

    if st.button("위치 확정 및 분석 실행", type="primary", use_container_width=True):
        if st.session_state.blank_point_orig is None or st.session_state.sample_point_orig is None:
            st.warning("Blank와 Sample 위치를 먼저 모두 선택하세요.")
        else:
            blank_x, blank_y = st.session_state.blank_point_orig
            sample_x, sample_y = st.session_state.sample_point_orig

            roi_radius_orig = max(2, int(round(roi_radius_disp / scale)))

            blank_rgb, blank_kept = robust_rgb_from_circle(
                image=image,
                x=blank_x,
                y=blank_y,
                radius=roi_radius_orig,
                keep_percent=keep_percent
            )

            sample_rgb, sample_kept = robust_rgb_from_circle(
                image=image,
                x=sample_x,
                y=sample_y,
                radius=roi_radius_orig,
                keep_percent=keep_percent
            )

            blank_lab = rgb_to_lab(blank_rgb)
            sample_lab = rgb_to_lab(sample_rgb)
            dE_val = delta_e(sample_lab, blank_lab)

            input_df = pd.DataFrame(
                [[sample_lab[0], sample_lab[1], sample_lab[2], dE_val]],
                columns=FEATURES
            )

            result = {
                "blank_rgb": blank_rgb,
                "sample_rgb": sample_rgb,
                "blank_lab": blank_lab,
                "sample_lab": sample_lab,
                "deltaE": dE_val,
                "blank_kept": blank_kept,
                "sample_kept": sample_kept,
            }

            # No metal 판단
            if dE_val < no_metal_threshold:
                result["predicted_group"] = "No_Metal_or_Below_Threshold"
                result["group_confidence"] = None
                result["group_top"] = []
                result["predicted_ppm"] = 0
                result["ppm_top"] = []
                result["is_no_metal"] = True
            else:
                group_pred = group_model.predict(input_df)[0]
                group_proba = group_model.predict_proba(input_df)[0]

                top_idx = np.argsort(group_proba)[::-1][:3]
                top_groups = [
                    (group_classes[i], float(group_proba[i]))
                    for i in top_idx
                ]

                predicted_group = top_groups[0][0]
                predicted_conf = top_groups[0][1]

                group_df = df[df["Group"] == predicted_group].copy()

                ppm_top = []
                ppm_pred = None

                if len(group_df) >= 1:
                    ppm_model = build_ppm_model(group_df, n_neighbors=3)
                    ppm_pred = int(ppm_model.predict(input_df)[0])

                    ppm_knn = ppm_model.named_steps["kneighborsclassifier"]
                    ppm_classes = ppm_knn.classes_
                    ppm_proba = ppm_model.predict_proba(input_df)[0]

                    ppm_idx = np.argsort(ppm_proba)[::-1][:3]
                    ppm_top = [
                        (int(ppm_classes[i]), float(ppm_proba[i]))
                        for i in ppm_idx
                    ]

                result["predicted_group"] = predicted_group
                result["group_confidence"] = predicted_conf
                result["group_top"] = top_groups
                result["predicted_ppm"] = ppm_pred
                result["ppm_top"] = ppm_top
                result["is_no_metal"] = False

            st.session_state.analysis_result = result
            st.rerun()

    # =====================================================
    # 결과 표시
    # =====================================================
    if st.session_state.analysis_result is not None:
        res = st.session_state.analysis_result

        st.subheader("3. 추출 RGB 결과")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Blank")
            st.write({
                "R": round(float(res["blank_rgb"][0]), 3),
                "G": round(float(res["blank_rgb"][1]), 3),
                "B": round(float(res["blank_rgb"][2]), 3),
                "사용 픽셀 비율": f"{res['blank_kept'] * 100:.1f}%"
            })

        with c2:
            st.markdown("#### Sample")
            st.write({
                "R": round(float(res["sample_rgb"][0]), 3),
                "G": round(float(res["sample_rgb"][1]), 3),
                "B": round(float(res["sample_rgb"][2]), 3),
                "사용 픽셀 비율": f"{res['sample_kept'] * 100:.1f}%"
            })

        st.subheader("4. CIE Lab 계산값")

        st.write({
            "L*": round(float(res["sample_lab"][0]), 3),
            "a*": round(float(res["sample_lab"][1]), 3),
            "b*": round(float(res["sample_lab"][2]), 3),
            "ΔE": round(float(res["deltaE"]), 3),
        })

        st.subheader("5. 예측 결과")

        if res["is_no_metal"]:
            st.warning("예상 결과: 중금속 미검출 또는 반응 미약")
            st.write(f"ΔE = **{res['deltaE']:.2f}**, 설정 기준값 = **{no_metal_threshold:.2f}**")
        else:
            st.success(f"예상 중금속군: **{res['predicted_group']}**")

            if res["group_confidence"] is not None:
                st.write(f"예측 신뢰도: **{res['group_confidence'] * 100:.1f}%**")

            if res["predicted_ppm"] is not None:
                st.success(f"예상 농도: **{res['predicted_ppm']} ppm**")

            if len(res["group_top"]) > 0:
                st.markdown("#### 유사 중금속군 후보")
                for g, p in res["group_top"]:
                    st.write(f"- {g}: {p * 100:.1f}%")

            if len(res["ppm_top"]) > 0:
                st.markdown("#### 유사 농도 후보")
                for ppm_val, p in res["ppm_top"]:
                    st.write(f"- {ppm_val} ppm: {p * 100:.1f}%")

        # =================================================
        # 검증 데이터 저장
        # =================================================
        st.subheader("6. 검증 데이터 저장(선택)")

        group_options = list(group_classes) + ["No_Metal_or_Below_Threshold"]
        default_group = res["predicted_group"] if res["predicted_group"] in group_options else group_options[0]
        default_ppm = int(res["predicted_ppm"]) if res["predicted_ppm"] is not None else 0

        with st.form("save_verified_form"):
            actual_group = st.selectbox(
                "실제 Group (정밀검사 후 입력)",
                options=group_options,
                index=group_options.index(default_group)
            )
            actual_metal = st.text_input("실제 Metal (선택 입력)", value="")
            actual_ppm = st.number_input(
                "실제 ppm",
                min_value=0,
                max_value=10000,
                value=default_ppm
            )

            save_btn = st.form_submit_button("검증 데이터 저장")

            if save_btn:
                save_row = {
                    "Metal": actual_metal,
                    "Group": actual_group,
                    "ppm": actual_ppm,
                    "L": float(res["sample_lab"][0]),
                    "a": float(res["sample_lab"][1]),
                    "b": float(res["sample_lab"][2]),
                    "deltaE": float(res["deltaE"]),
                }

                save_df = pd.DataFrame([save_row])

                if os.path.exists(AUG_PATH):
                    old = pd.read_csv(AUG_PATH)
                    new = pd.concat([old, save_df], ignore_index=True)
                else:
                    new = save_df

                new.to_csv(AUG_PATH, index=False, encoding="utf-8-sig")
                st.cache_data.clear()
                st.success(f"{AUG_PATH} 파일에 저장했습니다.")

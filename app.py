import os
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2lab
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="AuNP@DTZ Heavy Metal Detector", layout="wide")

DATA_PATH = "training_data.csv"
AUG_PATH = "training_data_augmented.csv"
FEATURES = ["L", "a", "b", "deltaE"]

st.title("AuNP@DTZ 중금속 이미지 색센싱 웹앱")
st.caption("CIE Lab 기반 distance-weighted kNN 8 Group 분류 모델")


@st.cache_data
def load_training_data():
    if not os.path.exists(DATA_PATH):
        st.error("training_data.csv 파일이 app.py와 같은 폴더에 없습니다.")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    if "deltaE" not in df.columns and "dE" in df.columns:
        df = df.rename(columns={"dE": "deltaE"})

    required = ["Group", "ppm"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CSV에 필요한 열이 없습니다: {missing}")
        st.stop()

    for col in FEATURES + ["ppm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Group"] = df["Group"].astype(str).str.strip()
    df = df[~df["Group"].isin(["", "nan", "None"])]
    df = df.dropna(subset=["Group"] + FEATURES + ["ppm"])

    # Orange3 분석 조건과 동일하게 0 ppm blank는 학습에서 제외
    df = df[df["ppm"] > 0]

    # 로컬에서만 실질적으로 유지됨. 클라우드에서는 재시작 시 사라질 수 있음.
    if os.path.exists(AUG_PATH):
        aug = pd.read_csv(AUG_PATH)

        if "deltaE" not in aug.columns and "dE" in aug.columns:
            aug = aug.rename(columns={"dE": "deltaE"})

        for col in FEATURES + ["ppm"]:
            if col in aug.columns:
                aug[col] = pd.to_numeric(aug[col], errors="coerce")

        if "Group" in aug.columns:
            aug["Group"] = aug["Group"].astype(str).str.strip()
            aug = aug[~aug["Group"].isin(["", "nan", "None"])]
            aug = aug.dropna(subset=["Group"] + FEATURES + ["ppm"])
            aug = aug[aug["ppm"] > 0]
            df = pd.concat([df, aug], ignore_index=True)

    return df


def rgb_to_lab(rgb):
    arr = np.array(rgb, dtype=np.float32).reshape(1, 1, 3) / 255.0
    return rgb2lab(arr)[0, 0]


def calc_delta_e(sample_lab, blank_lab):
    return float(np.sqrt(np.sum((sample_lab - blank_lab) ** 2)))


def robust_rgb_from_point(image, x, y, roi_size, keep_percent=65):
    """
    ROI 안에서 반사광, 그림자, 종이 배경 등 튀는 픽셀을 일부 제거한 뒤
    median RGB를 계산한다.
    """
    arr = np.array(image).astype(np.float32)
    h, w, _ = arr.shape

    half = roi_size // 2
    x1 = max(0, int(x) - half)
    x2 = min(w, int(x) + half)
    y1 = max(0, int(y) - half)
    y2 = min(h, int(y) + half)

    crop = arr[y1:y2, x1:x2, :]
    if crop.size == 0:
        return np.array([0, 0, 0], dtype=np.float32), 0.0

    pixels = crop.reshape(-1, 3)

    cx = int(x) - x1
    cy = int(y) - y1

    px1 = max(0, cx - 2)
    px2 = min(crop.shape[1], cx + 3)
    py1 = max(0, cy - 2)
    py2 = min(crop.shape[0], cy + 3)

    center_patch = crop[py1:py2, px1:px2, :].reshape(-1, 3)
    ref_rgb = np.median(center_patch, axis=0) if len(center_patch) else np.median(pixels, axis=0)

    brightness = pixels.mean(axis=1)
    low_b, high_b = np.percentile(brightness, [2, 98])
    brightness_mask = (brightness >= low_b) & (brightness <= high_b)

    color_distance = np.linalg.norm(pixels - ref_rgb, axis=1)
    valid_distances = color_distance[brightness_mask]

    if len(valid_distances) < 10:
        mask = brightness_mask
    else:
        threshold = np.percentile(valid_distances, keep_percent)
        mask = brightness_mask & (color_distance <= threshold)

    if mask.sum() < 10:
        selected_pixels = pixels[brightness_mask] if brightness_mask.sum() >= 10 else pixels
    else:
        selected_pixels = pixels[mask]

    robust_rgb = np.median(selected_pixels, axis=0)
    kept_ratio = len(selected_pixels) / len(pixels)

    return robust_rgb, kept_ratio


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


def predict_ppm_by_group(df, input_x, predicted_group, n_neighbors=3):
    group_df = df[df["Group"] == predicted_group].copy()
    group_df = group_df.dropna(subset=FEATURES + ["ppm"])

    if group_df.empty:
        return None, [], []

    X_ppm = group_df[FEATURES]
    y_ppm = group_df["ppm"].astype(int)

    k = min(n_neighbors, len(group_df))

    ppm_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=k,
            metric="euclidean",
            weights="distance"
        )
    )
    ppm_model.fit(X_ppm, y_ppm)

    ppm_pred = ppm_model.predict(input_x)[0]
    ppm_proba = ppm_model.predict_proba(input_x)[0]
    ppm_classes = ppm_model.named_steps["kneighborsclassifier"].classes_

    return int(ppm_pred), ppm_classes, ppm_proba


def draw_roi_preview(image, blank_x, blank_y, sample_x, sample_y, roi_size):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    half = roi_size // 2

    def draw_one(x, y, color, label):
        box = (x - half, y - half, x + half, y + half)
        draw.rectangle(box, outline=color, width=4)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        draw.text((x + 8, y + 8), label, fill=color)

    draw_one(blank_x, blank_y, "blue", "Blank")
    draw_one(sample_x, sample_y, "red", "Sample")

    return img


df = load_training_data()
group_model = build_group_model(df)
group_classes = group_model.named_steps["kneighborsclassifier"].classes_

positive_min_deltaE = float(df["deltaE"].min())
positive_q05_deltaE = float(df["deltaE"].quantile(0.05))

st.sidebar.header("Model Setting")
st.sidebar.write("Features: L*, a*, b*, ΔE")
st.sidebar.write("1st: kNN 8 Group classification")
st.sidebar.write("2nd: kNN ppm classification within predicted Group")
st.sidebar.write("k for Group = 4")
st.sidebar.write("k for ppm = 3")
st.sidebar.write("Metric: Euclidean")
st.sidebar.write("Weight: By Distance")
st.sidebar.write(f"Training samples: {len(df)}")

st.sidebar.divider()

st.sidebar.header("Display Setting")
display_width = st.sidebar.slider(
    "표시 이미지 너비(px)",
    min_value=280,
    max_value=1000,
    value=360,
    step=20,
    help="폰에서는 320~420px, PC에서는 700~1000px 정도가 적당합니다."
)

st.sidebar.divider()

st.sidebar.header("Detection Setting")
no_metal_threshold = st.sidebar.number_input(
    "No metal threshold 기준 ΔE",
    min_value=0.0,
    max_value=50.0,
    value=8.0,
    step=0.5
)

confidence_warning_threshold = st.sidebar.slider(
    "Low confidence warning 기준",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05
)

keep_percent = st.sidebar.slider(
    "ROI 노이즈 제거 강도",
    min_value=40,
    max_value=90,
    value=65,
    step=5,
    help="값이 낮을수록 중심 색과 비슷한 픽셀만 사용합니다."
)

st.sidebar.write(f"Training positive min ΔE: {positive_min_deltaE:.2f}")
st.sidebar.write(f"Training positive 5% ΔE: {positive_q05_deltaE:.2f}")

st.subheader("1. 이미지 입력")

input_method = st.radio(
    "이미지 입력 방식",
    ["사진 업로드", "카메라 촬영"],
    horizontal=True
)

if input_method == "사진 업로드":
    image_file = st.file_uploader("키트 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("키트를 촬영하세요")

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    orig_w, orig_h = image.size

    display_width = min(display_width, orig_w)
    scale = display_width / orig_w

    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)

    display_image = image.resize((disp_w, disp_h))

    st.write(f"원본 이미지 크기: {orig_w} × {orig_h}")
    st.write("아래 슬라이더로 Blank와 Sample 위치를 맞춘 뒤 **분석 실행** 버튼을 누르세요.")

    roi_size_display = st.slider(
        "점 주변 RGB 평균 영역 크기(화면 표시 기준 px)",
        min_value=6,
        max_value=80,
        value=20,
        step=2
    )

    st.subheader("2. ROI 위치 지정")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Blank 위치")
        blank_x_disp = st.slider("Blank X", 0, disp_w - 1, int(disp_w * 0.30))
        blank_y_disp = st.slider("Blank Y", 0, disp_h - 1, int(disp_h * 0.50))

    with col_b:
        st.markdown("### Sample 위치")
        sample_x_disp = st.slider("Sample X", 0, disp_w - 1, int(disp_w * 0.65))
        sample_y_disp = st.slider("Sample Y", 0, disp_h - 1, int(disp_h * 0.50))

    preview = draw_roi_preview(
        display_image,
        blank_x_disp,
        blank_y_disp,
        sample_x_disp,
        sample_y_disp,
        roi_size_display
    )

    st.image(preview, caption="ROI 위치 미리보기", use_container_width=False)

    if st.button("분석 실행"):
        blank_x = blank_x_disp / scale
        blank_y = blank_y_disp / scale
        sample_x = sample_x_disp / scale
        sample_y = sample_y_disp / scale
        roi_size_orig = max(1, int(roi_size_display / scale))

        blank_rgb, blank_kept = robust_rgb_from_point(
            image, blank_x, blank_y, roi_size_orig, keep_percent=keep_percent
        )
        sample_rgb, sample_kept = robust_rgb_from_point(
            image, sample_x, sample_y, roi_size_orig, keep_percent=keep_percent
        )

        blank_lab = rgb_to_lab(blank_rgb)
        sample_lab = rgb_to_lab(sample_rgb)
        delta_e = calc_delta_e(sample_lab, blank_lab)

        input_x = pd.DataFrame(
            [[sample_lab[0], sample_lab[1], sample_lab[2], delta_e]],
            columns=FEATURES
        )

        st.subheader("3. 추출된 RGB")
        rgb_col1, rgb_col2 = st.columns(2)

        with rgb_col1:
            st.markdown("### Blank RGB")
            st.write({
                "R": round(float(blank_rgb[0]), 3),
                "G": round(float(blank_rgb[1]), 3),
                "B": round(float(blank_rgb[2]), 3),
                "사용 픽셀 비율": f"{blank_kept * 100:.1f}%"
            })

        with rgb_col2:
            st.markdown("### Sample RGB")
            st.write({
                "R": round(float(sample_rgb[0]), 3),
                "G": round(float(sample_rgb[1]), 3),
                "B": round(float(sample_rgb[2]), 3),
                "사용 픽셀 비율": f"{sample_kept * 100:.1f}%"
            })

        st.subheader("4. 예측 결과")

        if delta_e < no_metal_threshold:
            st.warning("예상 결과: 중금속 미검출 또는 반응 미약")
            st.write(f"ΔE = **{delta_e:.2f}**")
            st.write(f"설정된 No metal threshold = **{no_metal_threshold:.2f}**")
            st.write("Blank와 Sample의 색 차이가 작아 중금속 반응이 뚜렷하지 않은 것으로 판단했습니다.")

            st.write("계산된 CIE Lab 특징값")
            st.write({
                "L*": round(float(sample_lab[0]), 3),
                "a*": round(float(sample_lab[1]), 3),
                "b*": round(float(sample_lab[2]), 3),
                "ΔE": round(float(delta_e), 3),
            })

            st.session_state["last_result"] = {
                "is_no_metal": True,
                "blank_R": float(blank_rgb[0]),
                "blank_G": float(blank_rgb[1]),
                "blank_B": float(blank_rgb[2]),
                "sample_R": float(sample_rgb[0]),
                "sample_G": float(sample_rgb[1]),
                "sample_B": float(sample_rgb[2]),
                "L": float(sample_lab[0]),
                "a": float(sample_lab[1]),
                "b": float(sample_lab[2]),
                "deltaE": float(delta_e),
                "predicted_group": "No_Metal_or_Below_Threshold",
                "group_confidence": None,
                "predicted_ppm": 0,
            }

        else:
            group_pred = group_model.predict(input_x)[0]
            group_proba = group_model.predict_proba(input_x)[0]
            group_top_idx = np.argsort(group_proba)[::-1][:3]
            top_confidence = float(group_proba[group_top_idx[0]])

            ppm_pred, ppm_classes, ppm_proba = predict_ppm_by_group(
                df, input_x, group_pred, n_neighbors=3
            )

            st.success(f"예상 중금속군: {group_pred}")
            st.write(f"중금속군 예측 신뢰도: **{top_confidence * 100:.1f}%**")

            if top_confidence < confidence_warning_threshold:
                st.warning("예측 신뢰도가 낮습니다. ROI 위치, 조명, 반사광 여부를 확인하거나 재촬영을 권장합니다.")

            if ppm_pred is not None:
                st.success(f"예상 농도 구간: {int(ppm_pred)} ppm")
            else:
                st.warning("예상 농도 구간을 계산할 수 없습니다.")

            st.write("계산된 CIE Lab 특징값")
            st.write({
                "L*": round(float(sample_lab[0]), 3),
                "a*": round(float(sample_lab[1]), 3),
                "b*": round(float(sample_lab[2]), 3),
                "ΔE": round(float(delta_e), 3),
            })

            st.write("중금속군 유사 후보")
            for idx in group_top_idx:
                st.write(f"{group_classes[idx]}: {group_proba[idx] * 100:.1f}%")

            if ppm_pred is not None and len(ppm_classes) > 0:
                st.write("농도 후보군")
                ppm_top_idx = np.argsort(ppm_proba)[::-1][:3]
                for idx in ppm_top_idx:
                    st.write(f"{int(ppm_classes[idx])} ppm: {ppm_proba[idx] * 100:.1f}%")

            st.session_state["last_result"] = {
                "is_no_metal": False,
                "blank_R": float(blank_rgb[0]),
                "blank_G": float(blank_rgb[1]),
                "blank_B": float(blank_rgb[2]),
                "sample_R": float(sample_rgb[0]),
                "sample_G": float(sample_rgb[1]),
                "sample_B": float(sample_rgb[2]),
                "L": float(sample_lab[0]),
                "a": float(sample_lab[1]),
                "b": float(sample_lab[2]),
                "deltaE": float(delta_e),
                "predicted_group": group_pred,
                "group_confidence": top_confidence,
                "predicted_ppm": ppm_pred,
            }

else:
    st.info("사진을 업로드하거나 카메라로 촬영하세요.")

if "last_result" in st.session_state:
    st.subheader("5. 검증된 데이터 추가 저장")

    st.warning(
        "앱 예측값을 그대로 정답으로 저장하면 오류가 누적될 수 있습니다. "
        "정밀검사 또는 표준시료로 실제 정답을 확인한 경우에만 추가하세요."
    )

    result = st.session_state["last_result"]
    group_options = list(group_classes) + ["No_Metal_or_Below_Threshold"]

    if result["predicted_group"] in group_options:
        default_index = group_options.index(result["predicted_group"])
    else:
        default_index = 0

    with st.form("verified_data_form"):
        actual_group = st.selectbox(
            "정밀검사 후 실제 Group",
            group_options,
            index=default_index
        )

        actual_metal = st.text_input("실제 Metal 선택사항", value="")
        default_ppm = int(result["predicted_ppm"]) if result["predicted_ppm"] is not None else 0

        actual_ppm = st.number_input(
            "실제 ppm",
            min_value=0,
            max_value=10000,
            value=default_ppm
        )

        save_clicked = st.form_submit_button("검증된 데이터로 저장")

        if save_clicked:
            new_row = {
                "Metal": actual_metal,
                "Group": actual_group,
                "ppm": actual_ppm,
                "R": result["sample_R"],
                "G": result["sample_G"],
                "B": result["sample_B"],
                "L": result["L"],
                "a": result["a"],
                "b": result["b"],
                "deltaE": result["deltaE"],
                "Predicted_Group": result["predicted_group"],
                "Predicted_ppm": result["predicted_ppm"],
                "Group_Confidence": result["group_confidence"],
            }

            new_df = pd.DataFrame([new_row])

            if os.path.exists(AUG_PATH):
                old_df = pd.read_csv(AUG_PATH)
                save_df = pd.concat([old_df, new_df], ignore_index=True)
            else:
                save_df = new_df

            save_df.to_csv(AUG_PATH, index=False, encoding="utf-8-sig")
            st.cache_data.clear()

            st.success(f"{AUG_PATH}에 저장했습니다. 앱을 새로고침하면 추가 데이터가 반영됩니다.")

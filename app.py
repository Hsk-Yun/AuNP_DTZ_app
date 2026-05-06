import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from streamlit_image_coordinates import streamlit_image_coordinates


st.set_page_config(
    page_title="색으로 찾는 중금속",
    layout="wide"
)

CSV_PATH_CANDIDATES = [
    "aunp_dtz_0_40ppm_dataset.csv",
    "training_data.csv",
]

PPM_MIN = 0
PPM_MAX = 40

METAL_K = 4
PPM_K = 3

MAX_DISPLAY_WIDTH = 760


METAL_COL_CANDIDATES = [
    "heavy metal",
    "Heavy Metal",
    "Metal",
    "metal",
    "중금속",
    "금속",
    "Group",
    "group",
]

PPM_COL_CANDIDATES = [
    "ppm",
    "PPM",
    "농도",
    "concentration",
    "Concentration",
]

FEATURE_CANDIDATES = {
    "L": ["L", "L*", "Lab_L", "sample_L"],
    "a": ["a", "a*", "Lab_a", "sample_a"],
    "b": ["b", "b*", "Lab_b", "sample_b"],

    "deltaL": ["deltaL", "DeltaL", "delta_L", "Delta_L", "dL", "DL", "ΔL", "ΔL*"],
    "deltaa": ["deltaa", "Deltaa", "deltaA", "DeltaA", "delta_a", "Delta_a", "da", "DA", "Δa", "Δa*"],
    "deltab": ["deltab", "Deltab", "deltaB", "DeltaB", "delta_b", "Delta_b", "db", "DB", "Δb", "Δb*"],
    "deltaE": ["deltaE", "DeltaE", "delta_E", "Delta_E", "dE", "DE", "ED", "ΔE"],
}


def srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_xyz(rgb):
    r, g, b = srgb_to_linear(np.array(rgb, dtype=float))

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    return np.array([x, y, z]) * 100


def xyz_to_lab(xyz):
    x, y, z = xyz

    xn, yn, zn = 95.047, 100.000, 108.883
    x, y, z = x / xn, y / yn, z / zn

    def f(t):
        return np.where(t > 0.008856, t ** (1 / 3), 7.787 * t + 16 / 116)

    fx, fy, fz = f(x), f(y), f(z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.array([L, a, b], dtype=float)


def rgb_to_lab(rgb):
    return xyz_to_lab(rgb_to_xyz(rgb))


def resize_for_display(image, max_width=760):
    original_width, original_height = image.size

    if original_width <= max_width:
        return image.copy(), 1.0

    scale = max_width / original_width
    display_height = int(original_height * scale)

    display_image = image.resize(
        (max_width, display_height),
        Image.Resampling.LANCZOS
    )

    return display_image, scale


def to_display_xy(original_xy, scale):
    if original_xy is None:
        return None

    x, y = original_xy
    return int(x * scale), int(y * scale)


def to_original_xy(display_xy, scale):
    x, y = display_xy

    if scale == 0:
        return x, y

    return int(x / scale), int(y / scale)


def mean_rgb_in_circle(image, center, radius):
    img = image.convert("RGB")
    arr = np.array(img)

    x0, y0 = center
    h, w, _ = arr.shape

    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))

    yy, xx = np.ogrid[:h, :w]
    mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= radius ** 2

    if mask.sum() == 0:
        raise ValueError("선택 영역이 너무 작습니다.")

    return arr[mask].mean(axis=0)


def draw_points(display_image, blank_xy=None, sample_xy=None, radius=12):
    img = display_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    if blank_xy is not None:
        x, y = blank_xy
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline="blue",
            width=4,
        )
        draw.text((x + radius + 5, y - radius), "Blank", fill="blue")

    if sample_xy is not None:
        x, y = sample_xy
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline="red",
            width=4,
        )
        draw.text((x + radius + 5, y - radius), "Sample", fill="red")

    return img


def load_raw_image(uploaded_file):
    raw_bytes = uploaded_file.getvalue()

    st.info(".raw 파일은 크기 정보가 없어서 가로, 세로, 채널 값을 직접 맞춰야 합니다.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        width = st.number_input("가로(px)", min_value=1, value=640, step=1)

    with col2:
        height = st.number_input("세로(px)", min_value=1, value=480, step=1)

    with col3:
        channels = st.selectbox("채널", [3, 1, 4], index=0)

    with col4:
        bit_depth = st.selectbox("비트", ["8-bit", "16-bit"], index=0)

    dtype = np.uint8 if bit_depth == "8-bit" else np.dtype("<u2")
    arr = np.frombuffer(raw_bytes, dtype=dtype)

    expected_size = int(width * height * channels)

    if arr.size < expected_size:
        st.error(
            f"RAW 설정이 맞지 않습니다. 현재 데이터 수: {arr.size}, 필요 데이터 수: {expected_size}"
        )
        st.stop()

    if arr.size > expected_size:
        arr = arr[:expected_size]

    arr = arr.reshape((int(height), int(width), int(channels)))

    if bit_depth == "16-bit":
        arr = (arr / 257).clip(0, 255).astype(np.uint8)

    if channels == 1:
        arr = np.repeat(arr, 3, axis=2)

    if channels == 4:
        arr = arr[:, :, :3]

    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def load_uploaded_image(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".raw"):
        return load_raw_image(uploaded_file)

    return Image.open(uploaded_file).convert("RGB")


def load_training_data():
    last_error = None

    for path in CSV_PATH_CANDIDATES:
        try:
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="cp949")

            df.columns = [str(c).strip() for c in df.columns]
            return df, path

        except Exception as e:
            last_error = e

    raise FileNotFoundError(
        f"CSV 파일을 찾지 못했습니다. 가능한 파일명: {CSV_PATH_CANDIDATES}. 마지막 오류: {last_error}"
    )


def find_column(df, candidates):
    for cand in candidates:
        if cand in df.columns:
            return cand

    lower_map = {str(c).strip().lower(): c for c in df.columns}

    for cand in candidates:
        key = cand.strip().lower()
        if key in lower_map:
            return lower_map[key]

    return None


def resolve_feature_columns(df):
    resolved = {}

    for feature_name, candidates in FEATURE_CANDIDATES.items():
        col = find_column(df, candidates)

        if col is not None:
            resolved[feature_name] = col

    required = ["L", "a", "b", "deltaL", "deltaa", "deltab"]

    missing = [name for name in required if name not in resolved]

    if missing:
        raise ValueError(
            "CSV에서 필요한 피처 컬럼을 찾지 못했습니다: "
            + ", ".join(missing)
            + "\n필요 컬럼: L, a, b, deltaL, deltaa, deltab, deltaE"
        )

    if "deltaE" not in resolved:
        resolved["deltaE"] = None

    return resolved


def normalize_metal_name(value):
    return str(value).strip()


def is_ag(value):
    v = str(value).strip().lower()
    v = v.replace(" ", "").replace("+", "")
    return v in ["ag", "silver", "은"]


def format_ppm_label(value):
    try:
        num = float(value)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except Exception:
        return str(value)


def make_feature_dict(blank_rgb, sample_rgb):
    blank_lab = rgb_to_lab(blank_rgb)
    sample_lab = rgb_to_lab(sample_rgb)

    delta_l = sample_lab[0] - blank_lab[0]
    delta_a = sample_lab[1] - blank_lab[1]
    delta_b = sample_lab[2] - blank_lab[2]
    delta_e = float(np.sqrt(delta_l ** 2 + delta_a ** 2 + delta_b ** 2))

    feature_dict = {
        "L": sample_lab[0],
        "a": sample_lab[1],
        "b": sample_lab[2],
        "deltaL": delta_l,
        "deltaa": delta_a,
        "deltab": delta_b,
        "deltaE": delta_e,
    }

    return feature_dict, blank_lab, sample_lab


def prepare_training_data(df):
    metal_col = find_column(df, METAL_COL_CANDIDATES)
    ppm_col = find_column(df, PPM_COL_CANDIDATES)

    if metal_col is None:
        raise ValueError("CSV에서 중금속 컬럼을 찾지 못했습니다. heavy metal 또는 Metal 컬럼이 필요합니다.")

    if ppm_col is None:
        raise ValueError("CSV에서 ppm 컬럼을 찾지 못했습니다.")

    feature_map = resolve_feature_columns(df)

    work = df.copy()

    work[metal_col] = work[metal_col].apply(normalize_metal_name)
    work[ppm_col] = pd.to_numeric(work[ppm_col], errors="coerce")

    for feature_name, col in feature_map.items():
        if col is not None:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work[~work[metal_col].apply(is_ag)]
    work = work[work[ppm_col].between(PPM_MIN, PPM_MAX)]

    if feature_map["deltaE"] is None:
        work["__deltaE__"] = np.sqrt(
            work[feature_map["deltaL"]] ** 2
            + work[feature_map["deltaa"]] ** 2
            + work[feature_map["deltab"]] ** 2
        )
        feature_map["deltaE"] = "__deltaE__"

    feature_names = ["L", "a", "b", "deltaL", "deltaa", "deltab", "deltaE"]
    feature_cols = [feature_map[name] for name in feature_names]

    work = work.dropna(subset=[metal_col, ppm_col] + feature_cols)

    if len(work) == 0:
        raise ValueError("학습 가능한 데이터가 없습니다. Ag 제외 또는 ppm 0~40 조건을 확인해 주세요.")

    metal_count = work[metal_col].nunique()

    if metal_count != 9:
        st.warning(f"Ag 제외 후 인식된 중금속 수가 {metal_count}종입니다. CSV의 중금속 이름을 확인해 주세요.")

    return work, metal_col, ppm_col, feature_names, feature_cols


def train_metal_model(work, metal_col, ppm_col, feature_cols):
    metal_train = work[work[ppm_col] > 0].copy()

    if metal_train[metal_col].nunique() < 2:
        metal_train = work.copy()

    x = metal_train[feature_cols].values
    y = metal_train[metal_col].astype(str).values

    k = min(METAL_K, len(metal_train))

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",
            metric="euclidean",
        ),
    )

    model.fit(x, y)
    return model


def predict_ppm(work, metal_col, ppm_col, feature_cols, predicted_metal, x_input):
    metal_rows = work[work[metal_col].astype(str) == str(predicted_metal)].copy()

    if len(metal_rows) == 0:
        return None

    x = metal_rows[feature_cols].values
    y = metal_rows[ppm_col].apply(format_ppm_label).values

    k = min(PPM_K, len(metal_rows))

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",
            metric="euclidean",
        ),
    )

    model.fit(x, y)
    return model.predict(x_input)[0]


st.title("색으로 찾는 중금속")

try:
    raw_df, loaded_csv_path = load_training_data()
    work, metal_col, ppm_col, feature_names, feature_cols = prepare_training_data(raw_df)
    metal_model = train_metal_model(work, metal_col, ppm_col, feature_cols)
except Exception as e:
    st.error(f"학습 데이터 오류: {e}")
    st.stop()


if "blank_xy" not in st.session_state:
    st.session_state.blank_xy = None

if "sample_xy" not in st.session_state:
    st.session_state.sample_xy = None


top_col1, top_col2 = st.columns([1.2, 1])

with top_col1:
    input_method = st.radio(
        "이미지 입력",
        ["사진 업로드", "카메라 촬영"],
        horizontal=True,
    )

with top_col2:
    radius = st.slider(
        "평균 색상 추출 영역",
        min_value=4,
        max_value=80,
        value=12,
        step=1,
    )


uploaded_file = None

if input_method == "사진 업로드":
    uploaded_file = st.file_uploader(
        "이미지를 올려주세요",
        type=["jpg", "jpeg", "png", "raw"],
    )
else:
    uploaded_file = st.camera_input("촬영하기")


if uploaded_file is None:
    st.stop()


try:
    image = load_uploaded_image(uploaded_file)
except Exception as e:
    st.error(f"이미지를 불러오지 못했습니다: {e}")
    st.stop()


display_image, scale = resize_for_display(image, MAX_DISPLAY_WIDTH)

display_blank_xy = to_display_xy(st.session_state.blank_xy, scale)
display_sample_xy = to_display_xy(st.session_state.sample_xy, scale)
display_radius = max(3, int(radius * scale))

image_col, control_col = st.columns([2, 1])

with control_col:
    select_target = st.radio(
        "선택할 위치",
        ["Blank", "Sample"],
        horizontal=True,
    )

    st.write("Blank:", st.session_state.blank_xy)
    st.write("Sample:", st.session_state.sample_xy)

    can_analyze = (
        st.session_state.blank_xy is not None
        and st.session_state.sample_xy is not None
    )

    run_clicked = st.button(
        "영역 확정 및 실행",
        disabled=not can_analyze,
        use_container_width=True,
    )

    reset_clicked = st.button(
        "선택 초기화",
        use_container_width=True,
    )

    if reset_clicked:
        st.session_state.blank_xy = None
        st.session_state.sample_xy = None
        st.rerun()


with image_col:
    marked_image = draw_points(
        display_image,
        blank_xy=display_blank_xy,
        sample_xy=display_sample_xy,
        radius=display_radius,
    )

    clicked = streamlit_image_coordinates(
        marked_image,
        key="image_click",
    )


if clicked is not None:
    display_x = int(clicked["x"])
    display_y = int(clicked["y"])
    original_x, original_y = to_original_xy((display_x, display_y), scale)

    if select_target == "Blank":
        st.session_state.blank_xy = (original_x, original_y)
    else:
        st.session_state.sample_xy = (original_x, original_y)

    st.rerun()


if run_clicked:
    try:
        blank_rgb = mean_rgb_in_circle(
            image,
            st.session_state.blank_xy,
            radius,
        )

        sample_rgb = mean_rgb_in_circle(
            image,
            st.session_state.sample_xy,
            radius,
        )

        feature_dict, blank_lab, sample_lab = make_feature_dict(
            blank_rgb,
            sample_rgb,
        )

        x_input = np.array(
            [feature_dict[name] for name in feature_names],
            dtype=float,
        ).reshape(1, -1)

        predicted_metal = metal_model.predict(x_input)[0]

        predicted_ppm = predict_ppm(
            work,
            metal_col,
            ppm_col,
            feature_cols,
            predicted_metal,
            x_input,
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric("중금속", predicted_metal)

        with result_col2:
            st.metric("농도", f"{predicted_ppm} ppm" if predicted_ppm is not None else "분류 불가")

        with result_col3:
            st.metric("deltaE", f"{feature_dict['deltaE']:.3f}")

        if str(predicted_ppm) == "0":
            st.info("0 ppm은 Blank에 가까운 상태라 중금속 종류 예측 신뢰도가 낮을 수 있습니다.")

        feature_result = pd.DataFrame(
            {
                "Feature": feature_names,
                "Value": [round(float(feature_dict[name]), 4) for name in feature_names],
            }
        )

        st.dataframe(feature_result, use_container_width=True)

        rgb_lab_result = pd.DataFrame(
            {
                "구분": ["Blank", "Sample"],
                "R": [round(float(blank_rgb[0]), 2), round(float(sample_rgb[0]), 2)],
                "G": [round(float(blank_rgb[1]), 2), round(float(sample_rgb[1]), 2)],
                "B": [round(float(blank_rgb[2]), 2), round(float(sample_rgb[2]), 2)],
                "L": [round(float(blank_lab[0]), 3), round(float(sample_lab[0]), 3)],
                "a": [round(float(blank_lab[1]), 3), round(float(sample_lab[1]), 3)],
                "b": [round(float(blank_lab[2]), 3), round(float(sample_lab[2]), 3)],
            }
        )

        st.dataframe(rgb_lab_result, use_container_width=True)

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------------------------------
# Page config: mở rộng trang
# --------------------------------------------------------------------------------
st.set_page_config(page_title="EWMA Simulation", layout="wide")

# --------------------------------------------------------------------------------
# Hàm chia dữ liệu thành các khối (block) và tính trung bình mỗi khối
# --------------------------------------------------------------------------------
def chunk_data_and_average(data_array, block_size):
    """
    Chia data_array thành từng khối (mỗi khối có block_size điểm),
    rồi tính trung bình cho mỗi khối. Trả về mảng 'block_means'.
    """
    block_means = []
    n = len(data_array)
    for i in range(0, n, block_size):
        chunk = data_array[i : i + block_size]
        avg = np.mean(chunk)
        block_means.append(avg)
    return np.array(block_means)

# --------------------------------------------------------------------------------
# Hàm tính EWMA
# --------------------------------------------------------------------------------
def calculate_ewma(data_array, lam):
    """
    Tính EWMA cho data_array (1D) với trọng số lam (λ).
    data_array: np.array
    lam: float (0 < lam <= 1)
    """
    ewma_vals = [data_array[0]]
    for i in range(1, len(data_array)):
        ewma_val = lam * data_array[i] + (1 - lam) * ewma_vals[-1]
        ewma_vals.append(ewma_val)
    return np.array(ewma_vals)

# --------------------------------------------------------------------------------
# Streamlit App
# --------------------------------------------------------------------------------
st.title("Công cụ mô phỏng AoN sử dụng training") # <--- ĐÃ SỬA TIÊU ĐỀ PHẦN MỀM

st.header("Upload Your Data")
uploaded_file = st.file_uploader("Upload your .xlsx or .csv file", type=["xlsx", "csv"])

if uploaded_file:
    # Đọc file
    if uploaded_file.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        data = pd.read_csv(uploaded_file)

    # Chọn cột để phân tích
    selected_column = st.selectbox("Select the column to analyze", data.columns)
    values = data[selected_column].dropna().values

    # 1) TÍNH PERCENTILES (NHƯNG KHÔNG HIỂN THỊ)
    percentiles = {
        "Percentile Ranges": ["1-99", "2-98", "3-97", "4-96", "5-95"],
        "Value Ranges": [
            np.percentile(values, 99),
            np.percentile(values, 98),
            np.percentile(values, 97),
            np.percentile(values, 96),
            np.percentile(values, 95),
        ]
    }

    # 2) HIỂN THỊ BIỂU ĐỒ PHÂN PHỐI GỐC + TRUNCATION
    st.subheader("Histogram Comparison: Original Data vs. Truncated Data")

    # Tính sẵn giá trị 5th percentile và 95th percentile làm default
    default_lower_percentile = 5
    default_upper_percentile = 95
    default_lower_value = float(np.percentile(values, default_lower_percentile))
    default_upper_value = float(np.percentile(values, default_upper_percentile))

    # Thay đổi từ selectbox thành checkbox
    use_percentile_input = st.checkbox("Use Percentile Input for Truncation Limits") # <--- SỬA ĐỔI TRUNCATION INPUT TYPE

    if use_percentile_input: # <--- ĐIỀU KIỆN HIỂN THỊ INPUT PERCENTILE HOẶC VALUE
        lower_percentile_input = st.number_input(
            "Lower Truncation Percentile",
            min_value=0.0,
            max_value=100.0,
            value=float(default_lower_percentile),
            step=0.5
        )
        upper_percentile_input = st.number_input(
            "Upper Truncation Percentile",
            min_value=0.0,
            max_value=100.0,
            value=float(default_upper_percentile),
            step=0.5
        )
        lower_limit = float(np.percentile(values, lower_percentile_input))
        upper_limit = float(np.percentile(values, upper_percentile_input))
    else:
        # Initialize session state for lower and upper limits if not already present
        if 'lower_limit_value' not in st.session_state:
            st.session_state['lower_limit_value'] = default_lower_value
        if 'upper_limit_value' not in st.session_state:
            st.session_state['upper_limit_value'] = default_upper_value
        if 'lower_limit_percentile' not in st.session_state:
            st.session_state['lower_limit_percentile'] = default_lower_percentile # Placeholder, calculate if needed
        if 'upper_limit_percentile' not in st.session_state:
            st.session_state['upper_limit_percentile'] = default_upper_percentile # Placeholder, calculate if needed


        def get_percentile_label(value, original_values):
            percentile_rank = np.sum(original_values <= value) / len(original_values) * 100
            return f"{value:.2f} ({percentile_rank:.0f} percentile)"

        def calculate_percentile_value(value, original_values):
            return np.sum(original_values <= value) / len(original_values) * 100

        lower_limit_value = st.number_input(
            "Lower Truncation Limit",
            min_value=float(values.min()),
            max_value=float(values.max()),
            value=st.session_state['lower_limit_value'],
            on_change=None, # Remove on_change to handle enter directly
        )

        upper_limit_value = st.number_input(
            "Upper Truncation Limit",
            min_value=float(values.min()),
            max_value=float(values.max()),
            value=st.session_state['upper_limit_value'],
            on_change=None, # Remove on_change to handle enter directly
        )

        if lower_limit_value != st.session_state['lower_limit_value']:
            # Ensure the new value is not above max_value before updating session state
            if lower_limit_value <= float(values.max()):
                st.session_state['lower_limit_value'] = lower_limit_value
                st.session_state['lower_limit_percentile'] = calculate_percentile_value(lower_limit_value, values)
            else:
                st.warning(f"Lower limit value cannot exceed the maximum value of the data: {values.max():.2f}")
                lower_limit_value = st.session_state['lower_limit_value'] # Revert to the valid session state value

        if upper_limit_value != st.session_state['upper_limit_value']:
            # Ensure the new value is not above max_value before updating session state
            if upper_limit_value <= float(values.max()): # While logically it should not exceed max, adding check for robustness
                st.session_state['upper_limit_value'] = upper_limit_value
                st.session_state['upper_limit_percentile'] = calculate_percentile_value(upper_limit_value, values)
            else:
                st.warning(f"Upper limit value cannot exceed the maximum value of the data: {values.max():.2f}")
                upper_limit_value = st.session_state['upper_limit_value'] # Revert to the valid session state value


        lower_limit_label = "Lower Truncation Limit"
        if 'lower_limit_percentile' in st.session_state:
            lower_limit_label = f"Lower Truncation Limit: {st.session_state['lower_limit_value']:.2f} ({st.session_state['lower_limit_percentile']:.0f} percentile)"

        upper_limit_label = "Upper Truncation Limit"
        if 'upper_limit_percentile' in st.session_state:
            upper_limit_label = f"Upper Truncation Limit: {st.session_state['upper_limit_value']:.2f} ({st.session_state['upper_limit_percentile']:.0f} percentile)"


        lower_limit = st.session_state['lower_limit_value']
        upper_limit = st.session_state['upper_limit_value']


    # Tạo hai histogram: Original và Truncated
    # Histogram 1: Dữ liệu gốc (Original)
    fig_original = px.histogram(
        x=values,
        nbins=50,
        title="Original Data",
        labels={"x": "Values", "y": "Count"},
        template="plotly_dark"
    )

    # Histogram 2: Dữ liệu đã truncate
    truncated_values = values[(values >= lower_limit) & (values <= upper_limit)]
    fig_truncated = px.histogram(
        x=truncated_values,
        nbins=50,
        title="Truncated Data",
        labels={"x": "Values", "y": "Count"},
        template="plotly_dark"
    )
    # Thêm vline cho histogram truncated
    fig_truncated.add_vline(
        x=lower_limit,
        line_dash="dash",
        line_color="red",
        annotation_text="Lower Limit",
        annotation_position="bottom right"
    )
    fig_truncated.add_vline(
        x=upper_limit,
        line_dash="dash",
        line_color="green",
        annotation_text="Upper Limit",
        annotation_position="top left"
    )

    # Dùng layout cột để đặt 2 biểu đồ cạnh nhau
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_original, use_container_width=True)
    with col2:
        st.plotly_chart(fig_truncated, use_container_width=True)

    #

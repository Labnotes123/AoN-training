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
st.title("Công cụ mô phỏng AoN sử dụng training")

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
    percentiles_data = np.arange(0, 101, 0.5)  # Tính percentiles với độ chi tiết 0.5
    percentile_values_lookup = {p: np.percentile(values, p) for p in percentiles_data} # Tạo lookup dictionary

    # 2) HIỂN THỊ BIỂU ĐỒ PHÂN PHỐI GỐC + TRUNCATION
    st.subheader("Histogram Comparison: Original Data vs. Truncated Data")

    # Tính sẵn giá trị 5th percentile và 95th percentile làm default
    default_lower_percentile = 5
    default_upper_percentile = 95
    default_lower_value = percentile_values_lookup[default_lower_percentile] # Lấy giá trị từ lookup
    default_upper_value = percentile_values_lookup[default_upper_percentile] # Lấy giá trị từ lookup

    # Sử dụng cột để bố trí các ô nhập liệu cạnh nhau
    col_lower, col_upper = st.columns(2)

    with col_lower:
        lower_limit_value = st.number_input( # Ô nhập giá trị Lower Limit
            "Lower Truncation Limit (Value)",
            min_value=float(values.min()),
            max_value=float(values.max()),
            value=default_lower_value
        )
        lower_limit_percentile_str = st.text_input( # Ô hiển thị/nhập percentile Lower Limit
            "Lower Truncation Limit (Percentile)",
            value=f"{default_lower_percentile:.2f}", # Hiển thị percentile mặc định
        )

    with col_upper:
        upper_limit_value = st.number_input( # Ô nhập giá trị Upper Limit
            "Upper Truncation Limit (Value)",
            min_value=float(values.min()),
            max_value=float(values.max()),
            value=default_upper_value
        )
        upper_limit_percentile_str = st.text_input( # Ô hiển thị/nhập percentile Upper Limit
            "Upper Truncation Limit (Percentile)",
            value=f"{default_upper_percentile:.2f}", # Hiển thị percentile mặc định
        )

    # Cập nhật lower_limit và upper_limit từ giá trị người dùng nhập (ban đầu là giá trị)
    lower_limit = lower_limit_value
    upper_limit = upper_limit_value

    # **Xử lý khi giá trị Percentile thay đổi**: (Cần chuyển đổi từ percentile sang value nếu người dùng nhập trực tiếp percentile)
    try:
        lower_percentile_input_val = float(lower_limit_percentile_str)
        if 0 <= lower_percentile_input_val <= 100: # Kiểm tra percentile hợp lệ
            lower_limit = percentile_values_lookup[lower_percentile_input_val] # Lấy value từ lookup
            lower_limit_value = lower_limit # Cập nhật ô giá trị để đồng bộ
    except ValueError:
        st.warning("Please enter a valid number for Lower Percentile.") # Xử lý nếu percentile nhập không hợp lệ

    try:
        upper_percentile_input_val = float(upper_limit_percentile_str)
        if 0 <= upper_percentile_input_val <= 100: # Kiểm tra percentile hợp lệ
            upper_limit = percentile_values_lookup[upper_percentile_input_val] # Lấy value từ lookup
            upper_limit_value = upper_limit # Cập nhật ô giá trị để đồng bộ
    except ValueError:
        st.warning("Please enter a valid number for Upper Percentile.") # Xử lý nếu percentile nhập không hợp lệ

    # **Xử lý khi giá trị Value thay đổi**: (Cần tính percentile tương ứng và cập nhật ô percentile)
    # Hàm tìm percentile gần nhất với value (sử dụng lookup table)
    def find_approximate_percentile(value, percentile_value_map):
        best_percentile = 0
        min_diff = float('inf')
        for percentile, val in percentile_value_map.items():
            diff = abs(val - value)
            if diff < min_diff:
                min_diff = diff
                best_percentile = percentile
        return best_percentile

    lower_limit_percentile_display = find_approximate_percentile(lower_limit_value, percentile_values_lookup)
    upper_limit_percentile_display = find_approximate_percentile(upper_limit_value, percentile_values_lookup)

    # Cập nhật ô percentile hiển thị
    with col_lower:
        lower_limit_percentile_str = st.text_input(
            "Lower Truncation Limit (Percentile)",
            value=f"{lower_limit_percentile_display:.2f}", # Hiển thị percentile đã tính toán
        )
    with col_upper:
        upper_limit_percentile_str = st.text_input(
            "Upper Truncation Limit (Percentile)",
            value=f"{upper_limit_percentile_display:.2f}", # Hiển thị percentile đã tính toán
        )


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

    # 3) TÍNH LẠI MEAN VÀ SD DỰA TRÊN DỮ LIỆU ĐÃ TRUNCATE + CHO PHÉP USER TÙY CHỈNH
    st.subheader("Enter Custom Mean/Target and Standard Deviation (after truncation)")

    # Tự động tính dựa trên truncated_values
    default_mean = float(np.mean(truncated_values))
    default_std = float(np.std(truncated_values))

    # Người dùng có thể điều chỉnh
    mean_custom = st.number_input("Mean (Target)", value=default_mean)
    std_dev_custom = st.number_input("Standard Deviation", value=default_std)

    # Tính sample std (ddof=1) để hiển thị
    sample_std = np.std(truncated_values, ddof=1)
    st.write(f"**Sample Standard Deviation (ddof=1) from truncated data**: {sample_std:.4f}")

    # 4) TÙY CHỌN VẼ EWMA
    st.sidebar.header("EWMA Approach")
    ewma_option = st.sidebar.radio("Choose EWMA Approach", ("Block-based", "Lambda-based"))

    block_size = st.sidebar.number_input("Block Size", min_value=1, value=100)
    lambda_factor = st.sidebar.slider("Weighting Factor (λ)", 0.01, 1.0, 0.1, 0.01)

    st.sidebar.header("Control Limit Approach")
    approach = st.sidebar.radio("Choose Control Limit Approach", ("Manual", "Confidence Interval"))

    # Dùng mean_custom, std_dev_custom để tính UCL, LCL tùy thuộc option
    if approach == "Manual":
        ucl_input = st.sidebar.number_input("Upper Control Limit (UCL)",
                 value=mean_custom + 2.0 * std_dev_custom)
        lcl_input = st.sidebar.number_input("Lower Control Limit (LCL)",
                 value=mean_custom - 2.0 * std_dev_custom)
        custom_ucl = ucl_input
        custom_lcl = lcl_input
        z_value = None
    else:
        z_value = st.sidebar.slider("Z Value", 1.0, 5.0, 3.0, 0.1)
        custom_ucl = mean_custom + z_value * std_dev_custom
        custom_lcl = mean_custom - z_value * std_dev_custom

    # --------------------------------------------------------------------------------
    # (5) TẠO DỮ LIỆU THEO EWMA OPTION
    # --------------------------------------------------------------------------------
    if ewma_option == "Block-based":
        data_for_ewma = chunk_data_and_average(truncated_values, block_size)
        x_label = "Block Index"
    else:
        data_for_ewma = truncated_values
        x_label = "Data Point"

    ewma_values = calculate_ewma(data_for_ewma, lambda_factor)

    # Xác định Out-of-control
    out_of_control_points = []
    for i, val in enumerate(ewma_values):
        if val > custom_ucl or val < custom_lcl:
            out_of_control_points.append((i, val))

    # --------------------------------------------------------------------------------
    # (6) VẼ BIỂU ĐỒ EWMA
    # --------------------------------------------------------------------------------
    fig_ewma = go.Figure()

    fig_ewma.add_trace(go.Scatter(
        x=np.arange(len(ewma_values)),
        y=ewma_values,
        mode='lines',
        name='EWMA',
        line=dict(color='blue')
    ))
    if out_of_control_points:
        idx_ooc, val_ooc = zip(*out_of_control_points)
        fig_ewma.add_trace(go.Scatter(
            x=idx_ooc,
            y=val_ooc,
            mode='markers',
            name='Out of Control',
            marker=dict(color='red', size=8)
        ))

    fig_ewma.add_hline(y=mean_custom, line_dash="dot", line_color="yellow", name="Mean")
    fig_ewma.add_hline(y=custom_ucl, line_dash="dash", line_color="red", name="UCL")
    fig_ewma.add_hline(y=custom_lcl, line_dash="dash", line_color="blue", name="LCL")

    title_chart = "EWMA Chart - " + ewma_option
    fig_ewma.update_layout(
        title=title_chart,
        xaxis_title=x_label,
        yaxis_title="EWMA Value",
        template="plotly_dark",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02)
    )

    # Annotation
    annotation_list = [
        f"Mean: {mean_custom:.2f}",
        f"Block Size: {block_size}",
        f"λ: {lambda_factor}",
        f"UCL: {custom_ucl:.2f}",
        f"LCL: {custom_lcl:.2f}",
    ]
    if approach == "Confidence Interval" and z_value is not None:
        annotation_list.append(f"z: {z_value}")

    annotation_text = "<br>".join(annotation_list)
    fig_ewma.add_annotation(
        x=1.02,
        y=0.85,
        xref="paper",
        yref="paper",
        text=annotation_text,
        showarrow=False,
        align="left",
        xanchor="left",
        yanchor="top",
        font=dict(size=12, color="white")
    )

    # Hiển thị chart
    st.plotly_chart(fig_ewma, use_container_width=True)

    # --------------------------------------------------------------------------------
    # 6.5) HIỂN THỊ BẢNG THÔNG TIN THAM SỐ
    # --------------------------------------------------------------------------------
    param_data = {
        "Mean (Custom)": [mean_custom],
        "Block Size": [block_size],
        "Weighting Factor (λ)": [lambda_factor],
        "UCL": [custom_ucl],
        "LCL": [custom_lcl],
        "Z Value": [z_value if (approach == "Confidence Interval") else None]
    }
    df_param = pd.DataFrame(param_data)

    st.subheader("EWMA Parameter Table")
    st.table(df_param)

    # --------------------------------------------------------------------------------
    # (7) HIỂN THỊ OUT-OF-CONTROL
    # --------------------------------------------------------------------------------
    st.metric(label="Total Out-of-Control Points", value=len(out_of_control_points))

    show_table = st.checkbox("Show Out-of-Control Points", value=False)
    if show_table:
        if out_of_control_points:
            st.subheader("Out-of-Control Points")
            out_of_control_df = pd.DataFrame(out_of_control_points, columns=[x_label, "EWMA Value"])
            st.table(out_of_control_df)
        else:
            st.info("No out-of-control points detected.")

else:
    st.warning("Please upload a file to proceed.")

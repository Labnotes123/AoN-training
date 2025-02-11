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

    # **ÉP KIỂU DỮ LIỆU SANG NUMERIC VÀ LOẠI BỎ NAN MỘT LẦN NỮA**
    values = pd.to_numeric(values, errors='coerce').dropna().values

    if values.size > 0:
        # 1) TÍNH PERCENTILES (NHƯNG KHÔNG HIỂN THỊ)
        percentiles_data = np.arange(0, 101, 0.5)
        try: # **THÊM TRY-EXCEPT BLOCK ĐỀ PHÒNG LỖI PERCENTILE**
            percentile_values_lookup = {p: np.percentile(values, p) for p in percentiles_data} # Tạo lookup dictionary
        except ValueError as e:
            st.error(f"Error calculating percentiles: {e}") # Hiển thị thông báo lỗi chi tiết
            st.stop() # Dừng ứng dụng nếu không tính được percentile
            percentile_values_lookup = {} # Gán giá trị rỗng để tránh lỗi ở các phần code sau

        # 2) HIỂN THỊ BIỂU ĐỒ PHÂN PHỐI GỐC + TRUNCATION
        st.subheader("Histogram Comparison: Original Data vs. Truncated Data")

        # Tính sẵn giá trị 5th percentile và 95th percentile làm default
        default_lower_percentile = 5
        default_upper_percentile = 95
        default_lower_value = percentile_values_lookup.get(default_lower_percentile, np.nan) # Sử dụng .get() để tránh KeyError
        default_upper_value = percentile_values_lookup.get(default_upper_percentile, np.nan) # Sử dụng .get() để tránh KeyError

        # Sử dụng cột để bố trí các ô nhập liệu cạnh nhau
        col_lower, col_upper = st.columns(2)

        with col_lower:
            lower_limit_value = st.number_input( # Ô nhập giá trị Lower Limit
                "Lower Truncation Limit (Value)",
                min_value=float(values.min()),
                max_value=float(values.max()),
                value=default_lower_value if not np.isnan(default_lower_value) else float(values.min()) # Xử lý NaN default value
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
                value=default_upper_value if not np.isnan(default_upper_value) else float(values.max()) # Xử lý NaN default value
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
            if 0 <= lower_percentile_input_val <= 100: # **KIỂM TRA PERCENTILE NHẬP VÀO CÓ HỢP LỆ KHÔNG**
                lower_limit = percentile_values_lookup.get(lower_percentile_input_val, np.nan) # Lấy value từ lookup, xử lý KeyError
                if np.isnan(lower_limit): # Xử lý nếu không tìm thấy percentile trong lookup (do lỗi tính percentile ban đầu)
                    lower_limit = default_lower_value if not np.isnan(default_lower_value) else float(values.min())
                    st.warning("Percentile value not found in lookup, using default value.")
                lower_limit_value = lower_limit # Cập nhật ô giá trị để đồng bộ
            else:
                st.warning("Lower Percentile must be between 0 and 100.") # Cảnh báo nếu percentile không hợp lệ
                lower_percentile_input_val = default_lower_percentile # Reset về giá trị default
                lower_limit = default_lower_value if not np.isnan(default_lower_value) else float(values.min())
                lower_limit_value = default_lower_value # Reset ô giá trị về default
                lower_limit_percentile_str = f"{default_lower_percentile:.2f}" # Reset ô percentile về default

        except ValueError:
            st.warning("Please enter a valid number for Lower Percentile.") # Xử lý nếu percentile nhập không hợp lệ
            lower_percentile_input_val = default_lower_percentile # Reset về default
            lower_limit = default_lower_value if not np.isnan(default_lower_value) else float(values.min())
            lower_limit_value = default_lower_value # Reset ô giá trị về default
            lower_limit_percentile_str = f"{default_lower_percentile:.2f}" # Reset ô percentile về default


        try:
            upper_percentile_input_val = float(upper_limit_percentile_str)
            if 0 <= upper_percentile_input_val <= 100: # **KIỂM TRA PERCENTILE NHẬP VÀO CÓ HỢP LỆ KHÔNG**
                upper_limit = percentile_values_lookup.get(upper_percentile_input_val, np.nan) # Lấy value từ lookup, xử lý KeyError
                if np.isnan(upper_limit): # Xử lý nếu không tìm thấy percentile trong lookup
                    upper_limit = default_upper_value if not np.isnan(default_upper_value) else float(values.max())
                    st.warning("Percentile value not found in lookup, using default value.")
                upper_limit_value = upper_limit # Cập nhật ô giá trị để đồng bộ
            else:
                st.warning("Upper Percentile must be between 0 and 100.") # Cảnh báo nếu percentile không hợp lệ
                upper_percentile_input_val = default_upper_percentile # Reset về giá trị default
                upper_limit = default_upper_value if not np.isnan(default_upper_value) else float(values.max())
                upper_limit_value = default_upper_value # Reset ô giá trị về default
                upper_limit_percentile_str = f"{default_upper_percentile:.2f}" # Reset ô percentile về default

        except ValueError:
            st.warning("Please enter a valid number for Upper Percentile.") # Xử lý nếu percentile nhập không hợp lệ
            upper_percentile_input_val = default_upper_percentile # Reset về default
            upper_limit = default_upper_value if not np.isnan(default_upper_value) else float(values.max())
            upper_limit_value = default_upper_value # Reset ô giá trị về default
            upper_limit_percentile_str = f"{default_upper_percentile:.2f}" # Reset ô percentile về default


        # **Xử lý khi giá trị

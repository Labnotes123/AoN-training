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
 st.title("MA Bias Detection & Exponentially Weighted Moving Average (EWMA) Simulation")

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

  truncation_input_type = st.selectbox("Truncation Limit Input Type", ["Value", "Percentile"])

  if truncation_input_type == "Value":
   lower_limit = st.number_input(
    "Lower Truncation Limit",
    min_value=float(values.min()),
    max_value=float(values.max()),
    value=default_lower_value
   )
   upper_limit = st.number_input(
    "Upper Truncation

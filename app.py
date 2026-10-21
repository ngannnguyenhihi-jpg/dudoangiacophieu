import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import date

# FIX SEED (ổn định kết quả) theo đúng mã nguồn gốc
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Cấu hình trang Streamlit
st.set_page_config(page_title="Dự Báo Giá Cổ Phiếu LSTM", layout="wide")

st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu bằng AI (LSTM)")
st.write("Mô hình AI dự đoán giá đóng cửa của cổ phiếu dựa trên dữ liệu lịch sử.")

# --- KHU VỰC NHẬP LIỆU ---
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Nhập mã chứng khoán (VD: VNM.VN, FPT.VN, AAPL, TSLA):", "VNM.VN")
with col2:
    predict_days = st.slider("Số ngày muốn dự báo trong tương lai:", min_value=1, max_value=10, value=5)

if st.button("🚀 Bắt đầu Dự Báo", type="primary"):
    # Sử dụng spinner trong lúc tải dữ liệu và train mô hình
    with st.spinner('Đang tải dữ liệu và huấn luyện AI (Quá trình này có thể mất 1-2 phút)...'):
        
        # 1. Tải dữ liệu từ Yahoo Finance
        try:
            # Lấy dữ liệu từ 2018 đến ngày hiện tại
            df = yf.download(ticker, start="2018-01-01", end=date.today().strftime("%Y-%m-%d"))
            if df.empty:
                st.error("❌ Không tìm thấy dữ liệu cho mã chứng khoán này. Vui lòng kiểm tra lại mã.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Lỗi tải dữ liệu: {e}")
            st.stop()

        # Xử lý format dữ liệu của yfinance phiên bản mới
        if isinstance(df.columns, pd.MultiIndex):
            data = df['Close'][ticker].values.reshape(-1, 1)
        else:
            data = df[['Close']].values

        if len(data) < 100:
            st.error("❌ Dữ liệu quá ngắn (dưới 100 ngày) để AI có thể học. Hãy chọn mã khác.")
            st.stop()

        current_price = float(data[-1][0])
        st.info(f"**Giá đóng cửa gần nhất (Ngày cập nhật cuối):** {current_price:,.2f}")

        # 2. Tiền xử lý dữ liệu (Áp dụng theo source code gốc)
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Hàm tạo chuỗi thời gian window_size = 60
        def create_sequences(data_seq, window_size=60):
            X, y = [], []
            for i in range(window_size, len(data_seq)):
                X.append(data_seq[i-window_size:i])
              y.append(data_seq[i])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_scaled)
        test_combined = np.vstack((train_scaled[-60:], test_scaled))
        X_test, y_test = create_sequences(test_combined)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Tách tập Validation (10%)
        val_size = int(0.1 * len(X_train))
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]

        # 3. Xây dựng và huấn luyện mô hình LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Dừng sớm nếu mô hình không cải thiện
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Train mô hình (ẩn log bằng verbose=0)
        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            shuffle=False,   
            callbacks=[early_stop],
            verbose=0 
        )

        # 4. Dự báo tương lai (predict_days ngày)
        last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
        last_60_days = last_60_days.reshape(1, 60, 1)
        future_predictions = []

        for i in range(predict_days):
            next_pred = model.predict(last_60_days, verbose=0)
            future_predictions.append(next_pred[0,0])
            last_60_days = np.append(
                last_60_days[:,1:,:],
                next_pred.reshape(1,1,1),
                axis=1
            )

        # Giải chuẩn hóa dữ liệu về giá trị thực
        future_predictions = np.array(future_predictions).reshape(-1,1)
        future_predictions = scaler.inverse_transform(future_predictions)

        # 5. Tính toán tỷ lệ Lãi / Lỗ (%)
        final_predicted_price = float(future_predictions[-1][0])
        profit_loss_value = final_predicted_price - current_price
        profit_loss_pct = (profit_loss_value / current_price) * 100

        st.markdown("---")
        st.subheader(f"📊 Kết quả dự báo cho {predict_days} ngày tới")
        
        # Hiển thị Metric Lãi/Lỗ
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(label=f"Giá dự báo ở ngày thứ {predict_days}", value=f"{final_predicted_price:,.2f}")
        with metric_col2:
            if profit_loss_pct > 0:
              st.metric(label="Dự báo Lãi/Lỗ (%)", value=f"+{profit_loss_pct:.2f}%", delta=f"+{profit_loss_value:,.2f}")
            elif profit_loss_pct < 0:
                st.metric(label="Dự báo Lãi/Lỗ (%)", value=f"{profit_loss_pct:.2f}%", delta=f"{profit_loss_value:,.2f}")
            else:
                st.metric(label="Dự báo Lãi/Lỗ (%)", value="0.00%", delta="Hòa vốn")

        # 6. Trực quan hóa dữ liệu (Biểu đồ)
        st.markdown("### 📈 Biểu đồ biến động giá")
        
        # Lấy 100 ngày gần nhất để vẽ cho dễ nhìn
        show_history = 100 
        historical_plot_data = data[-show_history:]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Tạo trục x cho quá khứ và tương lai
        history_x = np.arange(len(historical_plot_data))
        future_x = np.arange(len(historical_plot_data) - 1, len(historical_plot_data) + predict_days)
        
        # Điểm mỏ neo để nối đường biểu đồ
        future_plot_data = np.insert(future_predictions.flatten(), 0, historical_plot_data[-1][0])
        
        ax.plot(history_x, historical_plot_data, label="Giá thực tế (100 ngày qua)", color='#1f77b4', linewidth=2)
        ax.plot(future_x, future_plot_data, marker='o', label=f"Dự báo {predict_days} ngày tới", color='#ff7f0e', linewidth=2, markersize=6)
        
        ax.set_title(f"Lịch sử và Dự báo giá cổ phiếu {ticker.upper()}", fontsize=14)
        ax.set_xlabel("Số ngày", fontsize=12)
        ax.set_ylabel("Mức Giá", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        st.pyplot(fig)

        # 7. Hiển thị bảng chi tiết các ngày dự báo
        st.markdown("### 🗓️ Chi tiết giá từng ngày")
        
        # Bỏ qua ngày cuối tuần
        future_dates = pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=predict_days, freq='B')
        
        df_pred = pd.DataFrame({
            "Ngày dự báo (Dự kiến)": future_dates.strftime("%Y-%m-%d"),
            "Giá dự báo": future_predictions.flatten()
        })
        
        # Format lại để số hiển thị đẹp hơn
        df_pred['Giá dự báo'] = df_pred['Giá dự báo'].map("{:,.2f}".format)
        
        st.table(df_pred)

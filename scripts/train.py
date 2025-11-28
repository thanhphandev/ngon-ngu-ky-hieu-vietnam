import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC # Support Vector Machine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Vẽ và lưu ma trận nhầm lẫn"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Ma trận nhầm lẫn)')
    plt.ylabel('Nhãn thực tế (True Label)')
    plt.xlabel('Nhãn dự đoán (Predicted Label)')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📊 Đã lưu biểu đồ Confusion Matrix tại: {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training model")

    parser.add_argument("--model_name", help="Name of the model",
                        type=str, default="model")
    parser.add_argument("--dir", help="Location of the model",
                        type=str, default="models")
    args = parser.parse_args()

    print("=" * 80)
    print(f"🧠 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH: {args.model_name}")
    print("=" * 80)

    start_time = datetime.now()
    X, y, mapping = [], [], dict()

    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ Lỗi: Không tìm thấy thư mục '{data_dir}'.")
        exit(1)

    pose_files = list(os.scandir(data_dir))
    
    # Lọc chỉ lấy file .npy
    pose_files = [f for f in pose_files if f.name.endswith('.npy')]

    if not pose_files:
        print(f"❌ Lỗi: Không tìm thấy dữ liệu .npy nào trong '{data_dir}'.")
        exit(1)

    print(f"📂 Tìm thấy {len(pose_files)} file dữ liệu trong '{data_dir}'.")
    print("⏳ Đang tải dữ liệu...")

    for current_class_index, pose_file in enumerate(pose_files):
        file_path = os.path.join(data_dir, pose_file.name)
        try:
            pose_data = np.load(file_path)
            # Kiểm tra dữ liệu rỗng
            if pose_data.size == 0:
                print(f"⚠️ Cảnh báo: File {pose_file.name} rỗng, bỏ qua.")
                continue
                
            X.append(pose_data)
            y += [current_class_index] * pose_data.shape[0]
            mapping[current_class_index] = pose_file.name.split(".")[0]
            print(f"  + Đã tải lớp '{mapping[current_class_index]}': {pose_data.shape[0]} mẫu")
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc file {pose_file.name}: {e}")

    if not X:
        print("❌ Không có dữ liệu hợp lệ để huấn luyện.")
        exit(1)

    X, y = np.vstack(X), np.array(y)
    print(f"✅ Tải dữ liệu thành công.")
    print(f"→ Tổng số mẫu: {X.shape[0]}")
    print(f"→ Số lượng lớp: {len(mapping)} ({list(mapping.values())})\n")

    print("🚀 Đang huấn luyện mô hình SVM...")
    # Chia tập train/test tỉ lệ 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cấu hình SVM: probability=True để có thể tính độ tin cậy (confidence score) sau này
    model = SVC(decision_function_shape='ovo', kernel='rbf', C=100.0, gamma='scale', probability=True)
    model.fit(X_train, y_train)

    # Đánh giá
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("\n" + "-" * 30)
    print("KẾT QUẢ HUẤN LUYỆN")
    print("-" * 30)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")
    print(f"Classes: {len(mapping)}")
    print(f"✅ Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"✅ Test Accuracy:  {test_accuracy * 100:.2f}%")

    # Lưu model
    os.makedirs(args.dir, exist_ok=True)
    model_path = os.path.join(args.dir, f"{args.model_name}.pkl")
    with open(model_path, "wb") as file:
        pickle.dump((model, mapping), file)

    # Vẽ Confusion Matrix
    try:
        class_names = [mapping[i] for i in sorted(mapping.keys())]
        cm_path = os.path.join(args.dir, f"{args.model_name}_confusion_matrix.png")
        plot_confusion_matrix(y_test, y_test_pred, class_names, cm_path)
        
        # In báo cáo chi tiết dạng text
        print("\nChi tiết từng lớp (Classification Report):")
        print(classification_report(y_test, y_test_pred, target_names=class_names))
        
    except Exception as e:
        print(f"\n⚠️ Không thể vẽ biểu đồ (có thể thiếu thư viện matplotlib/seaborn): {e}")

    duration = (datetime.now() - start_time).seconds
    print(f"\n💾 Model đã lưu tại: {model_path}")
    print(f"⏱️ Hoàn thành trong {duration} giây.")
    print("=" * 80)

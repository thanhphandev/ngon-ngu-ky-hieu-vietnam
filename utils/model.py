import pickle
import numpy as np

class ASLClassificationModel:
    @staticmethod
    def load_model(model_path):
        # Load model and mapping from pickle
        with open(model_path, "rb") as file:
            model, mapping = pickle.load(file)

        if model is not None:
            return ASLClassificationModel(model, mapping)

        raise Exception("Model not loaded correctly!")

    def __init__(self, model, mapping):
        self.model = model
        self.mapping = mapping

    def predict(self, feature):
        """Trả về nhãn dự đoán (cách cũ)"""
        return self.mapping[self.model.predict(feature.reshape(1, -1)).item()]

    def predict_with_confidence(self, feature):
        """
        Trả về nhãn dự đoán VÀ độ tin cậy (confidence score).
        
        Returns:
            tuple: (label_str, confidence_float)
            Ví dụ: ("xin_chào", 0.95)
        """
        # Reshape feature để phù hợp input của model
        feature_reshaped = feature.reshape(1, -1)
        
        # Lấy xác suất của tất cả các lớp
        # Lưu ý: Model phải được train với probability=True (đã cập nhật trong train.py)
        try:
            # Kiểm tra xem model có hỗ trợ predict_proba không (SVC cần probability=True)
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(feature_reshaped)[0]
                max_index = np.argmax(probas)
                confidence = probas[max_index]
                label = self.mapping[max_index]
                return label, confidence
            else:
                # Fallback cho model cũ hoặc model không hỗ trợ xác suất
                return self.predict(feature), 1.0
        except Exception as e:
            print(f"Prediction Error: {e}")
            return "Error", 0.0
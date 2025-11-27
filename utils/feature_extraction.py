import numpy as np
from config import FEATURES_PER_HAND

def extract_hand_result(mp_hands: object, hand_results: object) -> np.ndarray:
    """
    Trích xuất đặc trưng bàn tay từ kết quả MediaPipe.
    
    Quy tắc:
    - Luôn trả về vector cố định kích thước (84 chiều).
    - Thứ tự: [Right Hand Landmarks] + [Left Hand Landmarks].
    - Nếu thiếu tay nào, điền toàn số 0 vào vị trí đó.
    
    Args:
        mp_hands: Module mp.solutions.hands (để truy cập hằng số).
        hand_results: Kết quả trả về từ hands.process().
        
    Returns:
        np.ndarray: Vector 1 chiều kích thước (84,).
    """
    # Nếu không phát hiện bàn tay nào → trả về vector toàn 0
    if hand_results is None or hand_results.multi_hand_landmarks is None:
        return np.zeros(FEATURES_PER_HAND * 4)  # 2 tay × (21 điểm × 2 toạ độ)

    hands = hand_results.multi_hand_landmarks
    handed = hand_results.multi_handedness

    right_hand_array = np.zeros((21, 2))
    left_hand_array = np.zeros((21, 2))

    for hand_landmark, hand_label in zip(hands, handed):
        label = hand_label.classification[0].label  # "Right" hoặc "Left"
        hand_array = extract_single_hand(mp_hands, hand_landmark)
        
        if label == "Right":  
            right_hand_array = hand_array
        else:
            left_hand_array = hand_array

    # Ghép 2 tay: Right trước, Left sau
    return np.hstack((right_hand_array.flatten(), left_hand_array.flatten()))

def extract_single_hand(mp_hands: object, hand_landmarks: object) -> np.ndarray:
    """
    Trích xuất tọa độ (x, y) của 21 điểm trên một bàn tay.
    
    Args:
        mp_hands: Module mp.solutions.hands.
        hand_landmarks: Đối tượng chứa 21 điểm landmark.
        
    Returns:
        np.ndarray: Mảng kích thước (21, 2).
    """
    landmarks_array = np.zeros((21, 2))
    for i, lm in enumerate(hand_landmarks.landmark):
        landmarks_array[i] = [lm.x, lm.y]
    return landmarks_array

def extract_face_result(face_results: object) -> np.ndarray:
    """
    Trích xuất đặc trưng khuôn mặt (tọa độ trung bình).
    
    Args:
        face_results: Kết quả từ face_mesh.process().
        
    Returns:
        np.ndarray: Vector 2 chiều [mean_x, mean_y].
    """
    if face_results is None or face_results.multi_face_landmarks is None:
        return np.zeros(2)

    face = face_results.multi_face_landmarks[0]
    # Chuyển 468 điểm thành mảng numpy
    face_array = np.array([[lm.x, lm.y] for lm in face.landmark])
    
    # Tính trung bình cộng để lấy tâm khuôn mặt
    return np.mean(face_array, axis=0)

def extract_features(mp_hands: object, face_results: object, hand_results: object) -> np.ndarray:
    """
    Hàm tổng hợp: Ghép đặc trưng Mặt + Tay thành một vector duy nhất.
    
    Structure: [Face (2)] + [Right Hand (42)] + [Left Hand (42)] = 86 features.
    
    Returns:
        np.ndarray: Vector đặc trưng tổng hợp (86 chiều).
    """
    face_features = extract_face_result(face_results)
    hand_features = extract_hand_result(mp_hands, hand_results)
    
    return np.hstack((face_features, hand_features))

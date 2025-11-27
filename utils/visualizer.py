import mediapipe as mp
import cv2

class Visualizer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands

    def draw_landmarks(self, image, face_results, hand_results):
        """
        Vẽ các điểm landmarks (xương khớp) lên hình ảnh camera.
        
        Args:
            image: Ảnh gốc từ camera (đã convert sang RGB hoặc BGR).
            face_results: Kết quả từ face_mesh.process().
            hand_results: Kết quả từ hands.process().
            
        Returns:
            image: Ảnh đã được vẽ các đường landmarks.
        """
        
        # 1. Vẽ lưới khuôn mặt (Face Mesh)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
                # Vẽ viền khuôn mặt & mắt mũi miệng cho rõ hơn (Optional)
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

        # 2. Vẽ xương bàn tay (Hand Landmarks)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), # Khớp màu đỏ
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # Xương màu xanh
                )
        
        return image

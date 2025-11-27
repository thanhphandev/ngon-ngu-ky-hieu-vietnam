
class ExpressionHandler:

    # Mapping từ nhãn gốc sang emoji + text
    MAPPING = {
        "buoi_sang": "Buổi sáng",
        "buoi_toi": "Buổi tối",
        "con_cho": "Con chó",
        "con_ga": "Con gà",
        "con_gian": "Con gián",
        "con_meo": "Con mèo",
        "con_muoi": "Con muỗi",
        "nhom":"Nhóm",
        "xin_chao":"Xin chào",
        "xin_loi":"Xin lỗi"
    }

    # Bản đọc cho TTS (không emoji, từ ngữ rõ ràng)
    SPEECH_MAPPING = {
        "buoi_sang": "Buổi sáng ",
        "buoi_toi": "Buổi tối",
        "con_cho": "Con chó",
        "con_ga": "Con gà",
        "con_gian": "Con gián",
        "con_meo": "Con mèo",
        "con_muoi": "Con muỗi",
        "nhom":"Nhóm",
        "xin_chao":"Xin chào",
        "xin_loi":"Xin lỗi"
    }

    def __init__(self):
        # Save the current message and the time received the current message
        self.current_message = ""

    def receive(self, message):
        self.current_message = message

    def get_message(self):
        # Trả về nhãn gốc nếu chưa có mapping thân thiện để tránh lỗi
        return ExpressionHandler.MAPPING.get(self.current_message, self.current_message)

    def get_speech_message(self):
        return ExpressionHandler.SPEECH_MAPPING.get(self.current_message, self.current_message)

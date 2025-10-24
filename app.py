import streamlit as st
import os
import shutil
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from google import genai
from google.genai import types
import torch.nn.functional as F

# ==================== CẤU HÌNH API & BIẾN TOÀN CỤC ====================
os.environ["GOOGLE_API_KEY"] = "AIzaSyAEnU_oVz1A18oC_zmxNvg4XR1NzSJYgzo"
API_KEY = os.getenv("GOOGLE_API_KEY")

# Khởi tạo Session State chỉ cho các biến cần thiết
if 'feedback_sent' not in st.session_state:
    st.session_state.feedback_sent = False
if 'current_image_hash' not in st.session_state:
    st.session_state.current_image_hash = None

# Khởi tạo Gemini Client
client = None
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo Gemini Client: {e}. Vui lòng kiểm tra GOOGLE_API_KEY.")
else:
    st.error("❌ Lỗi API: Không tìm thấy GOOGLE_API_KEY trong biến môi trường.")

# Danh sách các loại bệnh
CLASSES = {
    'Healthy': 'Khỏe mạnh (Healthy)',
    'Mosaic': 'Bệnh khảm lá (Mosaic Virus)',
    'RedRot': 'Bệnh thối đỏ (Red Rot)',
    'Rust': 'Bệnh gỉ sắt (Rust)',
    'Yellow': 'Vàng lá - Thiếu dinh dưỡng (Yellow Leaf)'
}

CONFIDENCE_THRESHOLD = 0.85

# ==================== HÀM HỖ TRỢ CƠ BẢN ====================
def set_seed(seed=42):
    """Đặt seed cho tính tái lập kết quả"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_transforms():
    """Trả về transform để xử lý ảnh đầu vào"""
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

@st.cache_resource
def load_model():
    """Tải mô hình AI từ file hoặc tạo mô hình mặc định"""
    model_path = 'Mo_Hinh/ModelAI_DuDoanBenh.pth'
    
    if os.path.exists(model_path):
        try:
            model = torch.jit.load(model_path, map_location=torch.device('cpu'))
            model.eval()
            return model
        except Exception as e:
            st.error(f"❌ Lỗi tải mô hình: {e}")

    st.warning("⚠️ Không tìm thấy mô hình. Đang tạo mô hình ResNet18 mặc định...")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    model.eval()
    return model

def predict_disease(image, model):
    transform = get_transforms()
    input_tensor = transform(image).unsqueeze(0)
    device = torch.device("cpu")
    input_tensor = input_tensor.to(device)
    model = model.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)
        pred_idx = preds.item()
        confidence_value = confidence.item()
    class_keys = list(CLASSES.keys())
    predicted_class_key = class_keys[pred_idx]
    predicted_class_name = CLASSES[predicted_class_key]
    is_confident = confidence_value >= CONFIDENCE_THRESHOLD
    return predicted_class_name, confidence_value, is_confident

def save_feedback(image_path, predicted_class, is_correct):
    base_dir = 'feedback'
    split = 'True' if is_correct else 'False'
    target_dir = os.path.join(base_dir, split, predicted_class)
    os.makedirs(target_dir, exist_ok=True)
    new_filename = f"{int(time.time())}_{os.path.basename(image_path)}"
    shutil.copy(image_path, os.path.join(target_dir, new_filename))
    return target_dir

def get_image_hash(image):
    import hashlib
    return hashlib.md5(image.tobytes()).hexdigest()

# ==================== LỚP HỖ TRỢ GEMINI ====================
class GeminiHelper:
    def __init__(self, client):
        self.chat = client.chats.create(model="gemini-2.5-flash")

    def consult_treatment(self, query: str):
        system_prompt = (
            "Bạn là trợ lý AI phác thảo lộ trình điều trị bệnh trên cây mía, được tạo ra bởi các bạn học sinh và thầy cô trường THCS Trung Thành – Tuyên Quang. "
            "Nhiệm vụ của bạn là chuyên gia bệnh cây mía Việt Nam, chỉ tư vấn về các loại bệnh trên cây mía như khảm lá, thối đỏ, gỉ sắt, vàng lá và các vấn đề sinh lý dinh dưỡng của cây mía. "
            "Yêu cầu thông tin phải chính xác, đúng khoa học và phù hợp với điều kiện nông nghiệp Việt Nam, được tổng hợp từ các tài liệu uy tín. "
            "Nếu người dùng hỏi chủ đề khác ngoài các loại bệnh trên cây mía, trả lời khéo không có thông tin thay vì bịa đặt. "
            "Nếu người dùng hỏi kỹ thông tin nhóm tác giả thì trả lời nhóm tác giả gồm: Giáo viên hướng dẫn: Lê Quang Phúc, Nguyễn Thị Lý. Học sinh thực hiện: Đặng Tiến Huynh, Trần Thành Hưng – lớp 8. "
            "Khi trả lời, hãy: "
            "Ngắn gọn, rõ ràng, tránh lan man nhầm lẫn sang các loại bệnh khác, xuống dòng khi cần thiết đối với các câu dài. "
            "Không bịa đặt thông tin, chỉ dùng kiến thức nông nghiệp chuẩn khoa học. "
            "Có thể dùng công cụ tìm kiếm web để xác thực nguồn (nếu có thì trả lời kèm theo Link bài viết). "
            "Trả lời bằng ngôn ngữ của người dùng (đa ngôn ngữ), sử dụng định dạng markdown đơn giản: ** cho bold, * cho italic, - cho danh sách không thứ tự, 1. cho danh sách có thứ tự, không sử dụng bất kỳ thẻ HTML nào như <p>, <ol>, <li>, <div> để tránh lỗi hiển thị trong Streamlit. "
            "Giữ cho phản hồi ngắn gọn, mỗi phần không quá dài, tránh khoảng trống thừa hoặc lặp lại để bố cục chat đẹp và dễ đọc. "
            "Luôn duy trì vai trò chuyên gia bệnh cây mía Việt Nam, đối với các câu hỏi không cần thông tin của nhóm tác giả thì không cần trả lời về nhóm tác giả. "
            "Trả lời tối đa 300 từ."
        )

        try:
            res = self.chat.send_message(
                [types.Part(text=query)],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[{"google_search": {}}],
                ),
            )
            text = res.text or "Không có phản hồi rõ ràng."
            cite = ""
            gm = getattr(res.candidates[0], "grounding_metadata", None)
            if gm and getattr(gm, "web_search_queries", None):
                cite = "🔎 Nguồn: " + ", ".join(gm.web_search_queries)
            return text, cite
        except Exception as e:
            return f"Lỗi Gemini: {e}", ""

# ==================== KẾ HOẠCH ĐIỀU TRỊ CƠ BẢN ====================
def get_treatment_plan(disease_name):
    plans = {
'Khỏe mạnh (Healthy)': """
<div class="treatment-plan">
<h4 style='color:#2ecc71;'>✅ CÂY MÍA KHỎE MẠNH</h4>

**🧪 Đặc điểm chung:**  
Cây mía khỏe mạnh có lá xanh bóng, thân đứng vững, rễ phát triển tốt, không héo, thối hay biến dạng. Đây là trạng thái lý tưởng giúp cây quang hợp tối đa và đạt năng suất cao.

**🧴 Hướng dẫn chăm sóc định kỳ:**  
- Theo dõi sinh trưởng mỗi **7–10 ngày**, chú ý sâu bệnh, độ ẩm, pH đất.  
- Bón phân **NPK 16-16-8: 500 kg/ha/vụ**, chia 2–3 lần trong mùa sinh trưởng.  
- Tưới đều, duy trì **20 mm nước/tuần**, tránh úng ngập kéo dài.  
- Làm cỏ, xới xáo nhẹ để đất thông thoáng.

**🛡 Phòng ngừa:**  
- Dùng **giống mía kháng bệnh**, nguồn gốc rõ ràng.  
- Cân đối phân bón, tránh thừa đạm, tăng kali & hữu cơ.  
- Quản lý nước hợp lý.

**🌾 Mẹo:**  
- Bổ sung **Trichoderma** để tăng đề kháng rễ.  
- Ghi nhật ký chăm sóc để tối ưu vụ sau.
</div>
""",

'Bệnh khảm lá (Mosaic Virus)': """
<div class="treatment-plan">
<h4 style='color:#e67e22;'>🦠 BỆNH KHẢM LÁ (MOSAIC VIRUS)</h4>

**🧪 Nguyên nhân:**  
- Do **virus SCMV** gây ra, lây qua **rệp muội (Aphis spp.)** hoặc cây giống nhiễm bệnh.  
- Phát triển mạnh khi **ẩm nóng**, trồng dày, chăm sóc kém.

**🧴 Hướng dẫn điều trị:**  
- Phun **Imidacloprid 0,5 ml/lít**, 2 lần cách 7 ngày để diệt rệp.  
- **Nhổ bỏ – tiêu hủy** cây bệnh nặng.  
- Khử trùng dụng cụ bằng **cồn 70° hoặc Cloramin B** sau khi cắt tỉa.

**🛡 Phòng ngừa:**  
- Dùng **giống kháng Mosaic**, giống sạch bệnh.  
- Không trồng nhiều vụ liên tiếp cùng giống.  
- Kiểm tra rệp định kỳ, giữ ruộng thoáng.

**🌾 Mẹo:**  
- Dùng **bẫy dính vàng** giám sát rệp.  
- **Trồng xen cúc vạn thọ, húng quế** để xua rệp tự nhiên.
</div>
""",

'Bệnh thối đỏ (Red Rot)': """
<div class="treatment-plan">
<h4 style='color:#c0392b;'>🍄 BỆNH THỐI ĐỎ (RED ROT)</h4>

**🧪 Nguyên nhân:**  
- Nấm **Colletotrichum falcatum Went** gây bệnh, phát triển khi **ẩm cao, đất úng, trồng dày**.  
- Dễ nhiễm sau mưa dài hoặc cây bị tổn thương.

**🧴 Hướng dẫn điều trị:**  
- Dùng **Carbendazim 50% – 500 g/ha**, tưới đều quanh gốc.  
- Cắt bỏ phần thối, **đốt tiêu hủy**.  
- Nếu lan rộng, **luân canh cây họ đậu 1 vụ** để cắt nguồn nấm.

**🛡 Phòng ngừa:**  
- Dùng **giống kháng nấm**, không lấy hom từ ruộng bệnh.  
- **Cải tạo rãnh thoát nước**, trồng đất cao.  
- Không trồng mía liên tục >3 vụ.

**🌾 Mẹo:**  
- Sau vụ thu, **cày phơi ải ≥3 tuần**.  
- Bón **vôi bột 300 kg/ha** sau thu hoạch để diệt nấm.
</div>
""",

'Bệnh gỉ sắt (Rust)': """
<div class="treatment-plan">
<h4 style='color:#d35400;'>🍂 BỆNH GỈ SẮT (SUGARCANE RUST)</h4>

**🧪 Nguyên nhân:**  
- Nấm **Uromyces scitamineus** gây bệnh, phát triển ở **25–30°C**, ẩm cao.  
- Lây lan qua **gió, mưa, dụng cụ nông nghiệp**.

**🧴 Hướng dẫn điều trị:**  
- Phun **Mancozeb 80WP 2 kg/ha**, 3 lần cách 7–10 ngày.  
- **Cắt lá bệnh – đốt tiêu hủy.**  
- Bón thêm **Kali** để tăng sức đề kháng.

**🛡 Phòng ngừa:**  
- Dùng **giống kháng gỉ sắt** được Viện Mía Đường khuyến cáo.  
- Giữ ruộng thoáng, tránh trồng quá dày.  
- Tránh **bón thừa đạm**.

**🌾 Mẹo:**  
- Phun **sáng sớm hoặc chiều mát**, không có gió.  
- **Luân phiên thuốc gốc đồng – Mancozeb** tránh kháng nấm.
</div>
""",

'Vàng lá - Thiếu dinh dưỡng (Yellow Leaf)': """
<div class="treatment-plan">
<h4 style='color:#f1c40f;'>🌱 VÀNG LÁ – THIẾU DINH DƯỠNG</h4>

**🧪 Nguyên nhân:**  
- Thiếu **đạm (N)**, **lưu huỳnh (S)** hoặc **sắt (Fe)**.  
- Đất chua **(pH < 5,5)** hoặc **rửa trôi phân bón** do mưa, tưới nhiều.

**🧴 Hướng dẫn điều trị:**  
- Bón **Urê 200 kg/ha**, chia 2–3 lần/vụ.  
- Phun **Urê 5%** hoặc **phân bón lá chứa Fe, Zn**.  
- Kiểm tra pH, nếu thấp thì **bón vôi 100–200 kg/ha**.

**🛡 Phòng ngừa:**  
- Duy trì **pH đất 6.0–7.0**, cân đối NPK.  
- Tăng **phân hữu cơ, phân xanh** để cải thiện đất.  
- Theo dõi lá định kỳ.

**🌾 Mẹo:**  
- Sau mưa, **bổ sung phân bón lá nhẹ**.  
- Dùng **than sinh học (biochar)** trộn đất để giữ ẩm, dinh dưỡng lâu hơn.
</div>
"""
    }
    return plans.get(disease_name, "<div class='treatment-plan'>Không có thông tin điều trị cho bệnh này.</div>")

# ==================== GIAO DIỆN CHÍNH ====================
def main():
    st.set_page_config(layout="wide", page_title="🌾 AI Cây Mía Nâng Cao")

    # CSS tùy chỉnh
    st.markdown("""
    <style>
    .stVerticalBlock.st-emotion-cache-wfksaw.e196pkbe2 {     border-radius:10px; padding:10px; }

    .chat-message {
        margin: 10px 0;
        margin-bottom: 0.5cm;
        padding: 10px;
        border-radius: 10px;
        max-width: 85%;
        word-wrap: break-word;
    }
    .chat-user {
        font-size: 14px;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        margin-left: auto;
        text-align: right;
        padding: 8px 12px;
        border-radius: 12px;
        max-width: 85%;
        width: fit-content;
        word-wrap: break-word;
    }
    .st-emotion-cache-12j140x p, .st-emotion-cache-12j140x ol, .st-emotion-cache-12j140x ul, .st-emotion-cache-12j140x dl, .st-emotion-cache-12j140x li {
    font-size: 18px;
    line-height: 1.6;
    align-items: justify;
    }
    .chat-assistant {
        font-size: 14px;
        background-color: #28a745;
        color: white;
        font-weight: bold;
        margin-right: auto;
        text-align: left;
        padding: 8px 12px;
        border-radius: 12px;
        max-width: 85%;
        width: fit-content;
        word-wrap: break-word;
    }
    .treatment-plan {
    font-size: 15px;
    line-height: 1.5;
    text-align: justify;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tải mô hình AI
    model = load_model()

    # Khởi tạo Gemini Helper
    gemini = None
    if client:
        gemini = GeminiHelper(client)

    # Tiêu đề ứng dụng
    st.markdown("""
    <div style ="text-align:center">
    <img src="https://sf-static.upanhlaylink.com/img/image_20251025808f56c2d9f005cbfe25d1e3fe66cd44.jpg" style="width:auto; height:200px; border-radius:5px;">
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; color:darkblue; font-weight:bold; margin-bottom:0.5cm;">
            <h2>AI NHẬN DIỆN VÀ PHÁC THẢO ĐIỀU TRỊ BỆNH CÂY MÍA</h2>
        </div>
    """, unsafe_allow_html=True)
    # Thanh bên (Sidebar)
    with st.sidebar:
        tab1, tab2 = st.tabs(["📖 Hướng dẫn", "ℹ️ Thông tin"])
        with tab1:
            st.markdown("""
            <h3 style = "color:red;text-align:center">HƯỚNG DẪN SỬ DỤNG</h3>""", unsafe_allow_html=True)
            st.markdown("""
            **Bước 1️⃣**: Tải hoặc chụp ảnh lá mía. 
                        
            **Bước 2️⃣**: AI phân tích và hiển thị kết quả nếu độ tin cậy ≥85%.  
                        
            **Bước 3️⃣**: Xem kế hoạch điều trị chi tiết.  
                        
            **Bước 4️⃣**: Đặt câu hỏi hoặc gửi phản hồi để cải thiện mô hình.  
            """)
        with tab2:
            # --- Phần tiêu đề ---
            st.markdown("""
                <h3 style='color:red; text-align:center;'>📘 THÔNG TIN ĐỀ TÀI</h3>
            """, unsafe_allow_html=True)

            # --- Thông tin đề tài chính ---
            st.markdown("""
            <div style="
                line-height: 1.6;
                text-align: justify;
                font-size: 16px;
            ">
            <b>🎯 Đề tài:</b> AI NHẬN DIỆN VÀ PHÁC THẢO ĐIỀU TRỊ BỆNH CÂY MÍA  
            <br>
            <br>
            <b>🏫 Đơn vị thực hiện:</b> Trường THCS Trung Thành – Tuyên Quang  
            <br>
            <br>
            <b>👨‍🏫 Giáo viên hướng dẫn:</b> Lê Quang Phúc, Nguyễn Thị Lý  
            <br>
            <br>
            <b>👩‍🎓 Học sinh thực hiện:</b> Đặng Tiến Huynh, Trần Thành Hưng (Lớp 8)
            </div>
            """, unsafe_allow_html=True)

    # Chia layout thành 2 cột
    col1, col2 = st.columns(2)

    # ========== CỘT 1: NHẬN DIỆN BỆNH ==========
    with col1:
        st.markdown('<h3 style="text-align: center;color:white; background-color: #7f69f4; padding: 10px; border-radius: 5px; margin-bottom:1cm;">🔍 AI CHẨN ĐOÁN BỆNH</h3>', unsafe_allow_html=True)
        st.markdown('<p style ="color:red;text-align:center; font-size:14px;"><strong>Lưu ý chỉ sử dụng hình ảnh lá cây mía có dấu hiệu bệnh để dự đoán<strong><div style="font-size: 16px; font-weight: bold; margin:0 0 0.5cm 0;">Chọn phương thức nhập ảnh:</div>  ', unsafe_allow_html=True)
        input_method = st.radio("Chọn phương thức nhập ảnh:", ["Tải ảnh", "Chụp từ webcam"], key="input_method", label_visibility="collapsed")
        image = None
        if input_method == "Tải ảnh":
            st.markdown('<div style="font-size: 16px; color: darkblue; font-weight:bold;margin-bottom:0.5cm;">📸 Tải ảnh lá cây mía</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("📸 Tải ảnh lá cây mía", type=['png', 'jpg', 'jpeg'], key="image_uploader", label_visibility="collapsed")
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
        else:
            st.markdown('<div style="font-size: 16px;">📷 Chụp ảnh từ webcam</div>', unsafe_allow_html=True)
            camera_input = st.camera_input("📷 Chụp ảnh từ webcam", key="camera_input", label_visibility="collapsed")
            if camera_input:
                image = Image.open(camera_input).convert("RGB")

        if image:
            st.image(image, width=200, caption="Ảnh được nhập")
            current_hash = get_image_hash(image)
            if st.session_state.current_image_hash != current_hash:
                st.session_state.feedback_sent = False
                st.session_state.current_image_hash = current_hash
            with st.spinner("🔬 AI đang phân tích..."):
                disease_name, confidence, is_confident = predict_disease(image, model)
            if is_confident:
                st.markdown(f"""
                <div style="background-color: #dc3545; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                    <p style="margin: 0; font-size:19px; font-weight: bold;"> 🎯 Cảnh báo: {disease_name}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background-color: blue; color: white; padding: 5px; border-radius: 10px; text-align: center; margin: 10px 0;">
                    <p style="margin: 0; font-size:19px; font-weight: bold;">📊 Độ chính xác dự đoán: {confidence*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
                st.subheader("💡 **KẾ HOẠCH ĐIỀU TRỊ CƠ BẢN**")
                with st.container():
                    st.markdown(f'<div class="treatment-plan">{get_treatment_plan(disease_name)}</div>', unsafe_allow_html=True)
                st.subheader("📝 **PHẢN HỒI (FEEDBACK)**")
                if st.session_state.feedback_sent:
                    st.success("✅ Bạn đã gửi phản hồi cho ảnh này rồi!")
                else:
                    st.markdown('<div style="font-size: 18px;">Kết quả dự đoán có đúng không?</div>', unsafe_allow_html=True)
                    correct = st.radio("", ["Đúng", "Sai"], key="feedback_radio", label_visibility="collapsed")
                    col_center1, col_center2, col_center3 = st.columns([1, 2, 1])
                    with col_center2:
                        if st.button("💾 Lưu Feedback", use_container_width=True):
                            img_path = f"temp_{int(time.time())}_feedback.jpg"
                            image.save(img_path)
                            save_feedback(img_path, disease_name, correct=="Đúng")
                            st.success("✅ Đã lưu phản hồi! Dữ liệu sẽ được dùng để cải thiện mô hình.")
                            os.remove(img_path)
                            st.session_state.feedback_sent = True
                            time.sleep(0.5)
                            st.rerun()
            else:
                st.warning("### ⚠️ KHÔNG THỂ XÁC ĐỊNH CHÍNH XÁC")
                st.info(f"📊 **Độ tin cậy:** {confidence*100:.2f}% (Cần ≥ {CONFIDENCE_THRESHOLD*100}%)")
                st.markdown("""
                **HÌNH ẢNH NÀY TÔI KHÔNG CHẮC CHẮN VÌ CÓ NHIỀU YẾU TỐ:**
                - 📸 Ảnh mờ hoặc không rõ nét
                - 💡 Ánh sáng không đủ hoặc quá sáng
                - 🍃 Chụp nhiều lá cùng
                - 🚫 Ảnh không liên quan đến lá cây mía
                - 🔄 Góc chụp không phù hợp
                            
                **💡 ĐỀ XUẤT:**
                - Chụp lại ảnh với ánh sáng tốt hơn
                - Chụp 1 lá riêng biệt, rõ nét
                - Đảm bảo ảnh là lá cây mía thật
                """)

    # ========== CỘT 2: CHATBOT TƯ VẤN ==========
    with col2:
        st.markdown('''
        <h3 style="text-align: center; background-color: #249adc; color: white; padding: 10px; border-radius: 5px; margin-bottom: 1cm;">
            <img src="https://cdn-icons-png.flaticon.com/512/8943/8943377.png" 
                alt="Icon" 
                style="width: 30px; height: 30px; vertical-align: middle; margin-right: 10px;">
            AI PHÁC THẢO ĐIỀU TRỊ
        </h3>
        ''', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message chat-assistant">Xin chào tôi là trợ lý ảo do nhóm học sinh Trường THCS Trung Thành – Tuyên Quang tạo ra!</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message chat-assistant">Tôi có thể đồng hành với bạn để hướng dẫn bạn điều trị các bệnh trên cây mía.</div>', unsafe_allow_html=True)

        if gemini:
            # Tạo biến tạm để lưu lịch sử chat trong phiên hiện tại
            if 'temp_chat_history' not in st.session_state:
                st.session_state.temp_chat_history = []

            # Hiển thị lịch sử chat tạm thời
            for message in st.session_state.temp_chat_history:
                role_class = "chat-user" if message['role'] == "user" else "chat-assistant"
                st.markdown(f'<div class="chat-message {role_class}">{message["text"]}</div>', unsafe_allow_html=True)

            # Ô nhập câu hỏi
            query = st.chat_input("Hỏi chuyên gia về các loại bệnh")
            if query:
                # Thêm câu hỏi của người dùng vào lịch sử tạm
                st.session_state.temp_chat_history.append({"role": "user", "text": query})
                with st.spinner("🤖 Chuyên gia Gemini đang trả lời..."):
                    response_text, citations = gemini.consult_treatment(query)
                    st.session_state.temp_chat_history.append({"role": "assistant", "text": response_text + (f"\n{citations}" if citations else "")})
                st.rerun()
        else:
            st.warning("⚠️ Chatbot bị vô hiệu hóa do lỗi API Key. Vui lòng kiểm tra cấu hình API.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style ="text-align: center; gap: 0.5cm;">
    <img src="https://sf-static.upanhlaylink.com/img/image_2025102510ade85883871ec5ef61bbd51e2d62c3.jpg" style="margin:3cm 0 0.5cm 0; width:auto; height:400px; border-radius:5px;">
    <img src="https://sf-static.upanhlaylink.com/img/image_202510254eab585eca869553b497f7ae99f2c212.jpg" style="margin:3cm 0 0.5cm 0; width:auto; height:400px; border-radius:5px;">
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p style="text-align: center;"> <strong style="color:darkblue">Tác giả dự án:</strong> Nhóm nghiên cứu Trường THCS Trung Thành – Tuyên Quang</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;"> <strong style="color:red">Học sinh thực hiện:</strong> Đặng Tiến Huynh, Trần Thành Hưng</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;"> <strong style="color:red">Giáo viên hướng dẫn:</strong> Lê Quang Phúc, Nguyễn Thị Lý</p>', unsafe_allow_html=True)
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('feedback/True', exist_ok=True)
    os.makedirs('feedback/False', exist_ok=True)
    main()
import base64
import io
import os

import streamlit as st
from openai import APIConnectionError, APIStatusError, OpenAI
from PIL import Image, ImageOps


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 1536  # Thu nhỏ để gửi nhanh và tiết kiệm token
MAX_OUTPUT_TOKENS = 4096  # Gemini 2.5 tính cả token suy luận vào giới hạn này
SO_MODEL_THU_TOI_DA = 3  # Khi hết hạn mức, thử tối đa 3 model Flash
# Gói miễn phí chỉ có hạn mức cho model Flash / Flash-Lite.
# Model Pro, Preview, tạo ảnh, TTS, Live, Embedding... không có hạn mức miễn phí nên bị ẩn.
TU_KHOA_BI_LOAI = ("pro", "preview", "image", "tts", "live", "audio", "embedding", "veo",
                   "lyria", "omni", "argon", "robotics", "computer-use", "native")
UU_TIEN_MODEL = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
GOI_Y_THEO_MA = {
    400: "Google báo yêu cầu hoặc key không hợp lệ. Kiểm tra lại key AIza... trong Google AI Studio.",
    401: "Key không hợp lệ hoặc thiếu. Kiểm tra lại key.",
    403: "Key không có quyền dùng model này. Kiểm tra trong Google AI Studio.",
    404: "Model không còn được hỗ trợ. Hãy chọn model Flash khác trong danh sách.",
    429: (
        "Đã hết hạn mức miễn phí (RESOURCE_EXHAUSTED) cho model này. Hạn mức tính theo dự án, "
        "nên tạo thêm key không làm tăng hạn mức. Hãy chờ rồi thử lại, hoặc chọn model Flash khác."
    ),
    503: "Model đang quá tải. Thử lại sau ít phút.",
}


st.set_page_config(
    page_title="Bác Sĩ Cây Trồng - Sinh Học 11",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 Bác Sĩ Cây Trồng - Chẩn Đoán Dinh Dưỡng Khoáng (Gemini AI)")
st.markdown("Ứng dụng AI hỗ trợ chẩn đoán triệu chứng thiếu hụt dinh dưỡng ở thực vật sử dụng Gemini API.")


def lay_key_cau_hinh() -> str:
    """Đọc key từ .streamlit/secrets.toml hoặc biến môi trường GOOGLE_API_KEY."""
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:
            key = ""
    return str(key).strip()


def mo_ta_loi(e: APIStatusError) -> str:
    """Lấy nguyên văn phản hồi từ Google để biết lỗi cụ thể."""
    resp = getattr(e, "response", None)
    text = (resp.text or "").strip() if resp is not None else ""
    return text or "(Google không gửi kèm nội dung lỗi)"


def _thu_tu_uu_tien(model_id: str) -> tuple:
    if model_id in UU_TIEN_MODEL:
        return (UU_TIEN_MODEL.index(model_id), model_id)
    return (len(UU_TIEN_MODEL), model_id)


@st.cache_data(show_spinner=False, ttl=600)
def lay_danh_sach_model(api_key: str) -> list[str]:
    """Lấy các model Flash có hạn mức miễn phí. Việc này không tốn hạn mức inference."""
    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    ids = set()
    for m in client.models.list():
        mid = m.id.removeprefix("models/")
        if mid.startswith("gemini") and "flash" in mid:
            if not any(k in mid for k in TU_KHOA_BI_LOAI):
                ids.add(mid)
    return sorted(ids, key=_thu_tu_uu_tien)


def goi_mo_hinh(client: OpenAI, model_name: str, messages: list, danh_sach: list[str]):
    """Gọi model đã chọn. Nếu hết hạn mức hoặc không dùng được, thử model Flash kế tiếp."""
    thu = [model_name] + [m for m in danh_sach if m != model_name][: SO_MODEL_THU_TOI_DA - 1]
    loi_cuoi = None
    for mid in thu:
        try:
            phan_hoi = client.chat.completions.create(
                model=mid,
                messages=messages,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            return mid, phan_hoi
        except APIStatusError as e:
            if e.status_code in (404, 429, 503):
                loi_cuoi = e
                continue
            raise
    raise loi_cuoi


def ma_hoa_anh(image: Image.Image) -> str:
    anh = image.copy()
    anh.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    anh.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def tao_prompt(mo_ta: str) -> str:
    mo_ta = mo_ta.strip() or "không có mô tả thêm"
    return f"""Bạn là chuyên gia sinh học thực vật, hỗ trợ học sinh lớp 11 (chương dinh dưỡng khoáng ở thực vật).
Phân tích ảnh lá cây. Mô tả triệu chứng của người dùng: "{mo_ta}".

Trả lời bằng tiếng Việt theo đúng 3 phần:
1. Nguyên tố khoáng có khả năng thiếu cao nhất (Đạm N, Lân P, Kali K, Canxi Ca, Magiê Mg, Sắt Fe...), kèm mức độ: cao, trung bình hoặc thấp.
2. Cơ sở sinh lý: vai trò của nguyên tố đó và vì sao thiếu nó gây ra triệu chứng quan sát được.
3. Biện pháp khắc phục khái quát cho cây trồng.

Lưu ý: đây chỉ là gợi ý dựa trên ảnh. Hãy nêu các nguyên nhân khác có triệu chứng tương tự (sâu bệnh, úng hoặc khô hạn, pH đất) và khuyến nghị xác nhận bằng phân tích đất hoặc mô lá."""


# ---------- Sidebar: cấu hình và kiểm tra key ----------
st.sidebar.header("Cấu hình Gemini AI")
api_key = lay_key_cau_hinh()
if api_key:
    st.sidebar.success("Đã tải API key từ cấu hình.")
    st.sidebar.caption(f"Key đang dùng: {api_key[:6]}…{api_key[-4:]}")
else:
    api_key = st.sidebar.text_input("Nhập Google API Key (Gemini):", type="password").strip()

if api_key and not api_key.startswith("AIza"):
    st.sidebar.warning(
        "Key Gemini thường bắt đầu bằng 'AIza'. Hãy kiểm tra xem có đang dán nhầm key khác không."
    )

model_name = None
danh_sach: list[str] = []
if not api_key:
    st.sidebar.warning("Vui lòng nhập Google API Key để sử dụng tính năng phân tích.")
else:
    try:
        danh_sach = lay_danh_sach_model(api_key)
        if danh_sach:
            st.sidebar.success("Đã kết nối Gemini thành công.")
            st.sidebar.info(
                "Gói miễn phí: chỉ hiện các model Flash có hạn mức miễn phí. "
                "Dữ liệu gửi qua gói miễn phí có thể được Google dùng để cải thiện sản phẩm."
            )
            model_name = st.sidebar.selectbox("Chọn model:", danh_sach)
        else:
            st.sidebar.error("Không tìm thấy model Flash nào cho key này.")
    except APIConnectionError:
        st.sidebar.error("Không kết nối được tới Google. Kiểm tra mạng.")
    except APIStatusError as e:
        st.sidebar.error(
            f"Google trả về lỗi {e.status_code} khi lấy danh sách model. "
            + GOI_Y_THEO_MA.get(e.status_code, "")
        )
        st.sidebar.code(f"HTTP {e.status_code}\n{mo_ta_loi(e)}")
    except Exception as e:
        st.sidebar.error(f"Lỗi không xác định: {e}")


# ---------- Tải ảnh ----------
uploaded_file = st.file_uploader(
    "Tải lên hình ảnh lá cây bị triệu chứng bất thường:",
    type=["jpg", "jpeg", "png", "jfif", "webp"],
)

if uploaded_file is None:
    st.info("Hãy tải lên một bức ảnh chụp lá cây để bắt đầu chẩn đoán.")
    st.stop()

if uploaded_file.size > MAX_UPLOAD_BYTES:
    st.error("Ảnh vượt quá 10 MB. Hãy chọn ảnh nhỏ hơn.")
    st.stop()

try:
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    image = ImageOps.exif_transpose(image).convert("RGB")
except Exception:
    st.error("Không đọc được file ảnh. Hãy thử ảnh khác.")
    st.stop()

st.image(image, caption="Hình ảnh cây trồng tải lên")

symptom_desc = st.text_area(
    "Mô tả thêm triệu chứng (ví dụ: lá vàng giữa gân, chóp lá cháy, lá non bị quăn...):"
)


# ---------- Chẩn đoán ----------
if st.button("🔍 Chẩn Đoán Ngay", type="primary"):
    if model_name is None:
        st.error("Hãy nhập API Key hợp lệ và chọn model ở thanh bên trái.")
        st.stop()

    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    img_b64 = ma_hoa_anh(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": tao_prompt(symptom_desc)},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ],
        }
    ]

    with st.spinner(f"Đang phân tích bằng {model_name}..."):
        try:
            mid_da_dung, phan_hoi = goi_mo_hinh(client, model_name, messages, danh_sach)
            lua_chon = phan_hoi.choices[0]
            noi_dung = (lua_chon.message.content or "").strip()

            if mid_da_dung != model_name:
                st.info(f"Model {model_name} không dùng được lúc này, đã chuyển sang {mid_da_dung}.")

            if noi_dung:
                st.success("Kết quả chẩn đoán từ Gemini AI:")
                st.markdown(noi_dung)
            else:
                st.warning(
                    f"Model không trả về nội dung (lý do kết thúc: {lua_chon.finish_reason}). "
                    "Hãy thử lại hoặc chọn model khác."
                )
        except APIConnectionError:
            st.error("Không kết nối được tới Google. Kiểm tra mạng.")
        except APIStatusError as e:
            st.error(f"Google trả về lỗi {e.status_code}. " + GOI_Y_THEO_MA.get(e.status_code, ""))
            st.code(f"HTTP {e.status_code}\n{mo_ta_loi(e)}")
        except Exception as e:
            st.error(f"Lỗi không xác định: {e}")
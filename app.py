import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageColor
import numpy as np
import cv2
from io import BytesIO
import sys

from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
import matplotlib.cm as cm

# ======================= 털 제거 함수 추가 ==========================
def remove_hairs(img):
    """
    img: RGB numpy (H, W, 3)
    return: hair-removed RGB numpy
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, np.ones((3,3), np.uint8), iterations=1)

    clean = cv2.inpaint(img, thresh, 3, cv2.INPAINT_TELEA)
    return clean

# ======================= UNetImproved ==========================
class ResidualConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return self.relu(
            self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x)
        )

class UNetImproved(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = ResidualConv(3,64)
        self.down2 = ResidualConv(64,128)
        self.down3 = ResidualConv(128,256)
        self.down4 = ResidualConv(256,512)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = ResidualConv(512+256,256)
        self.up3 = ResidualConv(256+128,128)
        self.up2 = ResidualConv(128+64,64)
        self.output = nn.Conv2d(64,1,1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        u4 = self.up(x4)
        u4 = self.up4(torch.cat([u4, x3], dim=1))
        u3 = self.up(u4)
        u3 = self.up3(torch.cat([u3, x2], dim=1))
        u2 = self.up(u3)
        u2 = self.up2(torch.cat([u2, x1], dim=1))
        return self.output(u2)

# ======================= 설정 ==========================
MODEL_PATH = "final_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

@st.cache_resource
def load_model():
    m = UNetImproved().to(device)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.eval()
    return m

# ======================= 후처리 ==========================
def postprocess(mask, open_k=0, close_k=0, fill=False, remove_small_val=0):
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if open_k > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    if close_k > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    if fill:
        mask = binary_fill_holes(mask > 0).astype(np.uint8) * 255
    if remove_small_val > 0:
        temp = remove_small_objects(mask > 0, remove_small_val)
        mask = (temp.astype(np.uint8) * 255)
    return mask

def largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    mask_largest = (labels == largest_idx).astype(np.uint8) * 255
    return mask_largest

def auto_crop(img, mask, pad=10):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img
    x1, x2 = max(xs.min()-pad,0), min(xs.max()+pad, img.shape[1])
    y1, y2 = max(ys.min()-pad,0), min(ys.max()+pad, img.shape[0])
    return img[y1:y2, x1:x2]

def mask_to_transparent_png(mask):
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = mask
    rgba[..., 0:3][mask>0] = 255
    _, buf = cv2.imencode('.png', rgba)
    return buf.tobytes()

# ======================= 메인 UI ==========================
st.title("피부 병변 분할 - 확장 기능 포함")

uploaded = st.file_uploader("이미지 업로드", type=["png","jpg","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    orig_w, orig_h = img.size
    img_np_orig = np.array(img)

    # =================== 털 제거 UI 추가 ===================
    st.subheader("전처리 옵션")
    apply_hair = st.checkbox("털 제거 (DullRazor)", False)

    if apply_hair:
        img_np_orig = remove_hairs(img_np_orig)

    # =================== 모델 입력 준비 ===================
    img_resized = Image.fromarray(img_np_orig).resize((IMG_SIZE, IMG_SIZE))
    img_np_resized = np.array(img_resized)

    model = load_model()
    x = transforms.ToTensor()(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_t = model(x)
    pred = torch.sigmoid(pred_t).squeeze().cpu().numpy()

    # ================= Prediction 옵션 =================
    st.subheader("Prediction 옵션")
    th = st.slider("Prediction Threshold", 0.0, 1.0, 0.5)
    base_mask = (pred > th).astype(np.uint8) * 255

    # ================ 후처리 옵션 =================
    st.subheader("후처리 옵션")
    open_k = st.slider("Open kernel", 0, 31, 0)
    close_k = st.slider("Close kernel", 0, 31, 0)
    fill_holes = st.checkbox("Fill Holes", False)
    remove_small = st.slider("Remove small objects", 0, 5000, 0)

    apply_largest = st.checkbox("가장 큰 병변만 유지하기 (Largest Component)", False)
    smooth_k = st.slider("Smoothing 강도 (Gaussian kernel radius)", 0, 25, 0)

    mask = postprocess(base_mask.copy(), open_k=open_k, close_k=close_k,
                       fill=fill_holes, remove_small_val=remove_small)

    if apply_largest:
        mask = largest_component(mask)

    if smooth_k > 0:
        ksize = smooth_k * 2 + 1
        blur = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        _, mask = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    # ================ 오버레이 설정 =================
    st.subheader("오버레이 설정")
    color = st.color_picker("오버레이 색상", "#FF0000")
    r, g, b = ImageColor.getcolor(color, "RGB")
    alpha = st.slider("오버레이 투명도", 0.0, 1.0, 0.4)

    # resize mask to original size
    mask_up = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # overlay
    overlay_up = img_np_orig.copy()
    overlay_up[mask_up > 0] = [r, g, b]
    overlay_up = (img_np_orig * (1 - alpha) + overlay_up * alpha).astype(np.uint8)

    # contour
    contours_up = cv2.findContours(mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour_img = img_np_orig.copy()
    cv2.drawContours(contour_img, contours_up, -1, (r, g, b), 2)

    contour_overlay = overlay_up.copy()
    cv2.drawContours(contour_overlay, contours_up, -1, (r, g, b), 2)

    # heatmap
    heat = (cm.jet(pred)[...,:3] * 255).astype(np.uint8)
    heat_up = cv2.resize(heat, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # crop
    crop_img = auto_crop(img_np_orig, mask_up, pad=10)

    # ================= 출력 =================
    st.subheader("결과 (원본 해상도)")
    c1, c2, c3 = st.columns(3)
    c1.image(img_np_orig, caption="원본 (Original)")
    c2.image(mask_up, caption="마스크 (원본 크기)")
    c3.image(overlay_up, caption="오버레이 (원본 크기)")

    st.subheader("윤곽선 / 합성")
    cc1, cc2 = st.columns(2)
    cc1.image(contour_img, caption="윤곽선 (Contour)")
    cc2.image(contour_overlay, caption="오버레이 + 윤곽선")

    st.subheader("Heatmap (모델 신뢰도)")
    st.image(heat_up, caption="Prediction Heatmap")

    st.subheader("자동 크롭 (병변만)")
    st.image(crop_img, caption="Cropped Lesion")

    st.subheader("Before / After 비교")
    w = orig_w
    slider_pos = st.slider("슬라이더 위치 (가로)", 0, w, w//2)
    before_after = img_np_orig.copy()
    before_after[:, :slider_pos] = img_np_orig[:, :slider_pos]
    before_after[:, slider_pos:] = overlay_up[:, slider_pos:]
    st.image(before_after, caption="Before(left) / After(right)")

    # ================= 다운로드 기능 =================
    st.subheader("다운로드")

    def to_png_bytes(image_np):
        _, buf = cv2.imencode('.png', image_np)
        return buf.tobytes()

    overlay_bytes = to_png_bytes(overlay_up)
    st.download_button("오버레이 다운로드 (PNG)", overlay_bytes, "overlay.png")

    _, buf_mask = cv2.imencode(".png", mask_up)
    st.download_button("마스크 다운로드 (PNG)", buf_mask.tobytes(), "mask.png")

    contour_bytes = to_png_bytes(contour_img)
    st.download_button("윤곽선 다운로드 (PNG)", contour_bytes, "contour.png")

    contour_overlay_bytes = to_png_bytes(contour_overlay)
    st.download_button("경계선+오버레이 다운로드 (PNG)", contour_overlay_bytes, "contour_overlay.png")

    transparent_mask_bytes = mask_to_transparent_png(mask_up)
    st.download_button("투명 배경 마스크 다운로드 (PNG)", transparent_mask_bytes, "mask_transparent.png")

    heat_bytes = to_png_bytes(heat_up)
    st.download_button("Heatmap 다운로드 (PNG)", heat_bytes, "heatmap.png")

    _, buf_crop = cv2.imencode(".png", crop_img)
    st.download_button("크롭된 병변 다운로드 (PNG)", buf_crop.tobytes(), "crop.png")

    # 전체 스크립트 다운로드
    st.subheader("앱 소스 코드 다운로드")
    try:
        script_path = sys.argv[0]
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
    except:
        script_content = "# Unable to read script."

    st.download_button("전체 스크립트 다운로드 (.py)",
                      data=script_content,
                      file_name="app.py",
                      mime="text/x-python")

else:
    st.info("이미지를 업로드하세요.")






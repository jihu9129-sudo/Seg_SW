# app_optimized_safe.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageColor
import numpy as np
import cv2
from io import BytesIO
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes

# ======================= 설정 ==========================
IMG_SIZE = 256           # 모델 입력 사이즈
PREVIEW_SIZE = 128       # 화면 출력용 축소
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================= 유틸리티 =======================
def to_png_bytes(img):
    if img.ndim == 2:
        pil = Image.fromarray(img.astype(np.uint8), mode="L")
    else:
        pil = Image.fromarray(img.astype(np.uint8))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

# ======================= UNetImproved ===================
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
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

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

# ======================= 모델 로드 =====================
st.title("피부 병변 분할 SW (UNet 기반)")

uploaded_model = st.file_uploader("모델 파일 업로드 (.pth)", type=["pth"])
@st.cache_resource
def load_model(uploaded_file):
    model = UNetImproved().to(device)
    if uploaded_file is not None:
        state = torch.load(uploaded_file, map_location=device)
        model.load_state_dict(state, strict=False)
        st.success("모델 로드 완료")
    else:
        st.info("모델을 추가하세요")
    model.eval()
    return model

model = load_model(uploaded_model)

# ======================= 전처리 =====================
def hair_removal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    thresh = cv2.GaussianBlur(thresh, (3,3), 0)
    return cv2.inpaint(img, thresh, 1, cv2.INPAINT_TELEA)

def clahe_equalization(img, clip=2.0, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)

def gaussian_blur(img, ksize=3):
    if ksize%2==0: ksize+=1
    return cv2.GaussianBlur(img,(ksize,ksize),0)

def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    return np.clip(img*alpha + beta,0,255).astype(np.uint8)

@st.cache_data
def preprocess_image(img, hair, clahe, clip, tile, blur, ksize, bc, alpha, beta):
    img_proc = img.copy()
    if hair: img_proc = hair_removal(img_proc)
    if clahe: img_proc = clahe_equalization(img_proc, clip, tile)
    if blur: img_proc = gaussian_blur(img_proc, ksize)
    if bc: img_proc = adjust_brightness_contrast(img_proc, alpha, beta)
    return img_proc

# ======================= 후처리 =====================
def auto_crop(image, mask, pad=10):
    coords = np.column_stack(np.where(mask>0))
    if len(coords)==0: return image
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    y_min, x_min = max(y_min-pad,0), max(x_min-pad,0)
    y_max, x_max = min(y_max+pad,image.shape[0]), min(x_max+pad,image.shape[1])
    return image[y_min:y_max, x_min:x_max]

def largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels<=1: return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas)+1
    return (labels==largest_label).astype(np.uint8)

def smooth_mask(mask, median_k=9, min_size=200):
    m = (mask*255).astype(np.uint8)
    if median_k%2==0: median_k+=1
    m = cv2.medianBlur(m,max(3,median_k))
    m = binary_fill_holes(m>0).astype(np.uint8)*255
    m_bool = remove_small_objects(m>0,min_size)
    return m_bool.astype(np.uint8)

def apply_heatmap(prob_map):
    p = (prob_map*255).astype(np.uint8)
    heat = cv2.applyColorMap(p, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

def draw_contour(img, mask, color=(255,0,0), thick=2):
    cnts,_ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = img.copy()
    cv2.drawContours(img_copy, cnts, -1, color, thick)
    return img_copy

# ======================= Streamlit UI ===================
uploaded = st.file_uploader("이미지 업로드", type=["png","jpg","jpeg"])
if not uploaded:
    st.info("이미지를 업로드하세요.")
    st.stop()

pil = Image.open(uploaded).convert("RGB")
img_orig = np.array(pil)
h_orig, w_orig = img_orig.shape[:2]

st.subheader("전처리 옵션")
apply_hair = st.checkbox("털 제거",False)
apply_clahe = st.checkbox("CLAHE 적용",False)
clahe_clip = st.slider("Clip",1.0,10.0,2.0,disabled=not apply_clahe)
clahe_tile = st.slider("Tile",4,16,8,disabled=not apply_clahe)
apply_blur = st.checkbox("Gaussian Blur",False)
blur_ksize = st.slider("Kernel Size",1,21,3,disabled=not apply_blur)
apply_bc = st.checkbox("밝기/대비 조정",False)
alpha_bc = st.slider("Contrast α",0.5,3.0,1.0,disabled=not apply_bc)
beta_bc = st.slider("Brightness β",-100,100,0,disabled=not apply_bc)

img_proc = preprocess_image(img_orig, apply_hair, apply_clahe, clahe_clip, clahe_tile,
                            apply_blur, blur_ksize, apply_bc, alpha_bc, beta_bc)

# 모델 추론
img_resized = np.array(Image.fromarray(img_proc).resize((IMG_SIZE, IMG_SIZE)))

@st.cache_data
def predict_mask(img_array):
    x = transforms.ToTensor()(img_array).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.sigmoid(out).squeeze().cpu().numpy()
    return prob

prob = predict_mask(img_resized)

# 후처리
st.subheader("분할 임계값")
th = st.slider("Segmentation Threshold",0.0,1.0,0.5)
base_mask = (prob>th).astype(np.uint8)

st.subheader("후처리 옵션")
remove_small = st.slider("잡음 완화",0,5000,200)
smooth_k = st.slider("Smoothing",0,9,0)
apply_largest = st.checkbox("가장 큰 병변만 유지",False)

mask_bin = base_mask.copy()
if apply_largest: mask_bin = largest_component(mask_bin)
if smooth_k>0: mask_bin = smooth_mask(mask_bin, median_k=smooth_k if smooth_k%2==1 else smooth_k+1, min_size=remove_small)

mask_up = cv2.resize((mask_bin*255).astype(np.uint8),(w_orig,h_orig),interpolation=cv2.INTER_NEAREST)
mask_up_bin = (mask_up>0).astype(np.uint8)

# Overlay / Contour / Heatmap
st.subheader("오버레이 설정")
color_hex = st.color_picker("색상","#FF0000")
r,g,b = ImageColor.getcolor(color_hex,"RGB")
alpha_overlay = st.slider("투명도",0.0,1.0,0.4)

overlay = img_proc.copy()
overlay[mask_up_bin>0] = [r,g,b]
overlay = (img_proc*(1-alpha_overlay)+overlay*alpha_overlay).astype(np.uint8)
contour_img = draw_contour(img_proc.copy(), mask_up_bin, color=(r,g,b))
heat_up = cv2.resize(apply_heatmap(prob),(w_orig,h_orig),interpolation=cv2.INTER_LINEAR)

# Display
c1,c2,c3 = st.columns(3)
c1.image(img_proc, caption="원본", use_container_width=True)
c2.image(np.stack([mask_up]*3,axis=-1), caption="마스크", use_container_width=True)
c3.image(overlay, caption="오버레이", use_container_width=True)

cc1,cc2 = st.columns(2)
cc1.image(contour_img, caption="윤곽선", use_container_width=True)
cc2.image(heat_up, caption="Heatmap", use_container_width=True)

# Cropping
st.subheader("크롭 이미지")
crop_overlay_on = st.checkbox("크롭 오버레이 적용", False)
crop_contour_on = st.checkbox("크롭 윤곽선 적용", False)
crop_img = auto_crop(img_proc, mask_up_bin, pad=10)
mask_crop = auto_crop(mask_up_bin, mask_up_bin, pad=10)
crop_display = crop_img.copy()
if crop_overlay_on:
    tmp = crop_display.copy()
    tmp[mask_crop>0] = [r,g,b]
    tmp = (crop_display*(1-alpha_overlay)+tmp*alpha_overlay).astype(np.uint8)
    crop_display = tmp
if crop_contour_on:
    crop_display = draw_contour(crop_display, mask_crop, color=(r,g,b))
st.image(crop_display, caption="크롭 영역", use_container_width=True)

# Downloads
st.subheader("다운로드")
st.download_button("마스크", to_png_bytes(mask_up), "mask.png")
st.download_button("오버레이", to_png_bytes(overlay), "overlay.png")
st.download_button("윤곽선", to_png_bytes(contour_img), "contour.png")
st.download_button("Heatmap", to_png_bytes(heat_up), "heatmap.png")
st.download_button("크롭 이미지", to_png_bytes(crop_display), "crop.png")


#streamlit run C:\Users\PC\PycharmProjects\PythonProject1\.venv\app.py

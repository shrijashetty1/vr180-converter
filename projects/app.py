import streamlit as st
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="2D to VR 180 Converter", layout="wide")

st.title("🎥 2D to VR 180 Video Converter (Prototype)")
st.write("Upload a video and preview the AI-based VR conversion simulation.")

# Upload video
uploaded_file = st.file_uploader("📂 Upload your 2D video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    ret, frame = cap.read()
    cap.release()

    if ret:
        st.subheader("📸 Original Frame")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Convert frame
        st.subheader("🤖 AI Depth Estimation (Sample)")
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        tensor_image = transform(pil_image).unsqueeze(0)

        
        depth_map = torch.mean(tensor_image, dim=1).squeeze().detach().numpy()
        st.image(depth_map, caption="Depth Map (Simulated)", use_container_width=True)

        
        st.subheader("🕶 VR 180 Preview (Simulated)")
        vr_sample_path = "sample_video.mp4"  

        if os.path.exists(vr_sample_path):
            st.video(vr_sample_path)
        else:
            st.warning("⚠️ VR sample video not found. Please place 'sample_video.mp4' in the same folder as app.py.")

        
        with open(vr_sample_path, "rb") as file:
            st.download_button(
                label="⬇️ Download VR 180 Video",
                data=file,
                file_name="converted_vr180.mp4",
                mime="video/mp4"
            )

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import io
import rawpy
import cv2
import torch
from rembg import remove
import mediapipe as mp
import requests
from io import BytesIO
import logging
from streamlit_drawable_canvas import st_canvas
from diffusers import StableDiffusionInpaintPipeline
import clip
import tempfile
import os
import insightface
from insightface.app import FaceAnalysis
from PIL.ExifTags import TAGS
import exifread

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Initialize CLIP for content moderation
@st.cache_resource
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    return model, preprocess

# Initialize Stable Diffusion Inpainting
@st.cache_resource
def load_inpainting_model():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        use_auth_token=False
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# Initialize InsightFace for face swapping
@st.cache_resource
def load_insightface_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

# Content moderation with CLIP
def moderate_content(image):
    model, preprocess = load_clip_model()
    image_tensor = preprocess(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    text = clip.tokenize(["explicit content", "dermatological skin", "neutral content"]).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text)
        logits_per_image, _ = model(image_tensor, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return probs[0][0] < 0.5 or probs[0][1] > 0.4  # Allow dermatological or neutral content

# Explicit inpainting for dermatology
def inpaint_image(image, mask, prompt, skin_type="medium", condition="healthy"):
    try:
        pipe = load_inpainting_model()
        img_array = np.array(image)
        mask_array = np.array(mask)[:, :, 0]  # Convert mask to grayscale
        mask_array = cv2.resize(mask_array, (img_array.shape[1], img_array.shape[0]))
        mask_image = Image.fromarray(mask_array)
        
        # Craft dermatology-specific prompt
        full_prompt = f"{condition} human skin, {skin_type} tone, realistic texture, high detail"
        negative_prompt = "explicit content, nudity, unnatural artifacts"
        
        # Perform inpainting
        result = pipe(
            prompt=full_prompt,
            image=image,
            mask_image=mask_image,
            strength=0.99,
            guidance_scale=7.5,
            negative_prompt=negative_prompt
        ).images[0]
        return result
    except Exception as e:
        logger.error(f"Error in inpainting: {str(e)}")
        return image

# Advanced face swapping with InsightFace
def face_swap(image, source_face_image):
    try:
        app = load_insightface_model()
        img_array = np.array(image)
        source_face_array = np.array(source_face_image)
        
        faces_target = app.get(img_array)
        faces_source = app.get(source_face_array)
        
        if len(faces_target) > 0 and len(faces_source) > 0:
            swapper = insightface.model_zoo.get_model("inswapper_128.onnx")
            result = img_array.copy()
            for face in faces_target:
                result = swapper.get(result, face, faces_source[0])
            return Image.fromarray(result)
        else:
            st.warning("No faces detected in target or source image.")
            return image
    except Exception as e:
        logger.error(f"Error in face swap: {str(e)}")
        return image

# Basic skin analysis (texture variance)
def analyze_skin(image):
    img_array = np.array(image.convert('L'))  # Convert to grayscale
    texture_variance = np.var(img_array)
    return {"texture_variance": texture_variance, "description": "Higher variance indicates more textured skin."}

# Brightness/Contrast adjustment
def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(contrast)

# Add text annotation with metadata
def add_text(image, text, position=(10, 10), font_size=20):
    draw = ImageDraw.Draw(image)
    draw.text(position, text, fill="white", stroke_width=2, stroke_fill="black")
    return image

# Add watermark
def add_watermark(image):
    draw = ImageDraw.Draw(image)
    draw.text((10, image.size[1] - 30), "AI-Generated (Dermatology Research)", fill="white", stroke_width=1, stroke_fill="black")
    return image

# Process NEF (RAW) files with metadata
def process_nef(file):
    try:
        with rawpy.imread(file) as raw:
            rgb = raw.postprocess()
        image = Image.fromarray(rgb)
        # Extract EXIF metadata
        with open(file, 'rb') as f:
            tags = exifread.process_file(f)
        metadata = {TAGS.get(tag): str(value) for tag, value in tags.items()}
        return image, metadata
    except Exception as e:
        logger.error(f"Error processing NEF file: {str(e)}")
        raise

# Background removal using U^2-Net
def remove_background(image):
    try:
        img_array = np.array(image)
        output = remove(img_array)
        return Image.fromarray(output)
    except Exception as e:
        logger.error(f"Error removing background: {str(e)}")
        raise

# Torso cropping using MediaPipe
def crop_torso(image):
    try:
        img_array = np.array(image)
        results = pose.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            height, width = img_array.shape[:2]
            shoulder_y = min(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) * height
            hip_y = max(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) * height
            return image.crop((0, shoulder_y, width, hip_y))
        else:
            st.warning("No person detected. Using default crop.")
            return image.crop((0, height * 0.25, width, height * 0.75))
    except Exception as e:
        logger.error(f"Error cropping torso: {str(e)}")
        raise

# Streamlit app
def main():
    st.title("Dermatology Research Image Manipulation App")
    st.write("Upload an image for AI-powered editing or dermatological research. All images are deleted after processing.")

    # Terms of use agreement
    agree = st.checkbox("I agree to use this app ethically for personal use and dermatological research, not for harmful or illegal purposes.")
    if not agree:
        st.error("You must agree to the terms of use to proceed.")
        return

    # File uploader for images
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
        uploaded_file = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png", "nef", "tiff", "bmp"]
        )
        if uploaded_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            metadata = {}
            
            # Load and compress image
            if file_extension == 'nef':
                image, metadata = process_nef(tmp_file_path)
            else:
                image = Image.open(tmp_file_path).convert('RGB')
                with open(tmp_file_path, 'rb') as f:
                    tags = exifread.process_file(f)
                    metadata = {TAGS.get(tag): str(value) for tag, value in tags.items()}
            
            # Compress image for performance
            image = image.resize((min(image.size[0], 512), min(image.size[1], 512)))

            # Content moderation
            if not moderate_content(image):
                st.error("Image content deemed inappropriate for dermatological research.")
                return

            # Display original image
            st.image(image, caption="Original Image", use_column_width=True)

            # Manipulation options
            manipulation_type = st.selectbox(
                "Select Manipulation Type",
                ["None", "Remove Background", "Crop Torso", "Dermatology Inpainting", "Manual Face Swap", 
                 "Manual Object Removal", "Adjust Brightness/Contrast", "Add Text", "Skin Analysis"]
            )

            processed_image = image

            if manipulation_type != "None":
                # Undo/redo support
                if 'history' not in st.session_state:
                    st.session_state.history = []
                    st.session_state.current_index = -1
                
                def save_to_history(img):
                    st.session_state.history = st.session_state.history[:st.session_state.current_index + 1]
                    st.session_state.history.append(img)
                    st.session_state.current_index += 1

                if manipulation_type == "Dermatology Inpainting":
                    st.write("Draw on the image to select the area for skin inpainting.")
                    skin_type = st.selectbox("Select Skin Type (Fitzpatrick Scale)", 
                                            ["Type I (Very Light)", "Type II (Light)", "Type III (Medium)", 
                                             "Type IV (Olive)", "Type V (Brown)", "Type VI (Dark)"])
                    condition = st.selectbox("Select Skin Condition", 
                                            ["Healthy", "Scar Tissue", "Pigmented", "Lesion", "Eczema", "Psoriasis"])
                    texture_prompt = st.text_input("Custom Texture Prompt (optional)", "")
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)",
                        stroke_width=3,
                        stroke_color="black",
                        background_image=image,
                        height=image.size[1],
                        width=image.size[0],
                        drawing_mode="freedraw",
                        key="canvas"
                    )
                    if canvas_result.image_data is not None:
                        mask = Image.fromarray(canvas_result.image_data)
                        with st.spinner("Processing dermatology inpainting..."):
                            skin_type_map = {
                                "Type I (Very Light)": "very light skin",
                                "Type II (Light)": "light skin",
                                "Type III (Medium)": "medium skin",
                                "Type IV (Olive)": "olive skin",
                                "Type V (Brown)": "brown skin",
                                "Type VI (Dark)": "dark skin"
                            }
                            full_prompt = texture_prompt or f"{condition.lower()} {skin_type_map[skin_type]}"
                            processed_image = inpaint_image(image, mask, full_prompt, skin_type_map[skin_type], condition.lower())
                            processed_image = add_watermark(processed_image)
                            save_to_history(processed_image)
                
                elif manipulation_type == "Manual Face Swap":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_face_file:
                        source_face_file = st.file_uploader("Choose source face image", type=["jpg", "jpeg", "png"])
                        if source_face_file:
                            tmp_face_file.write(source_face_file.read())
                            source_face_image = Image.open(tmp_face_file.name).convert('RGB')
                            source_face_image = source_face_image.resize((min(source_face_image.size[0], 512), min(source_face_image.size[1], 512)))
                            if moderate_content(source_face_image):
                                with st.spinner("Processing face swap..."):
                                    processed_image = face_swap(image, source_face_image)
                                    processed_image = add_watermark(processed_image)
                                    save_to_history(processed_image)
                            else:
                                st.error("Source face image deemed inappropriate.")
                            os.unlink(tmp_face_file.name)
                
                elif manipulation_type == "Manual Object Removal":
                    st.write("Draw on the image to select the area to remove.")
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)",
                        stroke_width=3,
                        stroke_color="black",
                        background_image=image,
                        height=image.size[1],
                        width=image.size[0],
                        drawing_mode="freedraw",
                        key="canvas_obj"
                    )
                    if canvas_result.image_data is not None:
                        mask = Image.fromarray(canvas_result.image_data)
                        with st.spinner("Processing object removal..."):
                            processed_image = inpaint_image(image, mask, "background")
                            processed_image = add_watermark(processed_image)
                            save_to_history(processed_image)
                
                elif manipulation_type == "Remove Background":
                    with st.spinner("Processing..."):
                        processed_image = remove_background(image)
                        processed_image = add_watermark(processed_image)
                        save_to_history(processed_image)
                
                elif manipulation_type == "Crop Torso":
                    with st.spinner("Processing..."):
                        processed_image = crop_torso(image)
                        processed_image = add_watermark(processed_image)
                        save_to_history(processed_image)
                
                elif manipulation_type == "Adjust Brightness/Contrast":
                    brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
                    contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
                    with st.spinner("Adjusting..."):
                        processed_image = adjust_brightness_contrast(image, brightness, contrast)
                        processed_image = add_watermark(processed_image)
                        save_to_history(processed_image)
                
                elif manipulation_type == "Add Text":
                    text = st.text_input("Enter text to add", "Sample Text")
                    x = st.slider("Text X Position", 0, image.size[0], 10)
                    y = st.slider("Text Y Position", 0, image.size[1], 10)
                    font_size = st.slider("Font Size", 10, 100, 20)
                    with st.spinner("Adding text..."):
                        processed_image = add_text(image, text, (x, y), font_size)
                        processed_image = add_watermark(processed_image)
                        save_to_history(processed_image)
                
                elif manipulation_type == "Skin Analysis":
                    with st.spinner("Analyzing skin..."):
                        analysis = analyze_skin(image)
                        st.write("Skin Analysis Results:")
                        st.json(analysis)
                        processed_image = image  # No change to image, just analysis
                        save_to_history(processed_image)

                # Undo/Redo buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Undo") and st.session_state.current_index > 0:
                        st.session_state.current_index -= 1
                        processed_image = st.session_state.history[st.session_state.current_index]
                with col2:
                    if st.button("Redo") and st.session_state.current_index < len(st.session_state.history) - 1:
                        st.session_state.current_index += 1
                        processed_image = st.session_state.history[st.session_state.current_index]

                # Display processed image
                st.image(processed_image, caption=f"{manipulation_type} Result", use_column_width=True)

                # Download processed image with metadata
                buffered = io.BytesIO()
                processed_image.save(buffered, format="PNG")
                st.download_button(
                    label="Download Processed Image",
                    data=buffered.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )
                
                # Display metadata
                if metadata:
                    st.write("Image Metadata:")
                    st.json(metadata)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Error in main: {str(e)}")
        finally:
            # Delete temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()

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
import clip as openai_clip
import tempfile
import os
import insightface
from insightface.app import FaceAnalysis
import exifread
from PIL.ExifTags import TAGS
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Initialize CLIP for content moderation
@st.cache_resource
def load_clip_model():
    try:
        model, preprocess = openai_clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        return model, preprocess
    except Exception as e:
        logger.error(f"Error loading CLIP model: {str(e)}")
        raise

# Initialize Stable Diffusion Inpainting
@st.cache_resource
def load_inpainting_model():
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            use_auth_token=False
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        return pipe
    except Exception as e:
        logger.error(f"Error loading inpainting model: {str(e)}")
        raise

# Initialize InsightFace for face swapping and expression modification
@st.cache_resource
def load_insightface_model():
    try:
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception as e:
        logger.error(f"Error loading InsightFace model: {str(e)}")
        raise

# Content moderation with CLIP
def moderate_content(image):
    model, preprocess = load_clip_model()
    image_tensor = preprocess(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    text = openai_clip.tokenize(["explicit content", "dermatological skin", "neutral face", "emotional expression"]).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text)
        logits_per_image, _ = model(image_tensor, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    logger.info(f"Content moderation probabilities: {probs}")
    return probs[0][0] < 0.5 or probs[0][1] > 0.3 or probs[0][2] > 0.2 or probs[0][3] > 0.2  # Allow dermatological or emotional content

# Explicit inpainting for medical and social-emotional contexts
def inpaint_image(image, mask, prompt, skin_type="medium", context="medical", condition="healthy"):
    try:
        pipe = load_inpainting_model()
        img_array = np.array(image)
        mask_array = np.array(mask)[:, :, 0]  # Convert mask to grayscale
        mask_array = cv2.resize(mask_array, (img_array.shape[1], img_array.shape[0]))
        mask_image = Image.fromarray(mask_array)
        
        # Craft context-specific prompt
        if context == "medical":
            full_prompt = f"{condition.lower()} human skin, {skin_type} tone, realistic texture, high detail"
            negative_prompt = "explicit content, nudity, unnatural artifacts, non-medical"
        else:  # social-emotional
            full_prompt = f"{condition.lower()} facial expression, {skin_type} tone, realistic human face"
            negative_prompt = "explicit content, distorted face, unnatural artifacts"
        
        if prompt:
            full_prompt = prompt
        
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

# Advanced face swapping or expression modification with InsightFace
def face_swap_or_expression(image, source_image=None, expression=None):
    try:
        app = load_insightface_model()
        img_array = np.array(image)
        if source_image:
            source_array = np.array(source_image)
            faces_source = app.get(source_array)
        else:
            faces_source = []
        
        faces_target = app.get(img_array)
        
        if len(faces_target) > 0:
            if expression:
                # Placeholder for expression modification via inpainting
                pipe = load_inpainting_model()
                face = faces_target[0]
                bbox = face.bbox
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
                mask = np.zeros_like(img_array)
                mask[y:y+h, x:x+w] = 255
                mask_image = Image.fromarray(mask[:, :, 0])
                result = pipe(
                    prompt=f"{expression} facial expression, realistic human face",
                    image=image,
                    mask_image=mask_image,
                    strength=0.8,
                    guidance_scale=7.5,
                    negative_prompt="distorted face, unnatural artifacts"
                ).images[0]
                return result
            elif len(faces_source) > 0:
                swapper = insightface.model_zoo.get_model("inswapper_128.onnx")
                result = img_array.copy()
                for face in faces_target:
                    result = swapper.get(result, face, faces_source[0])
                return Image.fromarray(result)
            else:
                st.warning("No source face or expression provided.")
                return image
        else:
            st.warning("No faces detected in target image.")
            return image
    except Exception as e:
        logger.error(f"Error in face swap/expression: {str(e)}")
        return image

# Enhanced skin analysis for dermatology
def analyze_skin(image):
    img_array = np.array(image.convert('L'))  # Grayscale for texture
    color_array = np.array(image)  # RGB for color analysis
    texture_variance = np.var(img_array)
    color_mean = np.mean(color_array, axis=(0, 1))
    return {
        "texture_variance": texture_variance,
        "color_mean_rgb": color_mean.tolist(),
        "description": "Texture variance indicates skin roughness; RGB mean reflects average skin tone."
    }

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
    draw.text((10, image.size[1] - 30), "AI-Generated (Medical/Social-Emotional Research)", fill="white", stroke_width=1, stroke_fill="black")
    return image

# Process NEF (RAW) files with metadata
def process_nef(file):
    try:
        with rawpy.imread(file) as raw:
            rgb = raw.postprocess()
        image = Image.fromarray(rgb)
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
    st.title("Medical and Social-Emotional Image Manipulation App")
    st.write("Upload an image for AI-powered editing in medical or social-emotional research contexts. All images are deleted after processing.")

    # Terms of use agreement
    agree = st.checkbox("I agree to use this app ethically for personal use, medical research (e.g., dermatology), or social-emotional research, not for harmful or illegal purposes.")
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
                st.error("Image content deemed inappropriate for research.")
                return

            # Display original image
            st.image(image, caption="Original Image", use_container_width=True)

            # Manipulation options
            manipulation_type = st.selectbox(
                "Select Manipulation Type",
                ["None", "Remove Background", "Crop Torso", "Dermatology Inpainting", "Social-Emotional Inpainting", 
                 "Manual Face Swap", "Expression Modification", "Manual Object Removal", "Adjust Brightness/Contrast", 
                 "Add Text", "Skin Analysis"]
            )

            processed_image = image
            annotations = []

            if manipulation_type != "None":
                # Undo/redo support
                if 'history' not in st.session_state:
                    st.session_state.history = []
                    st.session_state.current_index = -1
                
                def save_to_history(img):
                    st.session_state.history = st.session_state.history[:st.session_state.current_index + 1]
                    st.session_state.history.append(img)
                    st.session_state.current_index += 1

                if manipulation_type in ["Dermatology Inpainting", "Social-Emotional Inpainting"]:
                    context = "medical" if manipulation_type == "Dermatology Inpainting" else "social-emotional"
                    st.write(f"Draw on the image to select the area for {context} inpainting.")
                    if context == "medical":
                        skin_type = st.selectbox("Select Skin Type (Fitzpatrick Scale)", 
                                                ["Type I (Very Light)", "Type II (Light)", "Type III (Medium)", 
                                                 "Type IV (Olive)", "Type V (Brown)", "Type VI (Dark)"])
                        condition = st.selectbox("Select Skin Condition", 
                                                ["Healthy", "Scar Tissue", "Pigmented", "Lesion", "Eczema", "Psoriasis", "Melanoma"])
                    else:
                        skin_type = st.selectbox("Select Skin Tone", 
                                                ["Light", "Medium", "Dark"])
                        condition = st.selectbox("Select Expression", 
                                                ["Neutral", "Smiling", "Sad", "Angry", "Surprised"])
                    texture_prompt = st.text_input("Custom Prompt (optional)", "")
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
                        with st.spinner(f"Processing {context} inpainting..."):
                            skin_type_map = {
                                "Type I (Very Light)": "very light skin",
                                "Type II (Light)": "light skin",
                                "Type III (Medium)": "medium skin",
                                "Type IV (Olive)": "olive skin",
                                "Type V (Brown)": "brown skin",
                                "Type VI (Dark)": "dark skin",
                                "Light": "light skin",
                                "Medium": "medium skin",
                                "Dark": "dark skin"
                            }
                            full_prompt = texture_prompt or f"{condition.lower()} {skin_type_map[skin_type]}"
                            processed_image = inpaint_image(image, mask, full_prompt, skin_type_map[skin_type], context, condition.lower())
                            processed_image = add_watermark(processed_image)
                            save_to_history(processed_image)
                            annotations.append({"type": manipulation_type, "prompt": full_prompt, "skin_type": skin_type, "condition": condition})
                
                elif manipulation_type == "Manual Face Swap":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_face_file:
                        source_face_file = st.file_uploader("Choose source face image", type=["jpg", "jpeg", "png"])
                        if source_face_file:
                            tmp_face_file.write(source_face_file.read())
                            source_face_image = Image.open(tmp_face_file.name).convert('RGB')
                            source_face_image = source_face_image.resize((min(source_face_image.size[0], 512), min(source_face_image.size[1], 512)))
                            if moderate_content(source_face_image):
                                with st.spinner("Processing face swap..."):
                                    processed_image = face_swap_or_expression(image, source_face_image)
                                    processed_image = add_watermark(processed_image)
                                    save_to_history(processed_image)
                                    annotations.append({"type": "Face Swap", "source_image": source_face_file.name})
                            else:
                                st.error("Source face image deemed inappropriate.")
                            os.unlink(tmp_face_file.name)
                
                elif manipulation_type == "Expression Modification":
                    expression = st.selectbox("Select Target Expression", 
                                            ["Neutral", "Smiling", "Sad", "Angry", "Surprised"])
                    with st.spinner("Processing expression modification..."):
                        processed_image = face_swap_or_expression(image, expression=expression)
                        processed_image = add_watermark(processed_image)
                        save_to_history(processed_image)
                        annotations.append({"type": "Expression Modification", "expression": expression})
                
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
                            processed_image = inpaint_image(image, mask, "background", context="medical")
                            processed_image = add_watermark(processed_image)
                            save_to_history(processed_image)
                            annotations.append({"type": "Object Removal"})
                
                elif manipulation_type == "Remove Background":
                    with st.spinner("Processing..."):
                        processed_image = remove_background(image)
                        processed_image = add_watermark(processed_image)
                        save_to_history(processed_image)
                        annotations.append({"type": "Background Removal"})
                
                elif manipulation_type == "Crop Torso":
                    with st.spinner("Processing..."):
                        processed_image = crop_torso(image)
                        processed_image = add_watermark(processed_image)
                        save_to_history(processed_image)
                        annotations.append({"type": "Torso Crop"})
                
                elif manipulation_type == "Adjust Brightness/Contrast":
                    brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
                    contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
                    with st.spinner("Adjusting..."):
                        processed_image = adjust_brightness_contrast(image, brightness, contrast)
                        processed_image = add_watermark(processed_image)
                        save_to_history(processed_image)
                        annotations.append({"type": "Brightness/Contrast", "brightness": brightness, "contrast": contrast})
                
                elif manipulation_type == "Add Text":
                    text = st.text_input("Enter text to add", "Sample Text")
                    x = st.slider("Text X Position", 0, image.size[0], 10)
                    y = st.slider("Text Y Position", 0, image.size[1], 10)
                    font_size = st.slider("Font Size", 10, 100, 20)
                    with st.spinner("Adding text..."):
                        processed_image = add_text(image, text, (x, y), font_size)
                        processed_image = add_watermark(processed_image)
                        save_to_history(processed_image)
                        annotations.append({"type": "Text Annotation", "text": text, "position": [x, y], "font_size": font_size})
                
                elif manipulation_type == "Skin Analysis":
                    with st.spinner("Analyzing skin..."):
                        analysis = analyze_skin(image)
                        st.write("Skin Analysis Results:")
                        st.json(analysis)
                        processed_image = image  # No change to image
                        save_to_history(processed_image)
                        annotations.append({"type": "Skin Analysis", "results": analysis})

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
                st.image(processed_image, caption=f"{manipulation_type} Result", use_container_width=True)

                # Download processed image with metadata
                buffered = io.BytesIO()
                processed_image.save(buffered, format="PNG")
                st.download_button(
                    label="Download Processed Image",
                    data=buffered.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )

                # Download annotations as JSON
                if annotations:
                    annotation_data = {
                        "image_metadata": metadata,
                        "annotations": annotations,
                        "timestamp": "2025-08-04 08:11:00 CEST"
                    }
                    annotation_buffer = io.BytesIO(json.dumps(annotation_data, indent=2).encode('utf-8'))
                    st.download_button(
                        label="Download Annotations (JSON)",
                        data=annotation_buffer,
                        file_name="annotations.json",
                        mime="application/json"
                    )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Error in main: {str(e)}")
        finally:
            # Delete temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()

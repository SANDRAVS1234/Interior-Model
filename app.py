import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# -----------------------
# 1. Page configuration
# -----------------------
st.set_page_config(page_title="AI Interior Style Generator", layout="centered")
st.title("🏠 AI Interior Design Style Generator")
st.markdown(
    "Upload a room photo and choose a target interior design style. "
    "The app will reimagine your room in that style using a diffusion model."
)

# -----------------------
# 2. Load model (cached)
# -----------------------
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"  # or another model like "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_attention_slicing()
    return pipe




# -----------------------
# 3. Image upload section
# -----------------------
st.subheader("Upload Your Room Image 🖼️")
uploaded_file = st.file_uploader(
    "Choose a room image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

styles = [
    "Modern", "Minimalist", "Vintage", "Industrial",
    "Bohemian", "Scandinavian", "Traditional",
    "Contemporary", "Rustic", "Coastal"
]
target_style = st.selectbox("🎨 Choose Target Interior Style", styles)

generate_button = st.button("✨ Generate Styled Room")

# -----------------------
# 4. Generate new style
# -----------------------
if generate_button:
    if uploaded_file is None:
        st.warning("⚠️ Please upload a room image first.")
    elif pipe is None:
        st.error("Model not loaded properly. Check your path or GPU settings.")
    else:
        with st.spinner("🎨 Generating your new design... please wait."):
            # Open and preprocess image
            init_image = Image.open(uploaded_file).convert("RGB")
            init_image = init_image.resize((512, 512))

            prompt = f"A {target_style.lower()} style interior design of this room, photo-realistic, high quality."
            
            # Generate with diffusion model
            result = pipe(prompt=prompt, image=init_image, strength=0.8, guidance_scale=7.5)

            if "images" in result and len(result["images"]) > 0:
                styled_image = result["images"][0]
                st.image(styled_image, caption=f"{target_style} Style Room", use_column_width=True)
                st.success("✅ Style generation completed!")
            else:
                st.error("⚠️ No image generated. Please retry or check model configuration.")

# -----------------------
# 5. Sidebar Info
# -----------------------
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    "This demo fine-tunes Stable Diffusion to reimagine room photos in various interior design styles. "
    "Trained with LoRA on Pinterest-style interior datasets."
)





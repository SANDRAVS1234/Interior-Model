# app.py
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Interior Designer 🎨",
    layout="wide",
    page_icon="🏠"
)

# -----------------------------
# App Title
# -----------------------------
st.title("🏠 AI Interior Design Generator")
st.markdown("### Generate stunning interior designs from text prompts using Stable Diffusion!")

# -----------------------------
# Model Loading (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "./interior_model"  # Local model folder
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True  # ✅ Add this line
    ).to("cuda")
    pipe.enable_attention_slicing()
    return pipe


# -----------------------------
# UI Controls
# -----------------------------
room_types = ["Living Room", "Bedroom", "Kitchen", "Bathroom", "Office", "Dining Room"]
styles = ["Modern", "Minimalist", "Scandinavian", "Industrial", "Bohemian", "Traditional"]

col1, col2 = st.columns(2)
with col1:
    room = st.selectbox("Select Room Type:", room_types)
with col2:
    style = st.selectbox("Select Interior Style:", styles)

custom_prompt = st.text_input(
    "Add extra details (optional):",
    placeholder="e.g. with wooden flooring and indoor plants"
)

prompt = f"A {style.lower()} style {room.lower()} interior design {custom_prompt}"

steps = st.slider("Diffusion Steps", 10, 50, 25)
guidance = st.slider("Guidance Scale", 1.0, 10.0, 7.5)
generate_btn = st.button("✨ Generate Design")

# -----------------------------
# Image Generation
# -----------------------------
if generate_btn:
    with st.spinner("🎨 Generating your interior design... please wait..."):
        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance
        ).images[0]

        st.image(image, caption=f"Generated: {prompt}", use_container_width=True)

        # Save and offer download
        output_path = "generated_design.png"
        image.save(output_path)
        with open(output_path, "rb") as file:
            st.download_button(
                label="⬇️ Download Design",
                data=file,
                file_name="interior_design.png",
                mime="image/png"
            )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("💡 *Built with Stable Diffusion and Streamlit — by AI Interior Designer*")


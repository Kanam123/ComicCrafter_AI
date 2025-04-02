# Install required libraries
!pip install -q transformers diffusers torch matplotlib bitsandbytes gradio

# Import libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline
import torch
import gc
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import random

# Verify GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Hugging Face token for gated models (replace with your token)
hf_token = "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"

# Configure 4-bit quantization for text generation (only if CUDA is available)
quantization_config = None
if device == "cuda":
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    except Exception as e:
        print(f"Quantization not supported: {e}. Falling back to CPU.")
        device = "cpu"
        quantization_config = None

# Load the text generation model (Mistral 7B)
try:
    text_model_name = "mistralai/Mistral-7B-v0.1"
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_auth_token=hf_token)
    text_model = AutoModelForCausalLM.from_pretrained(
        text_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        use_auth_token=hf_token
    )
except Exception as e:
    print(f"Failed to load text model: {e}")
    # Fallback to a smaller model
    text_model_name = "gpt2"
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModelForCausalLM.from_pretrained(text_model_name).to(device)

# Load the image generation model (Stable Diffusion)
try:
    image_model_name = "runwayml/stable-diffusion-v1-5"
    image_pipe = StableDiffusionPipeline.from_pretrained(image_model_name, torch_dtype=torch.float16)
    image_pipe = image_pipe.to(device)
except Exception as e:
    print(f"Failed to load image model: {e}")
    raise RuntimeError("Image model could not be loaded. Please check your setup.")

# Function to generate text (story or dialogue)
def generate_text(prompt):
    try:
        with torch.no_grad():
            inputs = text_tokenizer(prompt, return_tensors="pt").to(device)
            outputs = text_model.generate(
                **inputs,
                max_length=300,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                pad_token_id=text_tokenizer.eos_token_id
            )
            generated_text = text_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean up
            del inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()

            return generated_text
    except Exception as e:
        print(f"Error generating text: {e}")
        return f"Error generating story: {str(e)}"

# Function to generate a modern-style image
def generate_modern_image(prompt):
    try:
        modern_prompt = (
            f"{prompt}, modern digital art, clean lines, contemporary style, "
            "smooth shading, vibrant colors, high detail, trending on artstation, "
            "ultra HD, 8k resolution"
        )
        with torch.no_grad():
            image = image_pipe(
                modern_prompt,
                height=512,
                width=512,
                num_inference_steps=30
            ).images[0]
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        # Create error placeholder image
        img = Image.new('RGB', (512, 512), color=(230, 240, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, 250), "Failed to generate image", fill=(50, 50, 50))
        return img

# Function to split story into 4 parts for panels
def split_story(story):
    sentences = [s.strip() for s in story.split('.') if s.strip()]
    if len(sentences) < 4:
        # If not enough sentences, duplicate some
        sentences = sentences * (4 // len(sentences) + 1)

    # Distribute sentences across 4 panels
    panel_texts = []
    for i in range(4):
        start = i * len(sentences) // 4
        end = (i + 1) * len(sentences) // 4
        panel_text = '. '.join(sentences[start:end]) + '.'
        panel_texts.append(panel_text)

    return panel_texts

# Function to create modern comic panel with text
def create_comic_panel(text, image, panel_size=(512, 512)):
    try:
        # Modern color scheme
        bg_color = (240, 245, 255)  # Very light blue
        text_bg_color = (255, 255, 255, 200)  # Semi-transparent white
        text_color = (30, 30, 30)  # Dark gray

        panel = Image.new("RGB", panel_size, bg_color)
        draw = ImageDraw.Draw(panel)

        # Resize and paste image (top 80% of panel)
        img_height = int(panel_size[1] * 0.80)
        panel.paste(image.resize((panel_size[0], img_height)), (0, 0))

        # Add text area (bottom 20%)
        text_area_height = panel_size[1] - img_height
        text_area = Image.new('RGBA', (panel_size[0], text_area_height), text_bg_color)
        panel.paste(text_area, (0, img_height), text_area)

        # Add text
        font = ImageFont.load_default()
        text_position = (10, img_height + 10)
        draw.text(text_position, text, fill=text_color, font=font)

        return panel
    except Exception as e:
        print(f"Error creating panel: {e}")
        error_img = Image.new('RGB', panel_size, color=(230, 240, 255))
        draw = ImageDraw.Draw(error_img)
        draw.text((50, 250), "Panel creation error", fill=(50, 50, 50))
        return error_img

# Main generation function
def generate_comic(story_prompt):
    try:
        # Generate story
        story = generate_text(f"Write a short story about: {story_prompt}")

        # Split into 4 parts
        panel_texts = split_story(story)

        # Generate panels
        panels = []
        for i, text in enumerate(panel_texts):
            image = generate_modern_image(text)
            panel = create_comic_panel(text, image)
            panels.append(panel)

        return story, *panels

    except Exception as e:
        print(f"Error in comic generation: {e}")
        error_img = Image.new('RGB', (512, 512), color=(230, 240, 255))
        draw = ImageDraw.Draw(error_img)
        draw.text((50, 250), "Generation error", fill=(50, 50, 50))
        return f"Error: {str(e)}", error_img, error_img, error_img, error_img

# Modern CSS styling
css = """
.gradio-container {
    max-width: 1200px !important;
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
}
.panel-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 15px;
    margin-top: 20px;
}
.panel {
    border: 1px solid #d1d9e6;
    border-radius: 12px;
    box-shadow: 8px 8px 16px #d1d9e6, -8px -8px 16px #ffffff;
    transition: transform 0.3s ease;
}
.panel:hover {
    transform: translateY(-5px);
}
.story-box {
    background: rgba(255, 255, 255, 0.9);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 25px;
    box-shadow: 4px 4px 10px rgba(0,0,0,0.05);
    border: 1px solid rgba(0,0,0,0.05);
}
h1 {
    color: #4a6baf;
    text-align: center;
    margin-bottom: 25px;
}
button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    transition: all 0.3s ease !important;
}
button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 14px rgba(0,0,0,0.1) !important;
}
.textbox {
    border-radius: 10px !important;
    border: 1px solid #d1d9e6 !important;
    box-shadow: inset 3px 3px 6px #d1d9e6, inset -3px -3px 6px #ffffff !important;
}
"""

# Gradio interface with modern theme
with gr.Blocks(css=css, title="Modern AI Comic Generator") as demo:
    gr.Markdown("""
    # 🚀 Modern AI Comic Generator
    Enter a story idea below and the AI will generate a sleek 4-panel comic strip!
    """)

    with gr.Row():
        with gr.Column():
            story_prompt = gr.Textbox(
                label="Story Prompt",
                placeholder="e.g., 'A robot who wants to be a chef'",
                lines=3,
                elem_classes=["textbox"]
            )
            generate_btn = gr.Button("Generate Comic", variant="primary")

    with gr.Row():
        story_output = gr.Textbox(
            label="Generated Story",
            interactive=False,
            elem_classes=["story-box"]
        )

    with gr.Row(elem_classes=["panel-container"]):
        panel_outputs = []
        for i in range(4):
            panel_outputs.append(
                gr.Image(
                    label=f"Panel {i+1}",
                    elem_classes=["panel"],
                    width=512,
                    height=512
                )
            )

    generate_btn.click(
        fn=generate_comic,
        inputs=story_prompt,
        outputs=[story_output] + panel_outputs
    )

# Launch the app
demo.launch(share=True)
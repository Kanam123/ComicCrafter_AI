Introduction:

The AI Comic Crafter is a web application that generates a 4-panel comic strip based on a user-provided story prompt. It utilizes advanced AI models for text and image generation, leveraging the Mistral 7B model for text and Stable Diffusion for image creation.


Key Features:

->Generate a short story based on a given prompt.
->Create retro cartoon-style images for each panel of the comic.
->Display the generated story and images in a user-friendly interface.


System Requirements:

Recommended Setup
To run this code locally, you need the following system setup:

->Operating System: Windows, macOS, or Linux
->GPU: NVIDIA GPU with CUDA support (recommended for better performance)
->RAM: At least 16 GB (32 GB recommended for larger models)
->Python: Version 3.7 or higher
->CUDA: Version compatible with your GPU (if using GPU acceleration)


Required Dependencies

The following libraries need to be installed, which can be listed in a requirements.txt file:
->transformers
->diffusers
->torch
->matplotlib
->bitsandbytes
->gradio


Running on Google Colab

1. Open Google Colab: Navigate to Google Colab.
2. Create a New Notebook: Click on File > New Notebook.
3. Install Dependencies: Run the following command in a new cell:
    !pip install -q -r requirements.txt
4. Copy the Code: Copy the entire code from this repository into a new cell in the Colab notebook.
5. Set Hugging Face Token: Replace the hf_token variable in the code with your Hugging Face token. Obtain a token by creating an account on Hugging Face.
6. Run the Code: Execute the cell containing the code. The Gradio interface will launch in a few moments.
7. Use the App: Enter a story prompt and click Generate Comic to view the generated comic strip.


Running Locally

1. Install Python: Ensure Python 3.7 or higher is installed. Download it from python.org.

2. Set Up a Virtual Environment (Optional): To manage dependencies, create a virtual environment:
    python -m venv comic_crafter_env
    source comic_crafter_env/bin/activate  # On Windows use `comic_crafter_env\Scripts\activate`

3. Install Dependencies: Use pip to install the required dependencies:
    pip install -r requirements.txt

4. Copy the Code: Save the provided code in a Python file (e.g., comic_crafter.py).

5. Set Hugging Face Token: Replace the hf_token variable in the script with your Hugging Face token.

6. Run the Code: Execute the Python script via terminal/command prompt:
    python comic_crafter.py

7. Launch the App: Once the script is running, a local server will start. Open the provided URL (usually http://127.0.0.1:7860) in a web browser to access the Gradio interface.

8. Use the App: Enter a story prompt and click Generate Comic to see the generated comic strip.


Troubleshooting & Tips

1. CUDA Errors: Ensure that your GPU drivers and CUDA toolkit are correctly installed and compatible with your PyTorch version.

2. Dependency Issues: If you encounter errors during library installation, try upgrading pip:
    pip install --upgrade pip
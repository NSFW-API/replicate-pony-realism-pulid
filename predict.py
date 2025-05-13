import json
import os
import subprocess
import time
import shutil
from typing import List

from cog import BasePredictor, Input, Path

from comfyui import ComfyUI  # pip-installed via cog.yaml

# ---- constants -------------------------------------------------------------
WORKFLOW_JSON = "pulid_workflow.json"
OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
ALL_DIRS = [OUTPUT_DIR, INPUT_DIR]

# ----------------------------------------------------------------------------

class Predictor(BasePredictor):
    def setup(self):
        # Create necessary directories
        for dir_path in ALL_DIRS:
            os.makedirs(dir_path, exist_ok=True)

        # Clone PuLID node if missing
        if not os.path.exists("ComfyUI/custom_nodes/PuLID_ComfyUI"):
            print("Cloning PuLID_ComfyUI custom node...")
            subprocess.check_call([
                "git", "clone", "--depth", "1",
                "https://github.com/cubiq/PuLID_ComfyUI.git",
                "ComfyUI/custom_nodes/PuLID_ComfyUI"
            ])

        # Create model directories
        models_dir = "ComfyUI/models"
        os.makedirs(models_dir, exist_ok=True)

        # Create PuLID directory
        pulid_dir = f"{models_dir}/pulid"
        os.makedirs(pulid_dir, exist_ok=True)
        os.makedirs(f"{pulid_dir}/eva_clip", exist_ok=True)

        # Create insightface and facexlib directories
        insightface_dir = f"{models_dir}/insightface/models/antelopev2"
        facexlib_dir = f"{models_dir}/facexlib"
        os.makedirs(insightface_dir, exist_ok=True)
        os.makedirs(facexlib_dir, exist_ok=True)

        # Download PuLID weights
        pulid_adapter = f"{pulid_dir}/ip-adapter_pulid_sdxl_fp16.safetensors"
        if not os.path.exists(pulid_adapter):
            print(f"Downloading PuLID weights to {pulid_adapter}...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/huchenlei/ipadapter_pulid/resolve/main/ip-adapter_pulid_sdxl_fp16.safetensors",
                pulid_adapter
            ])

        # Pre-download EVA02-CLIP model to avoid first-run latencyclear
        eva_model = f"{pulid_dir}/eva_clip/pytorch_model.bin"
        eva_config = f"{pulid_dir}/eva_clip/config.json"
        if not os.path.exists(eva_model):
            print(f"Downloading EVA02-CLIP model to {eva_model}...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/microsoft/LLM2CLIP-EVA02-L-14-336/resolve/main/pytorch_model.bin",
                eva_model
            ])

        if not os.path.exists(eva_config):
            print(f"Downloading EVA02-CLIP config to {eva_config}...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/microsoft/LLM2CLIP-EVA02-L-14-336/resolve/main/config.json",
                eva_config
            ])

        # Download facexlib parsing model
        facexlib_model = f"{facexlib_dir}/parsing_parsenet.pth"
        if not os.path.exists(facexlib_model):
            print(f"Downloading facexlib model to {facexlib_model}...")
            subprocess.check_call([
                "pget", "-vf",
                "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
                facexlib_model
            ])

        # Download AntelopeV2 face detection model
        antelopev2_model = f"{insightface_dir}/2d106det.onnx"
        if not os.path.exists(antelopev2_model):
            print("Downloading AntelopeV2 model...")
            # Since AntelopeV2 comes as a zip with multiple files, we need to download and extract it
            tmp_zip = "/tmp/antelopev2.zip"
            subprocess.check_call([
                "pget", "-vf",
                "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
                tmp_zip
            ])
            # Extract the zip file
            subprocess.check_call(["unzip", "-o", tmp_zip, "-d", f"{models_dir}/insightface/models"])
            # Fix directory structure - move files one level up if nested
            if os.path.exists(f"{insightface_dir}/antelopev2"):
                # Move all files from nested directory up one level
                os.system(f"mv {insightface_dir}/antelopev2/* {insightface_dir}/")
                # Remove the now-empty directory
                os.system(f"rmdir {insightface_dir}/antelopev2 || true")
            # Clean up
            os.remove(tmp_zip)

        # SDXL checkpoint will be handled by weights_manifest.py with a custom mapping
        # No need to download it here

        # Download SDXL VAE
        vae_dir = f"{models_dir}/vae"
        os.makedirs(vae_dir, exist_ok=True)

        vae_path = f"{vae_dir}/sdxl_vae.safetensors"
        if not os.path.exists(vae_path):
            print(f"Downloading SDXL VAE to {vae_path}...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
                vae_path
            ])

        # Download T5 clip for SDXL
        clip_dir = f"{models_dir}/clip"
        os.makedirs(clip_dir, exist_ok=True)

        clip_path = f"{clip_dir}/t5xxl_fp16.safetensors"
        if not os.path.exists(clip_path):
            print(f"Downloading CLIP to {clip_path}...")
            subprocess.check_call([
                "pget", "-vf",
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
                clip_path
            ])

        # Start ComfyUI server
        print("Starting ComfyUI server...")
        self.comfy = ComfyUI("127.0.0.1:8188")
        self.comfy.start_server(OUTPUT_DIR, INPUT_DIR)

        # Wait for server to be ready
        max_retries = 30
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.comfy.connect()
                print("ComfyUI server is ready")
                break
            except Exception as e:
                print(f"Waiting for ComfyUI server to be ready... {e}")
                time.sleep(1)
                retry_count += 1

        if retry_count == max_retries:
            raise RuntimeError("Failed to connect to ComfyUI server")

    # ---- helpers -----------------------------------------------------------
    def _nearest_multiple(self, x: int, k: int = 8) -> int:
        return ((x + k - 1) // k) * k

    # -----------------------------------------------------------------------

    def predict(
            self,
            prompt: str = Input(description="Main text prompt."),
            negative_prompt: str = Input(default="", description="Optional negative."),
            reference_image: Path = Input(
                description="An image containing a face that you want to use as reference for face swapping.",
                default=None,
            ),
            width: int = Input(default=512, ge=64, le=1536),
            height: int = Input(default=512, ge=64, le=1536),
            steps: int = Input(default=30, ge=1, le=150),
            cfg: float = Input(default=3.0, ge=1.0, le=20.0),
            sampler_name: str = Input(default="euler_ancestral", choices=["euler", "euler_ancestral", "heun", "dpmpp_2s_ancestral", "uni_pc"]),
            scheduler: str = Input(default="normal", choices=["beta", "normal"]),
            seed: int = Input(default=0, description="0 = random"),
            method: str = Input(
                default="fidelity",
                choices=["fidelity", "style", "both"],
                description="PuLID method to use: fidelity for face swapping, style for style transfer, both for a mix.",
            ),
            face_weight: float = Input(
                default=0.8,
                ge=0.0,
                le=1.0,
                description="Weight of the face adaptation effect (0.0 to 1.0)",
            ),
    ) -> List[Path]:

        # 1. housekeeping
        self.comfy.cleanup(ALL_DIRS)
        if seed == 0: seed = int.from_bytes(os.urandom(2), "big")

        # If reference image is provided, copy it to input directory
        if reference_image:
            reference_path = os.path.join(INPUT_DIR, "reference.png")
            shutil.copy2(reference_image, reference_path)
        else:
            # If no reference image, we can't use PuLID effectively
            print("Warning: No reference image provided. PuLID requires a reference face image to work properly.")
            # We'll still run but results may not show face swap

        # 2. Load workflow.json
        with open(WORKFLOW_JSON) as f:
            wf = json.load(f)

        # 3. Update workflow with user inputs
        if "nodes" in wf:  # Style B
            by_id = {str(n["id"]): n for n in wf["nodes"]}
        else:  # Style A
            by_id = wf

        def node(idx: int):
            """Return the node dict for a given numeric id"""
            return by_id[str(idx)]

        # ----- prompt nodes -------------------------------------------------

        node(4)["inputs"]["text"] = prompt
        node(5)["inputs"]["text"] = negative_prompt

        # ----- latent size --------------------------------------------------
        latent_inputs = node(6)["inputs"]
        latent_inputs["width"] = self._nearest_multiple(width)
        latent_inputs["height"] = self._nearest_multiple(height)
        latent_inputs["batch_size"] = 1

        # ----- sampler settings --------------------------------------------
        sampler_inputs = node(7)["inputs"]
        sampler_inputs["seed"] = seed
        sampler_inputs["steps"] = steps
        sampler_inputs["cfg"] = cfg
        sampler_inputs["sampler_name"] = sampler_name
        sampler_inputs["scheduler"] = scheduler
        sampler_inputs["denoise"] = 1.0

        # ----- PuLID settings ----------------------------------------------
        # Update the ApplyPulid node with the user's method and weight
        if str(15) in by_id:
            node(15)["inputs"]["method"] = method
            node(15)["inputs"]["weight"] = face_weight

        # Make sure the ImageLoad node points to the reference image
        if str(17) in by_id:
            node(17)["inputs"]["image"] = "reference.png"

        # 4. Run the workflow
        print("Loading workflow...")
        wf_loaded = self.comfy.load_workflow(wf)
        print("Running workflow...")
        self.comfy.run_workflow(wf_loaded)

        # 5. Get the output images
        print("Getting output files...")
        all_files = self.comfy.get_files(OUTPUT_DIR)
        image_files = [
            p
            for p in all_files
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]
        return image_files

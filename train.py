import logging
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from typing import Optional
from zipfile import ZipFile, is_zipfile

from cog import BaseModel, Input, Path, Secret
from huggingface_hub import HfApi

# Configure logging to suppress INFO messages
logging.basicConfig(level=logging.WARNING, format="%(message)s")

# Suppress all common loggers
loggers_to_quiet = [
    "torch",
    "accelerate",
    "transformers",
    "__main__",
    "dataset",
    "networks",
    "hunyuan_model",
    "PIL",
    "qwen_vl_utils",
    "huggingface_hub",
    "diffusers",
    "filelock",
    "safetensors",
    "xformers",
    "datasets",
    "tokenizers",
    "sageattention",
]

for logger_name in loggers_to_quiet:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Suppress third-party warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# We return a path to our tarred LoRA weights at the end
class TrainingOutput(BaseModel):
    weights: Path


# [ADDED] Constants for Qwen2-VL model
QWEN_MODEL_CACHE = "qwen_checkpoints"
QWEN_MODEL_URL = (
    "https://weights.replicate.delivery/default/qwen/Qwen2-VL-7B-Instruct/model.tar"
)

INPUT_DIR = "input"
OUTPUT_DIR = "output"
MODEL_CACHE = "ckpts"
HF_UPLOAD_DIR = "hunyuan-lora-for-hf"

MODEL_FILES = ["hunyuan-video-t2v-720p.tar", "text_encoder.tar", "text_encoder_2.tar"]
BASE_URL = "https://weights.replicate.delivery/default/hunyuan-video/ckpts/"

sys.path.append("musubi-tuner")


def train(
        input_videos: Path = Input(
            description="A zip file containing videos that will be used for training. If you include captions, include them as one .txt file per video, e.g. my-video.mp4 should have a caption file named my-video.txt. If you don't include captions, you can use autocaptioning (enabled by default).",
            default=None,
        ),
        trigger_word: str = Input(
            description="The trigger word refers to the object, style or concept you are training on. Pick a string that isn't a real word, like TOK or something related to what's being trained, like STYLE3D. The trigger word you specify here will be associated with all videos during training. Then when you use your LoRA, you can include the trigger word in prompts to help activate the LoRA.",
            default="TOK",
        ),
        autocaption: bool = Input(
            description="Automatically caption videos using QWEN-VL", default=True
        ),
        autocaption_prefix: str = Input(
            description="Optional: Text you want to appear at the beginning of all your generated captions; for example, 'a video of TOK, '. You can include your trigger word in the prefix. Prefixes help set the right context for your captions.",
            default=None,
        ),
        autocaption_suffix: str = Input(
            description="Optional: Text you want to appear at the end of all your generated captions; for example, ' in the style of TOK'. You can include your trigger word in suffixes. Suffixes help set the right concept for your captions.",
            default=None,
        ),
        epochs: int = Input(
            description="Number of training epochs. Each epoch processes all your videos once. Note: If max_train_steps is set, training may end before completing all epochs.",
            default=16,
            ge=1,
            le=2000,
        ),
        max_train_steps: int = Input(
            description="Maximum number of training steps to perform. Each step processes one batch of frames. Set to -1 to train for the full number of epochs. If positive, training will stop after this many steps even if all epochs aren't complete.",
            default=-1,
            ge=-1,
            le=1_000_000,
        ),
        rank: int = Input(
            description="LoRA rank for training. Higher ranks take longer to train but can capture more complex features. Caption quality is more important for higher ranks.",
            default=32,
            ge=1,
            le=128,
        ),
        batch_size: int = Input(
            description="Batch size for training. Lower values use less memory but train slower.",
            default=4,
            ge=1,
            le=8,
        ),
        learning_rate: float = Input(
            description="Learning rate for training. If you're new to training you probably don't need to change this.",
            default=1e-3,
            ge=1e-5,
            le=1,
        ),
        optimizer: str = Input(
            description="Optimizer type for training. If you're unsure, leave as default.",
            default="adamw8bit",
            choices=["adamw", "adamw8bit", "AdaFactor", "adamw16bit"],
        ),
        timestep_sampling: str = Input(
            description="Controls how timesteps are sampled during training. 'sigmoid' (default) concentrates samples in the middle of the diffusion process. 'uniform' samples evenly across all timesteps. 'sigma' samples based on the noise schedule. 'shift' uses shifted sampling with discrete flow shift. If unsure, use 'sigmoid'.",
            default="sigmoid",
            choices=["sigma", "uniform", "sigmoid", "shift"],
        ),
        seed: int = Input(
            description="Random seed for training. Use <=0 for random.",
            default=0,
        ),
        hf_repo_id: str = Input(
            description="Hugging Face repository ID, if you'd like to upload the trained LoRA to Hugging Face. For example, username/my-video-lora. If the given repo does not exist, a new public repo will be created.",
            default=None,
        ),
        hf_token: Secret = Input(
            description="Hugging Face token, if you'd like to upload the trained LoRA to Hugging Face.",
            default=None,
        ),
        network_weights: Path = Input(
            description="Optional: Path to existing LoRA weights to continue training from. Use this to resume training from a previous run or to further fine-tune an existing LoRA.",
            default=None,
        ),
) -> TrainingOutput:
    """Minimal Hunyuan LoRA training script using musubi-tuner"""
    print("\n=== 🎥 Hunyuan Video & Image LoRA Training ===")
    print(f"📊 Configuration:")
    print(f"  • Input: {input_videos}")
    print(f"  • Training:")
    print(f"    - Epochs: {epochs}")
    if max_train_steps > 0:
        print(f"    - Max Steps: {max_train_steps}")
    print(f"    - LoRA Rank: {rank}")
    print(f"    - Learning Rate: {learning_rate}")
    print(f"    - Batch Size: {batch_size}")
    if network_weights:
        print(f"    - Continuing from: {network_weights}")
    if autocaption:
        print(f"  • Auto-captioning (videos only):")
        print(f"    - Enabled: {autocaption}")
        print(f"    - Trigger Word: {trigger_word}")
        if autocaption_prefix:
            print(f"    - Prefix: {autocaption_prefix}")
        if autocaption_suffix:
            print(f"    - Suffix: {autocaption_suffix}")
    print("=====================================\n")

    if not input_videos:
        raise ValueError(
            "You must provide a zip with videos, images, & optionally .txt captions."
        )

    clean_up()
    download_weights()
    seed = handle_seed(seed)

    # Setup directories for video duration folders
    videos_base_path = os.path.join(INPUT_DIR, "videos")
    os.makedirs(videos_base_path, exist_ok=True)

    # Create image directory
    images_base_path = os.path.join(INPUT_DIR, "images")
    os.makedirs(images_base_path, exist_ok=True)

    # Create cache directories
    for folder in ["1s", "2s", "3s", "4s", "5s"]:
        os.makedirs(os.path.join(INPUT_DIR, "cache_video", folder), exist_ok=True)
    os.makedirs(os.path.join(INPUT_DIR, "cache_image"), exist_ok=True)

    # Extract media files (videos and/or images) organized by duration
    extraction_results = extract_zip(
        input_videos,
        INPUT_DIR,
        autocaption=autocaption,
        trigger_word=trigger_word,
        autocaption_prefix=autocaption_prefix,
        autocaption_suffix=autocaption_suffix,
    )

    # Create training configuration based on what media was found
    create_train_toml(extraction_results)

    # Continue with the rest of the training process
    cache_latents()
    cache_text_encoder_outputs(batch_size)
    run_lora_training(
        epochs=epochs,
        rank=rank,
        optimizer=optimizer,
        learning_rate=learning_rate,
        timestep_sampling=timestep_sampling,
        seed=seed,
        max_train_steps=max_train_steps,
        network_weights=network_weights,
    )
    convert_lora_to_comfyui_format()
    output_path = archive_results()

    if hf_token and hf_repo_id:
        os.makedirs(HF_UPLOAD_DIR, exist_ok=True)
        shutil.move(
            os.path.join(OUTPUT_DIR, "lora.safetensors"),
            os.path.join(HF_UPLOAD_DIR, "lora.safetensors"),
        )
        handle_hf_upload(hf_repo_id, hf_token)

    return TrainingOutput(weights=Path(output_path))


def handle_seed(seed: int) -> int:
    """Handle random seed logic"""
    if seed <= 0:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")
    return seed


def create_train_toml(extraction_results: dict):
    """Create train.toml configuration file for frame-based video datasets and image dataset"""
    print("\n=== 📝 Creating Training Configuration ===")

    has_videos = extraction_results.get("has_videos", False)
    has_images = extraction_results.get("has_images", False)
    video_stats = extraction_results.get("video_stats", {})
    image_stats = extraction_results.get("image_stats", {})
    videos_path = extraction_results.get("videos_path", "input/videos")
    images_path = extraction_results.get("images_path", "input/images")

    with open("train.toml", "w") as f:
        config = """[general]
        caption_extension = ".txt"
        enable_bucket = true
        bucket_no_upscale = true
        """

        # Configure video datasets based on frame count folders
        if has_videos:
            # Add datasets for each frame count folder that has video files
            for folder, stats in video_stats.items():
                if stats["pairs"] > 0:
                    # Extract frame count from folder name (format: "frames_X")
                    frame_count = int(folder.split('_')[1])

                    # Calculate appropriate batch size based on frame count
                    if frame_count < 30:
                        batch_size = 12
                    elif frame_count < 60:
                        batch_size = 12
                    elif frame_count < 90:
                        batch_size = 12
                    elif frame_count < 120:
                        batch_size = 12
                    else:
                        batch_size = 12

                    # Use actual frame count for target_frames
                    actual_path = os.path.join(videos_path, folder)
                    cache_path = os.path.join("input/cache_video", folder)

                    config += f"""
                    [[datasets]]
                    video_directory = "{actual_path}"
                    cache_directory = "{cache_path}"
                    target_frames = [{frame_count}]
                    frame_extraction = "head"
                    resolution = [128, 128]
                    batch_size = {batch_size}
                    """

        # Configure image dataset if there are images (unchanged)
        if has_images and image_stats["pairs"] > 0:
            config += f"""
            [[datasets]]
            image_directory = "{images_path}"
            cache_directory = "input/cache_image"
            resolution = [1024, 1024]
            batch_size = 8
            num_repeats = 1
            """

        f.write(config)
        print("Training config:\n==================\n")
        print(config)
        print("\n==================\n")
    print("✅ Configuration file created")
    print("=====================================")


def cache_latents():
    """Cache latents using musubi-tuner"""
    print("\n=== 💾 Caching Video Latents ===")
    latent_args = [
        "python",
        "musubi-tuner/cache_latents.py",
        "--dataset_config",
        "train.toml",
        "--vae",
        os.path.join(MODEL_CACHE, "hunyuan-video-t2v-720p/vae/pytorch_model.pt"),
        "--vae_chunk_size",
        "32",
        "--vae_tiling",
    ]
    subprocess.run(latent_args, check=True)
    print("✅ Latents cached successfully")
    print("=====================================")


def cache_text_encoder_outputs(batch_size: int):
    """Cache text encoder outputs"""
    print("\n=== 💭 Caching Text Encodings ===")
    text_encoder_args = [
        "python",
        "musubi-tuner/cache_text_encoder_outputs.py",
        "--dataset_config",
        "train.toml",
        "--text_encoder1",
        os.path.join(MODEL_CACHE, "text_encoder"),
        "--text_encoder2",
        os.path.join(MODEL_CACHE, "text_encoder_2"),
        "--batch_size",
        str(batch_size),
    ]
    subprocess.run(text_encoder_args, check=True)
    print("✅ Text encodings cached")
    print("=====================================")


def run_lora_training(
        epochs: int,
        rank: int,
        optimizer: str,
        learning_rate: float,
        timestep_sampling: str,
        seed: int,
        max_train_steps: int = -1,  # Keep same interface as train()
        network_weights: Optional[Path] = None,  # Added parameter for existing weights
):
    """Run LoRA training with optional step limit and network weights"""
    print("\n=== 🚀 Starting LoRA Training ===")

    # Convert negative to zero for hv_train_network.py
    actual_max_steps = max(0, max_train_steps)

    if actual_max_steps > 0:
        print(f"• Max Train Steps: {actual_max_steps}")
    else:
        print(f"• Epochs: {epochs}")

    if network_weights:
        print(f"• Continuing from weights: {network_weights}")

    print("=====================================")

    training_args = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process",
        "8",
        "--mixed_precision",
        "bf16",
        "musubi-tuner/hv_train_network.py",
        "--dit",
        os.path.join(
            MODEL_CACHE,
            "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
        ),
        "--dataset_config",
        "train.toml",
        "--flash_attn",
        "--mixed_precision",
        "bf16",
        "--fp8_base",
        "--optimizer_type",
        optimizer,
        "--learning_rate",
        str(learning_rate),
        "--max_data_loader_n_workers",
        "16",
        "--persistent_data_loader_workers",
        "--network_module",
        "networks.lora",
        "--network_dim",
        str(rank),
        "--timestep_sampling",
        timestep_sampling,
        "--discrete_flow_shift",
        "3.0",
        "--seed",
        str(seed),
        "--output_dir",
        OUTPUT_DIR,
        "--output_name",
        "lora",
        "--gradient_checkpointing",
    ]

    # Add network_weights parameter if provided
    if network_weights:
        training_args.extend(["--network_weights", str(network_weights)])

    # Only add one of max_train_epochs or max_train_steps
    if actual_max_steps > 0:
        training_args.extend(["--max_train_steps", str(actual_max_steps)])
    else:
        training_args.extend(["--max_train_epochs", str(epochs)])

    subprocess.run(training_args, check=True)
    print("\n✨ Training Complete!")
    print("=====================================")


def convert_lora_to_comfyui_format():
    """Convert LoRA to ComfyUI-compatible format"""
    print("\n=== 🔄 Converting LoRA Format ===")
    original_lora_path = os.path.join(OUTPUT_DIR, "lora.safetensors")
    if os.path.exists(original_lora_path):
        converted_lora_path = os.path.join(OUTPUT_DIR, "lora_comfyui.safetensors")
        print(
            f"Converting from {original_lora_path} -> {converted_lora_path} (ComfyUI format)"
        )
        convert_args = [
            "python",
            "musubi-tuner/convert_lora.py",
            "--input",
            original_lora_path,
            "--output",
            converted_lora_path,
            "--target",
            "other",  # "other" -> diffusers style (ComfyUI)
        ]
        subprocess.run(convert_args, check=True)
    else:
        print("⚠️  Warning: lora.safetensors not found, skipping conversion.")
    print("✅ Converted to ComfyUI format")
    print("=====================================")


def archive_results() -> str:
    """Archive final results and return output path"""
    print("\n=== 📦 Archiving Results ===")
    output_path = "/tmp/trained_model.tar"

    print(f"Archiving LoRA outputs to {output_path}")
    os.system(f"tar -cvf {output_path} -C {OUTPUT_DIR} .")
    print(f"✅ Results archived to: {output_path}")
    print("=====================================")
    return output_path


def clean_up():
    """Removes INPUT_DIR, OUTPUT_DIR, and HF_UPLOAD_DIR if they exist."""
    for dir in [INPUT_DIR, OUTPUT_DIR, HF_UPLOAD_DIR]:
        if os.path.exists(dir):
            shutil.rmtree(dir)

    # Ensure necessary directories exist after cleanup
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create video directories for different durations
    videos_path = os.path.join(INPUT_DIR, "videos")
    os.makedirs(videos_path, exist_ok=True)

    # Create images directory
    os.makedirs(os.path.join(INPUT_DIR, "images"), exist_ok=True)

    # Create cache directories
    os.makedirs(os.path.join(INPUT_DIR, "cache_video"), exist_ok=True)
    os.makedirs(os.path.join(INPUT_DIR, "cache_image"), exist_ok=True)


def download_weights():
    """Download base Hunyuan model weights if not already cached."""
    os.makedirs(MODEL_CACHE, exist_ok=True)
    for model_file in MODEL_FILES:
        filename_no_ext = model_file.split(".")[0]
        dest_path = os.path.join(MODEL_CACHE, filename_no_ext)
        if not os.path.exists(dest_path):
            url = BASE_URL + model_file
            print(f"Downloading {url} to {MODEL_CACHE}")
            subprocess.check_call(["pget", "-xf", url, MODEL_CACHE])


def autocaption_videos(
        videos_path: str,
        video_files: set,
        caption_files: set,
        trigger_word: Optional[str] = None,
        autocaption_prefix: Optional[str] = None,
        autocaption_suffix: Optional[str] = None,
) -> set:
    """Generate captions for videos that don't have matching .txt files."""
    videos_without_captions = video_files - caption_files
    if not videos_without_captions:
        return caption_files

    print("\n=== 🤖 Auto-captioning Videos ===")
    print(f"Found {len(videos_without_captions)} videos without captions")
    model, processor = setup_qwen_model()

    new_caption_files = caption_files.copy()
    for i, vid_name in enumerate(videos_without_captions, 1):
        mp4_path = os.path.join(videos_path, vid_name + ".mp4")
        if os.path.exists(mp4_path):
            print(f"\n[{i}/{len(videos_without_captions)}] 🎥 {vid_name}.mp4")

            # Use absolute path
            abs_path = os.path.abspath(mp4_path)

            # Build caption components
            prefix = f"{autocaption_prefix.strip()} " if autocaption_prefix else ""
            suffix = f" {autocaption_suffix.strip()}" if autocaption_suffix else ""
            trigger = f"{trigger_word} " if trigger_word else ""

            # Prepare messages format with customized prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": abs_path,
                        },
                        {
                            "type": "text",
                            "text": "Describe this video clip in detail, focusing on the key visual elements, actions, and overall scene.",
                        },
                    ],
                }
            ]

            # Process inputs
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            try:
                # Import qwen utils here to avoid circular imports
                from qwen_vl_utils import process_vision_info

                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

                print("\nGenerating caption...")
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                caption = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                # Combine prefix, trigger, caption, and suffix
                final_caption = f"{prefix}{trigger}{caption.strip()}{suffix}"
                print("\n📝 Generated Caption:")
                print("--------------------")
                print(f"{final_caption}")
                print("--------------------")

            except Exception as e:
                print(f"\n⚠️  Warning: Failed to autocaption {vid_name}.mp4")
                print(f"Error: {str(e)}")
                final_caption = (
                    f"{prefix}{trigger}A video clip named {vid_name}{suffix}"
                )
                print("\n📝 Using fallback caption:")
                print("--------------------")
                print(f"{final_caption}")
                print("--------------------")

            # Save caption
            txt_path = os.path.join(videos_path, vid_name + ".txt")
            with open(txt_path, "w") as f:
                f.write(final_caption.strip() + "\n")
            new_caption_files.add(vid_name)
            print(f"✅ Saved to: {txt_path}")

    # Clean up QWEN model
    print("\n=== 🧹 Cleaning Up ===")
    del model
    del processor
    import torch

    torch.cuda.empty_cache()

    print(f"✨ Successfully processed {len(videos_without_captions)} videos!")
    print("=====================================")

    return new_caption_files


def extract_zip(
        zip_path: Path,
        extraction_dir: str,
        autocaption: bool = True,
        trigger_word: Optional[str] = None,
        autocaption_prefix: Optional[str] = None,
        autocaption_suffix: Optional[str] = None,
):
    """
    Extract videos, images & .txt captions from the provided zip.
    Sort videos by duration (frame count) and images by aspect ratio.
    If autocaption is True, generate captions for missing media files.
    """
    if not is_zipfile(zip_path):
        raise ValueError("The provided input must be a zip file.")

    # Setup directories
    os.makedirs(extraction_dir, exist_ok=True)

    # Create aspect ratio folders for images
    images_base_path = os.path.join(extraction_dir, "images")
    os.makedirs(images_base_path, exist_ok=True)

    # Create duration-based folders for videos
    videos_base_path = os.path.join(extraction_dir, "videos")

    # Extract and track files
    video_files = set()
    image_files = set()
    caption_files = set()
    file_count = 0

    # Track files by lowercase name to handle case sensitivity
    image_map = {}  # maps lowercase name to actual filename
    video_map = {}  # maps lowercase name to actual filename and duration folder
    caption_map = {}  # maps lowercase name to actual filename

    # Track image dimensions for sorting and video frame counts
    media_dimensions = {}  # maps filename to (width, height)
    video_frame_counts = {}  # maps filename to frame count

    # Define image extensions
    image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP")

    # Function to determine which video folder to use based on frame count
    def get_video_folder(frame_count):
        # Use actual frame count as folder name
        return f"frames_{frame_count}"

    # Function to get image dimensions
    def get_image_dimensions(image_path):
        from PIL import Image
        try:
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            print(f"⚠️  Warning: Could not determine dimensions for {image_path}: {e}")
            return (512, 512)  # Default to square if we can't determine

    # Function to get video dimensions and frame count
    def get_video_info(video_path):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return (width, height), frame_count
        except Exception as e:
            print(f"⚠️  Warning: Could not determine info for {video_path}: {e}")
            return (512, 512), 25  # Default values if we can't determine

    # Create a temporary directory for initial extraction
    temp_extract_dir = os.path.join(extraction_dir, "temp_extract")
    os.makedirs(temp_extract_dir, exist_ok=True)

    with ZipFile(zip_path, "r") as zip_ref:
        # First, extract all files to a temporary directory
        for file_info in zip_ref.infolist():
            if not file_info.filename.startswith("__MACOSX/") and not file_info.filename.startswith("._"):
                zip_ref.extract(file_info, temp_extract_dir)

    # Now process and categorize the extracted files
    for root, _, files in os.walk(temp_extract_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            filename_lower = filename.lower()
            base_name = os.path.basename(filename)
            base_name_no_ext = os.path.splitext(base_name)[0]
            base_name_lower_no_ext = base_name_no_ext.lower()

            # Process videos
            if filename_lower.endswith(".mp4"):
                # Get video dimensions and frame count
                dimensions, frame_count = get_video_info(filepath)
                media_dimensions[base_name_no_ext] = dimensions
                video_frame_counts[base_name_no_ext] = frame_count

                # Determine which folder to use
                video_folder = get_video_folder(frame_count)

                # Move to appropriate directory
                target_dir = os.path.join(videos_base_path, video_folder)
                os.makedirs(target_dir, exist_ok=True)

                target_path = os.path.join(target_dir, base_name)
                shutil.copy2(filepath, target_path)

                video_files.add(base_name_no_ext)
                video_map[base_name_lower_no_ext] = (base_name_no_ext, video_folder)
                file_count += 1

                print(
                    f"Video: {base_name} → {video_folder} folder ({dimensions[0]}x{dimensions[1]}, {frame_count} frames)")

            # Process images
            elif filename_lower.endswith(image_extensions):
                # Get image dimensions
                dimensions = get_image_dimensions(filepath)
                media_dimensions[base_name_no_ext] = dimensions

                # Move to appropriate directory
                target_path = os.path.join(images_base_path, base_name)
                shutil.copy2(filepath, target_path)

                image_files.add(base_name_no_ext)
                image_map[base_name_lower_no_ext] = base_name_no_ext
                file_count += 1

                print(f"Image: {base_name} ({dimensions[0]}x{dimensions[1]})")

            # Process caption files (will handle later)
            elif filename_lower.endswith(".txt"):
                caption_files.add(base_name_no_ext)
                caption_map[base_name_lower_no_ext] = base_name_no_ext

    # Now handle caption files based on their associated media files
    for root, _, files in os.walk(temp_extract_dir):
        for filename in files:
            if filename.lower().endswith(".txt"):
                filepath = os.path.join(root, filename)
                base_name = os.path.basename(filename)
                base_name_no_ext = os.path.splitext(base_name)[0]
                base_name_lower_no_ext = base_name_no_ext.lower()

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        caption_content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(filepath, 'r', encoding='latin-1') as f:
                            caption_content = f.read()
                    except Exception as e:
                        print(f"⚠️  Warning: Could not read caption file {base_name}: {e}")
                        caption_content = f"{trigger_word if trigger_word else ''} {base_name_no_ext}"

                # Copy caption to appropriate directory for images
                if base_name_lower_no_ext in image_map:
                    img_name = image_map[base_name_lower_no_ext]
                    target_dir = images_base_path
                    with open(os.path.join(target_dir, f"{img_name}.txt"), 'w', encoding='utf-8') as f:
                        f.write(caption_content)
                    print(f"Caption for image: {base_name}")

                # Copy caption to appropriate directory for videos
                if base_name_lower_no_ext in video_map:
                    vid_name, video_folder = video_map[base_name_lower_no_ext]
                    target_dir = os.path.join(videos_base_path, video_folder)
                    with open(os.path.join(target_dir, f"{vid_name}.txt"), 'w', encoding='utf-8') as f:
                        f.write(caption_content)
                    print(f"Caption for video: {base_name} → {video_folder} folder")

                file_count += 1

    # Clean up temporary extraction directory
    shutil.rmtree(temp_extract_dir)

    # Verify media and caption pairs for videos and images
    video_stats = defaultdict(lambda: {"files": 0, "captions": 0, "pairs": 0})
    image_stats = {"files": 0, "captions": 0, "pairs": 0}

    # Check image directory
    image_files_in_dir = [f for f in os.listdir(images_base_path) if
                          any(f.lower().endswith(ext) for ext in image_extensions)]
    image_captions_in_dir = [f for f in os.listdir(images_base_path) if f.lower().endswith(".txt")]

    image_stats["files"] = len(image_files_in_dir)
    image_stats["captions"] = len(image_captions_in_dir)

    # Count image pairs
    image_basenames = {os.path.splitext(f)[0] for f in image_files_in_dir}
    image_caption_basenames = {os.path.splitext(f)[0] for f in image_captions_in_dir}
    image_pairs = image_basenames.intersection(image_caption_basenames)
    image_stats["pairs"] = len(image_pairs)

    # Handle missing image captions - create placeholders
    for img_file in image_basenames:
        if img_file not in image_caption_basenames:
            caption_path = os.path.join(images_base_path, f"{img_file}.txt")
            with open(caption_path, 'w', encoding='utf-8') as f:
                placeholder = f"{trigger_word if trigger_word else 'Image'} {img_file}"
                f.write(placeholder)
                print(f"  Created placeholder caption: '{placeholder}' for image")
            image_stats["captions"] += 1
            image_stats["pairs"] += 1

    # Check video directories
    video_folders = [d for d in os.listdir(videos_base_path)
                     if os.path.isdir(os.path.join(videos_base_path, d))]

    for folder in video_folders:
        folder_path = os.path.join(videos_base_path, folder)

        # Skip if the directory doesn't exist
        if not os.path.exists(folder_path):
            continue

        video_files_in_dir = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")]
        video_captions_in_dir = [f for f in os.listdir(folder_path) if f.lower().endswith(".txt")]

        video_stats[folder]["files"] = len(video_files_in_dir)
        video_stats[folder]["captions"] = len(video_captions_in_dir)

        # Count video pairs
        video_basenames = {os.path.splitext(f)[0] for f in video_files_in_dir}
        video_caption_basenames = {os.path.splitext(f)[0] for f in video_captions_in_dir}
        video_pairs = video_basenames.intersection(video_caption_basenames)
        video_stats[folder]["pairs"] = len(video_pairs)

    # Autocaption videos if needed
    videos_without_captions = set()
    if autocaption:
        for folder in video_folders:
            folder_path = os.path.join(videos_base_path, folder)
            video_files_in_dir = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")]
            video_captions_in_dir = [f for f in os.listdir(folder_path) if f.lower().endswith(".txt")]

            video_basenames = {os.path.splitext(f)[0] for f in video_files_in_dir}
            video_caption_basenames = {os.path.splitext(f)[0] for f in video_captions_in_dir}

            # Collect videos without captions
            for vid_base in video_basenames:
                if vid_base not in video_caption_basenames:
                    videos_without_captions.add((folder, vid_base))

        # Autocaption videos
        if videos_without_captions:
            print(f"\n=== 🤖 Auto-captioning Videos ===")
            print(f"Found {len(videos_without_captions)} videos without captions")
            model, processor = setup_qwen_model()

            for i, (folder, vid_name) in enumerate(videos_without_captions, 1):
                folder_path = os.path.join(videos_base_path, folder)
                mp4_path = os.path.join(folder_path, f"{vid_name}.mp4")

                if os.path.exists(mp4_path):
                    print(f"\n[{i}/{len(videos_without_captions)}] 🎥 {vid_name}.mp4 ({folder})")

                    # Use absolute path
                    abs_path = os.path.abspath(mp4_path)

                    # Build caption components
                    prefix = f"{autocaption_prefix.strip()} " if autocaption_prefix else ""
                    suffix = f" {autocaption_suffix.strip()}" if autocaption_suffix else ""
                    trigger = f"{trigger_word} " if trigger_word else ""

                    # Prepare messages format with customized prompt
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": abs_path,
                                },
                                {
                                    "type": "text",
                                    "text": "Describe this video clip in detail, focusing on the key visual elements, actions, and overall scene.",
                                },
                            ],
                        }
                    ]

                    # Process inputs
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    try:
                        # Import qwen utils here to avoid circular imports
                        from qwen_vl_utils import process_vision_info

                        image_inputs, video_inputs = process_vision_info(messages)

                        inputs = processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        ).to("cuda")

                        print("\nGenerating caption...")
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                        )
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):]
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        caption = processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]

                        # Combine prefix, trigger, caption, and suffix
                        final_caption = f"{prefix}{trigger}{caption.strip()}{suffix}"
                        print("\n📝 Generated Caption:")
                        print("--------------------")
                        print(f"{final_caption}")
                        print("--------------------")

                    except Exception as e:
                        print(f"\n⚠️  Warning: Failed to autocaption {vid_name}.mp4")
                        print(f"Error: {str(e)}")
                        final_caption = f"{prefix}{trigger}A video clip named {vid_name}{suffix}"
                        print("\n📝 Using fallback caption:")
                        print("--------------------")
                        print(f"{final_caption}")
                        print("--------------------")

                    # Save caption
                    txt_path = os.path.join(folder_path, f"{vid_name}.txt")
                    with open(txt_path, "w") as f:
                        f.write(final_caption.strip() + "\n")
                    video_stats[folder]["captions"] += 1
                    video_stats[folder]["pairs"] += 1
                    print(f"✅ Saved to: {txt_path}")

            # Clean up QWEN model
            print("\n=== 🧹 Cleaning Up ===")
            del model
            del processor
            import torch
            torch.cuda.empty_cache()

            print(f"✨ Successfully processed {len(videos_without_captions)} videos!")
            print("=====================================")

    # Create placeholder captions for remaining videos without autocaption
    if not autocaption:
        for folder in video_folders:
            folder_path = os.path.join(videos_base_path, folder)
            video_files_in_dir = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")]
            video_captions_in_dir = [f for f in os.listdir(folder_path) if f.lower().endswith(".txt")]

            video_basenames = {os.path.splitext(f)[0] for f in video_files_in_dir}
            video_caption_basenames = {os.path.splitext(f)[0] for f in video_captions_in_dir}

            # Create placeholders for videos without captions
            for vid_base in video_basenames:
                if vid_base not in video_caption_basenames:
                    caption_path = os.path.join(folder_path, f"{vid_base}.txt")
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        placeholder = f"{trigger_word if trigger_word else 'Video'} {vid_base}"
                        f.write(placeholder)
                        print(f"  Created placeholder caption: '{placeholder}' in {folder}")
                    video_stats[folder]["captions"] += 1
                    video_stats[folder]["pairs"] += 1

    # Print statistics
    print("\n=== 📊 Media Statistics ===")

    total_valid_pairs = 0
    has_videos = False
    has_images = False

    # Image statistics
    if image_stats["files"] > 0:
        has_images = True
        print(
            f"\nImages: {image_stats['files']} files, {image_stats['captions']} captions, {image_stats['pairs']} valid pairs")
        total_valid_pairs += image_stats["pairs"]

    # Video statistics by folder
    print("\nVideos by frame count:")
    total_video_pairs = 0
    for folder, stats in video_stats.items():
        if stats["files"] > 0:
            has_videos = True
            print(f"  • {folder}: {stats['files']} files, {stats['captions']} captions, {stats['pairs']} valid pairs")
            total_video_pairs += stats["pairs"]

    print(f"\nTotal video pairs: {total_video_pairs}")
    total_valid_pairs += total_video_pairs

    print(f"\nTotal valid media-caption pairs: {total_valid_pairs}")
    print("=====================================")

    if not (has_videos or has_images):
        raise ValueError("No media files (videos or images) found in zip!")
    if total_valid_pairs == 0:
        raise ValueError("No matching media-caption pairs found after checking or generating captions!")

    # Return information about what was found and where
    return {
        "has_videos": has_videos,
        "has_images": has_images,
        "videos_path": videos_base_path,
        "images_path": images_base_path,
        "video_stats": video_stats,
        "image_stats": image_stats
    }


def handle_hf_upload(hf_repo_id: str, hf_token: Secret):
    print(f"HF Token: {hf_token}")
    print(f"HF Repo ID: {hf_repo_id}")
    if hf_token is not None and hf_repo_id is not None:
        try:
            title = handle_hf_readme(hf_repo_id)
            print(f"Uploading to Hugging Face: {hf_repo_id}")
            api = HfApi()

            repo_url = api.create_repo(
                hf_repo_id,
                private=False,
                exist_ok=True,
                token=hf_token.get_secret_value(),
            )

            print(f"HF Repo URL: {repo_url}")

            # Rename lora.safetensors to hunyuan-[title].safetensors
            old_path = HF_UPLOAD_DIR / Path("lora.safetensors")
            new_name = title.lower()
            if not new_name.startswith("hunyuan"):
                new_name = f"hunyuan-{new_name}"
            new_path = HF_UPLOAD_DIR / Path(f"{new_name}.safetensors")
            os.rename(old_path, new_path)

            api.upload_folder(
                repo_id=hf_repo_id,
                folder_path=HF_UPLOAD_DIR,
                repo_type="model",
                use_auth_token=hf_token.get_secret_value(),
            )
        except Exception as e:
            print(f"Error uploading to Hugging Face: {str(e)}")


def handle_hf_readme(hf_repo_id: str) -> str:
    readme_path = HF_UPLOAD_DIR / Path("README.md")
    license_path = Path("hf-lora-readme-template.md")
    shutil.copy(license_path, readme_path)

    content = readme_path.read_text()

    repo_parts = hf_repo_id.split("/")
    if len(repo_parts) > 1:
        title = repo_parts[1].replace("-", " ").title()
        content = content.replace("[title]", title)
    else:
        title = hf_repo_id
        content = content.replace("[title]", title)

    print("HF readme content:\n==================\n")
    print(content)
    print("\n==================\n")
    readme_path.write_text(content)
    return title.replace(" ", "-")


def setup_qwen_model():
    """Download and setup Qwen2-VL model for auto-captioning"""
    # TODO: use download_weights() instead
    if not os.path.exists(QWEN_MODEL_CACHE):
        print(f"Downloading Qwen2-VL model to {QWEN_MODEL_CACHE}")
        start = time.time()
        subprocess.check_call(["pget", "-xf", QWEN_MODEL_URL, QWEN_MODEL_CACHE])
        print(f"Download took: {time.time() - start:.2f}s")

    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print("\nLoading QWEN model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_CACHE,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_CACHE)

    return model, processor

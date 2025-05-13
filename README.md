# Hunyuan Video Fine-Tuning
[![Replicate](https://replicate.com/zsxkib/hunyuan-video-lora/badge)](https://replicate.com/zsxkib/hunyuan-video-lora)

A powerful toolkit for fine-tuning [Hunyuan Video LoRA](https://replicate.com/zsxkib/hunyuan-video-lora) using LoRA, plus advanced video inference and automatic captioning via QWEN-VL. This guide focuses on the most important aspect: how to run fine-tuning (training) and generation (inference) using Cog, with detailed explanations of all parameters.

---

## Table of Contents
- [Hunyuan Video Fine-Tuning](#hunyuan-video-fine-tuning)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [Installation \& Setup](#installation--setup)
  - [Training](#training)
    - [Training Command](#training-command)
    - [Training Parameters](#training-parameters)
    - [Examples](#examples)
  - [Inference](#inference)
    - [Inference Command](#inference-command)
    - [Inference Parameters](#inference-parameters)
    - [Examples](#examples-1)
  - [Tips \& Tricks](#tips--tricks)
  - [License](#license)

---

## Quick Start

1. Place your training videos in a ZIP file. Optionally include <video_name>.txt captions alongside each <video_name>.mp4, e.g.:
   ```
   your_data.zip/
   ├── dance_scene.mp4
   ├── dance_scene.txt
   ├── city_stroll.mp4
   └── ...
   ```
   > **Tip**: You can use [create-video-dataset](https://replicate.com/zsxkib/create-video-dataset) to easily prepare your training data with automatic QWEN-VL captioning.

2. Install [Cog](https://github.com/replicate/cog) and [Docker](https://www.docker.com).
3. Run the training example command (see below).  
4. After training, run the inference example command to generate a new video.  

---

## Installation & Setup

1. Install Docker (required by Cog).  
2. Install Cog from [cog.run](https://cog.run):
   ```bash
   curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
   chmod +x /usr/local/bin/cog
   pip install cog
   ```
3. Clone or download this repository.  
4. From the project root directory, you can run Cog commands with parameters:
   ```bash
   # For training:
   sudo cog train -i "input_videos=@your_videos.zip" -i "trigger_word=MYSTYLE"

   # For inference:
   sudo cog predict -i "prompt=your prompt here" -i "replicate_weights=@/tmp/trained_model.tar"
   ```
5. See below for detailed parameter explanations and more examples.

---

## Training

### Training Command

Use:
```bash
sudo cog train \
  -i "input_videos=@your_videos.zip" \
  [other parameters...]
```

The result of training is saved to `/tmp/trained_model.tar` containing:
- LoRA weights (.safetensors)  
- (Optional) ComfyUI-compatible LoRA  
- Any logs or training artifacts  

You can use this output directly in inference by passing it to the `replicate_weights` parameter:
```bash
sudo cog predict \
  -i "prompt='Your prompt here'" \
  -i "replicate_weights=@/tmp/trained_model.tar" \
  [other parameters...]
```

### Training Parameters

Below are the key parameters you can supply to `cog train`. All parameters have validated types and ranges:

• input_videos (Path)  
  - Description: A ZIP file containing videos (and optional .txt captions).  
  - Example: -i "input_videos=@my_videos.zip"  

• trigger_word (str)  
  - Description: A "fake" or "rare" word that represents the style or concept you're training on.  
  - Default: "TOK"  
  - Example: -i "trigger_word=STYLE3D"  

• autocaption (bool)  
  - Description: Whether to auto-caption your videos using QWEN-VL.  
  - Default: True  
  - Example: -i "autocaption=false"  

• autocaption_prefix (str)  
  - Description: Text prepended to all generated captions (helps set consistent context).  
  - Default: None  
  - Example: -i "autocaption_prefix='A cinematic scene of TOK, '"  

• autocaption_suffix (str)  
  - Description: Text appended to all generated captions (helps reinforce the concept).  
  - Default: None  
  - Example: -i "autocaption_suffix='in the art style of TOK.'"  

• epochs (int)  
  - Description: Number of full passes (epochs) over the dataset.  
  - Range: 1–2000  
  - Default: 16  

• max_train_steps (int)  
  - Description: Limit the total number of steps (each step processes one batch). -1 for unlimited.  
  - Range: -1–1,000,000  
  - Default: -1  

• rank (int)  
  - Description: LoRA rank. Higher rank can capture more detail but also uses more resources.  
  - Range: 1–128  
  - Default: 32  

• batch_size (int)  
  - Description: Batch size (frames per iteration). Lower for less VRAM usage.  
  - Range: 1–8  
  - Default: 4  

• learning_rate (float)  
  - Description: Training learning rate.  
  - Range: 1e-5–1  
  - Default: 1e-3  

• optimizer (str)  
  - Description: Which optimizer to use. Usually "adamw8bit" is a good default.  
  - Choices: ["adamw", "adamw8bit", "AdaFactor", "adamw16bit"]  
  - Default: "adamw8bit"  

• timestep_sampling (str)  
  - Description: Sampling strategy across diffusion timesteps.  
  - Choices: ["sigma", "uniform", "sigmoid", "shift"]  
  - Default: "sigmoid"  

• consecutive_target_frames (str)  
  - Description: How many consecutive frames to pull from each video.  
  - Choices: ["[1, 13, 25]", "[1, 25, 45]", "[1, 45, 89]", "[1, 13, 25, 45]"]  
  - Default: "[1, 25, 45]"  

• frame_extraction_method (str)  
  - Description: How frames are extracted (start, chunk, sliding-window, uniform).  
  - Choices: ["head", "chunk", "slide", "uniform"]  
  - Default: "head"  

• frame_stride (int)  
  - Description: Stride used for slide-based extraction.  
  - Range: 1–100  
  - Default: 10  

• frame_sample (int)  
  - Description: Number of samples used in uniform extraction.  
  - Range: 1–20  
  - Default: 4  

• seed (int)  
  - Description: Random seed. Use <= 0 for truly random.  
  - Default: 0  

• hf_repo_id (str)  
  - Description: If you want to push your LoRA to Hugging Face, specify "username/my-video-lora".  
  - Default: None  

• hf_token (Secret)  
  - Description: Hugging Face token for uploading to a private or public repository.  
  - Default: None  

### Examples

1. **Simple Training**  
   ```bash
   sudo cog train \
     -i "input_videos=@your_videos.zip" \
     -i "trigger_word=MYSTYLE" \
     -i "epochs=4"
   ```
   This runs 4 epochs with default batch size and autocaption.

2. **Memory-Constrained Training**  
   ```bash
   sudo cog train \
     -i "input_videos=@your_videos.zip" \
     -i "rank=16" \
     -i "batch_size=1" \
     -i "gradient_checkpointing=true"
   ```
   Uses a lower rank and smaller batch size to reduce VRAM usage, plus gradient checkpointing.

3. **Motion-Focused Training**  
   ```bash
   sudo cog train \
     -i "input_videos=@videos.zip" \
     -i "consecutive_target_frames=[1, 45, 89]" \
     -i "frame_extraction_method=slide" \
     -i "frame_stride=10"
   ```
   Extracts frames in sliding windows to capture more motion variety.

4. **Quick Test Run**  
   ```bash
   sudo cog train \
     -i "input_videos=@test.zip" \
     -i "rank=16" \
     -i "epochs=4" \
     -i "max_train_steps=100" \
     -i "batch_size=1" \
     -i "gradient_checkpointing=true"
   ```
   Minimal training to verify your setup and data.

5. **Style Focus**  
   ```bash
   sudo cog train \
     -i "input_videos=@style.zip" \
     -i "consecutive_target_frames=[1]" \
     -i "frame_extraction_method=uniform" \
     -i "frame_sample=8" \
     -i "epochs=16"
   ```
   Optimized for learning static style elements rather than motion.

---

## Inference

### Inference Command

Use:
```bash
sudo cog predict \
  -i "prompt='Your prompt here'" \
  [other parameters...]
```

The generated video is saved to the output directory (usually /src or /outputs inside Docker), and Cog returns the path.

### Inference Parameters

Below are the key parameters for `cog predict`:

• prompt (str)  
  - Description: Your text prompt for the scene or style.  
  - Example: -i "prompt='A cinematic shot of a forest in MYSTYLE'"  

• lora_url (str)  
  - Description: URL or Hugging Face repo ID for the LoRA weights.  
  - Example: -i "lora_url='myuser/my-lora-repo'"  

• lora_strength (float)  
  - Description: How strongly the LoRA style is applied.  
  - Range: -10.0–10.0  
  - Default: 1.0  

• scheduler (str)  
  - Description: The diffusion sampling/flow algorithm.  
  - Choices: ["FlowMatchDiscreteScheduler", "SDE-DPMSolverMultistepScheduler", "DPMSolverMultistepScheduler", "SASolverScheduler", "UniPCMultistepScheduler"]  
  - Default: "DPMSolverMultistepScheduler"  

• steps (int)  
  - Description: Number of diffusion steps.  
  - Range: 1–150  
  - Default: 50  

• guidance_scale (float)  
  - Description: How strongly the prompt influences the generation.  
  - Range: 0.0–30.0  
  - Default: 6.0  

• flow_shift (int)  
  - Description: Adjusts motion consistency across frames.  
  - Range: 0–20  
  - Default: 9  

• num_frames (int)  
  - Description: Total frames in the output video.  
  - Range: 1–1440  
  - Default: 33  

• width (int), height (int)  
  - Description: Dimensions of generated frames.  
  - Range: width (64–1536), height (64–1024)  
  - Default: 640x360  

• denoise_strength (float)  
  - Description: Controls how strongly noise is applied each step: 0 = minimal noise, 2 = heavy noise.  
  - Range: 0.0–2.0  
  - Default: 1.0  

• force_offload (bool)  
  - Description: Offload layers to CPU for lower VRAM usage.  
  - Default: True  

• frame_rate (int)  
  - Description: Frames per second in the final video.  
  - Range: 1–60  
  - Default: 16  

• crf (int)  
  - Description: H.264 compression quality. Lower = better.  
  - Range: 0–51  
  - Default: 19  

• enhance_weight (float)  
  - Description: Strength of optional enhancement effect.  
  - Range: 0.0–2.0  
  - Default: 0.3  

• enhance_single (bool) & enhance_double (bool)  
  - Description: Whether to enable enhancement on single frames or across pairs of frames.  
  - Default: True, True  

• enhance_start (float) & enhance_end (float)  
  - Description: Control when in the video enhancement starts or ends (fractional times, 0.0–1.0 range).  
  - Default: 0.0–1.0  

• seed (int)  
  - Description: Random seed for reproducible output.  
  - Default: random if not provided  

• replicate_weights (Path)  
  - Description: Path to a local .tar containing LoRA weights from replicate training.  
  - Default: None  

### Examples

1. **Basic Inference with Local LoRA**  
   ```bash
   sudo cog predict \
     -i "prompt='A serene lake at sunrise in the style of MYSTYLE'" \
     -i "lora_url='local-file.safetensors'" \
     -i "width=512" \
     -i "height=512" \
     -i "steps=30"
   ```

2. **Advanced Motion and Quality**  
   ```bash
   sudo cog predict \
     -i "prompt='TOK winter cityscape, moody lighting'" \
     -i "lora_url='myuser/my-lora-repo'" \
     -i "steps=50" \
     -i "flow_shift=15" \
     -i "num_frames=80" \
     -i "frame_rate=30" \
     -i "crf=17" \
     -i "lora_strength=1.2"
   ```
   Here, we use more frames, faster frame rate, and a lower CRF for higher quality.

3. **Using Replicate Tar**  
   ```bash
   sudo cog predict \
     -i "prompt='An astronaut dancing on Mars in style TOK'" \
     -i "replicate_weights=@trained_model.tar" \
     -i "guidance_scale=8" \
     -i "num_frames=45"
   ```
   Instead of lora_url, we pass a local .tar with LoRA weights.

4. **Quick Preview**  
   ```bash
   sudo cog predict \
     -i 'steps=30' \
     -i 'width=512' \
     -i 'height=512' \
     -i 'num_frames=33' \
     -i 'force_offload=true'
   ```

5. **Smooth Motion**  
   ```bash
   sudo cog predict \
     -i 'scheduler=FlowMatchDiscreteScheduler' \
     -i 'flow_shift=15' \
     -i 'frame_rate=30' \
     -i 'num_frames=89'
   ```

---

## Tips & Tricks

1. **Reduce OOM Errors**  
   - Use a smaller `batch_size` or lower `rank` during training.  
   - Enable `force_offload=true` during inference.  

2. **Better Quality**  
   - Increase `steps` and `guidance_scale`.  
   - Use a lower `crf` (e.g., 17 or 18).  

3. **Faster Training**  
   - For smaller datasets, reduce `epochs`.  
   - Increase `learning_rate` slightly (e.g., 2e-3) while monitoring for overfitting.  

4. **Motion Emphasis**  
   - Use `frame_extraction_method=slide` or `consecutive_target_frames=[1, 25, 45]` during training for improved motion consistency.  
   - Adjust `flow_shift` (5–15 range) during inference.  

5. **Style Activation**  
   - Always include your `trigger_word` in the inference prompt.  

---

## License

This project is released under the MIT License.  
Please see the [LICENSE](LICENSE) file for details.
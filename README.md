### GarmentGen

Generate realistic, front-facing garment images (with transparent backgrounds) from a text prompt and a body pose map using ControlNet.

#### Features
- Text-to-garment with faithful style, fabric, color, and pattern
- Pose-aligned via OpenPose ControlNet
- Transparent background (RGBA PNG) via background removal
- Front-facing, garment-only output (no human, no background)
- CPU-compatible (slow) and GPU-accelerated when available

#### Install (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Note: On some distros you may need to install `python3-venv` first.

#### Usage

```bash
garmentgen \
  --prompt "A red silk A-line dress with floral prints and short sleeves." \
  --pose_image /path/to/pose.png \
  --output out/dress.png \
  --height 1024 --width 768
```

Alternatively, provide COCO-format keypoints JSON to render a pose map:

```bash
garmentgen \
  --prompt "A navy denim trucker jacket with contrast stitching." \
  --pose_keypoints pose.json \
  --output out/jacket.png
```

#### Inputs
- `--prompt`: Natural language garment description
- `--pose_image`: Path to a pose/skeleton map image (recommended)
- `--pose_keypoints`: Path to COCO keypoints JSON (person[0])

#### Output
- Single RGBA PNG with only the garment, transparent background. Default size 1024×1024.

#### Models
- Base: `runwayml/stable-diffusion-v1-5`
- ControlNet: `lllyasviel/sd-controlnet-openpose`

Override with environment variables:
- `BASE_MODEL_ID`
- `CONTROLNET_MODEL_ID`

#### Tips
- Describe garment type, fabric, color, and patterns.
- Add details like sleeve length, collar shape, fit, hemline.
- For pose, a front-facing openpose map yields best alignment.

#### License
MIT


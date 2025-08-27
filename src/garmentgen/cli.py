import argparse
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a garment image aligned to a body pose map, with transparent background.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Garment description.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pose_image", type=str, help="Path to pose/skeleton image (OpenPose-like).")
    group.add_argument("--pose_keypoints", type=str, help="Path to COCO keypoints JSON.")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path (RGBA).")
    parser.add_argument("--height", type=int, default=1024, help="Output height.")
    parser.add_argument("--width", type=int, default=1024, help="Output width.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cpu"))
    parser.add_argument("--base_model_id", type=str, default=os.environ.get("BASE_MODEL_ID", "runwayml/stable-diffusion-v1-5"))
    parser.add_argument(
        "--controlnet_model_id",
        type=str,
        default=os.environ.get("CONTROLNET_MODEL_ID", "lllyasviel/sd-controlnet-openpose"),
    )
    parser.add_argument("--disable_bg_removal", action="store_true", help="Skip background removal.")
    parser.add_argument("--strength", type=float, default=1.0, help="ControlNet conditioning scale.")
    parser.add_argument("--neg_prompt", type=str, default=(
        "human, person, skin, face, hands, head, legs, arms, body, mannequin, doll, portrait,"
        " nsfw, nude, duplicate, watermark, text, logo, caption, blurry, lowres, deformed"
    ))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Delay heavy imports until actually running generation
    from .pose_utils import load_pose_image, render_pose_from_coco
    from .generator import GarmentGenerator

    if args.pose_image:
        control_image = load_pose_image(args.pose_image, (args.width, args.height))
    else:
        control_image = render_pose_from_coco(args.pose_keypoints, (args.width, args.height))

    generator = GarmentGenerator(
        base_model_id=args.base_model_id,
        controlnet_model_id=args.controlnet_model_id,
        device=args.device,
    )

    image = generator.generate(
        prompt=args.prompt,
        control_image=control_image,
        width=args.width,
        height=args.height,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.neg_prompt,
        controlnet_conditioning_scale=args.strength,
        remove_background=not args.disable_bg_removal,
    )

    image.save(output_path, format="PNG")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()


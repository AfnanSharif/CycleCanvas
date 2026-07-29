from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from cyclegan import CycleGANConfig
from cyclegan.data import list_images
from cyclegan.preview import write_synthetic_domains
from cyclegan.comparison import architecture_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="CycleGAN unpaired image translation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    demo = sub.add_parser("demo"); demo.add_argument("--output", default=str(ROOT / "outputs" / "synthetic_domains")); demo.add_argument("--seed", type=int, default=42)
    inspect = sub.add_parser("inspect-data"); inspect.add_argument("--domain-a", default=str(ROOT / "data" / "trainA")); inspect.add_argument("--domain-b", default=str(ROOT / "data" / "trainB")); inspect.add_argument("--test-a", default=str(ROOT / "data" / "testA")); inspect.add_argument("--test-b", default=str(ROOT / "data" / "testB"))
    train_parser = sub.add_parser("train"); train_parser.add_argument("--domain-a", default=str(ROOT / "data" / "trainA")); train_parser.add_argument("--domain-b", default=str(ROOT / "data" / "trainB")); train_parser.add_argument("--epochs", type=int, default=200); train_parser.add_argument("--decay-start", type=int, default=100); train_parser.add_argument("--image-size", type=int, default=256); train_parser.add_argument("--device", default="auto")
    translate_parser = sub.add_parser("translate"); translate_parser.add_argument("source"); translate_parser.add_argument("--checkpoint", default=str(ROOT / "outputs" / "checkpoint.pt")); translate_parser.add_argument("--output", default=str(ROOT / "outputs" / "translated.jpg")); translate_parser.add_argument("--direction", choices=("AtoB", "BtoA"), default="AtoB")
    evaluate_parser = sub.add_parser("evaluate"); evaluate_parser.add_argument("--checkpoint", default=str(ROOT / "outputs" / "checkpoint.pt")); evaluate_parser.add_argument("--test-a", default=str(ROOT / "data" / "testA")); evaluate_parser.add_argument("--test-b", default=str(ROOT / "data" / "testB")); evaluate_parser.add_argument("--output", default=str(ROOT / "outputs" / "evaluation")); evaluate_parser.add_argument("--device", default="auto")
    compare = sub.add_parser("compare-architectures"); compare.add_argument("--image-size", type=int, default=32); compare.add_argument("--dense-depth", type=int, default=3); compare.add_argument("--hidden-features", type=int, default=256); compare.add_argument("--residual-blocks", type=int, default=3); compare.add_argument("--resnet-features", type=int, default=16); compare.add_argument("--steps", type=int, default=0, help="run a PyTorch learning comparison when greater than zero"); compare.add_argument("--batch-size", type=int, default=4); compare.add_argument("--learning-rate", type=float, default=1e-3); compare.add_argument("--seed", type=int, default=42); compare.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.command == "doctor":
        available = {name: bool(importlib.util.find_spec(name)) for name in ("torch", "torchvision", "PIL")}
        print(json.dumps({"python": platform.python_version(), **available, "training_ready": all(available.values())}, indent=2)); return
    if args.command == "demo":
        print("\n".join(map(str, write_synthetic_domains(args.output, args.seed)))); return
    if args.command == "inspect-data":
        print(json.dumps({"trainA": len(list_images(args.domain_a)), "trainB": len(list_images(args.domain_b)), "testA": len(list_images(args.test_a)), "testB": len(list_images(args.test_b))}, indent=2)); return
    if args.command == "train":
        from cyclegan.training import train
        train(CycleGANConfig(epochs=args.epochs, decay_start_epoch=args.decay_start, image_size=args.image_size, device=args.device), args.domain_a, args.domain_b, ROOT); return
    if args.command == "evaluate":
        from cyclegan.evaluation import evaluate_checkpoint
        print(json.dumps(evaluate_checkpoint(args.checkpoint, args.test_a, args.test_b, args.output, args.device), indent=2)); return
    if args.command == "compare-architectures":
        if args.steps:
            from cyclegan.comparison import run_learning_comparison
            output = run_learning_comparison(
                steps=args.steps,
                image_size=args.image_size,
                batch_size=args.batch_size,
                hidden_features=args.hidden_features,
                dense_depth=args.dense_depth,
                residual_blocks=args.residual_blocks,
                features=args.resnet_features,
                learning_rate=args.learning_rate,
                seed=args.seed,
                device=args.device,
            )
        else:
            output = {
                "profiles": [item.to_dict() for item in architecture_profiles(args.image_size, args.hidden_features, args.dense_depth, args.residual_blocks, args.resnet_features)],
                "hint": "Add --steps 20 to run the controlled PyTorch learning comparison.",
            }
        print(json.dumps(output, indent=2)); return
    from cyclegan.inference import translate
    print(translate(args.checkpoint, args.source, args.output, args.direction))


if __name__ == "__main__": main()

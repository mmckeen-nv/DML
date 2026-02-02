"""Minimal demo for the Transformers backend with STM controller."""
from __future__ import annotations

import argparse

from .dml_adapter import DMLAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="DML Transformers demo")
    parser.add_argument("--hf-model", required=True, help="HuggingFace model name")
    parser.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps")
    parser.add_argument("--dtype", default="auto", help="dtype: auto/float16/bfloat16/float32")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantization")
    parser.add_argument("--load-in-8bit", action="store_true", help="Enable 8-bit quantization")
    parser.add_argument("--enable-stm", action="store_true", help="Enable STM controller")
    args = parser.parse_args()

    overrides = {
        "model_name": args.hf_model,
        "llm_backend": "transformers",
        "llm_device": args.device,
        "llm_dtype": args.dtype,
        "load_in_4bit": bool(args.load_in_4bit),
        "load_in_8bit": bool(args.load_in_8bit),
        "enable_stm_controller": bool(args.enable_stm),
    }

    adapter = DMLAdapter(config_overrides=overrides, start_aging_loop=False)
    session_id = "demo-session"
    prompts = [
        "My favorite drink is jasmine tea.",
        "What drink do I prefer?",
        "Can you remind me which beverage I like best?",
    ]
    try:
        for prompt in prompts:
            result = adapter.generate_with_controller(
                prompt, max_new_tokens=128, session_id=session_id
            )
            print("\nUser:", prompt)
            print("Assistant:", result.get("response"))
            if adapter.enable_stm_controller and adapter.stm_controller:
                stm = adapter.get_stm_state(session_id)
                summary = adapter.stm_controller.build_stm_summary(stm)
                print("\nSTM Summary:\n", summary or "(empty)")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Arabic dataset translation and instruction localization tooling for FastVLA.
Translates English robotics instructions into standardized Arabic task prompts,
generating the translation mapping dictionary consumed by FastVLA DataOrchestrator.
"""

import os
import json
import argparse
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard robotics domain translations for spatial & manipulation terminology
STANDARD_ARABIC_ROBOTICS_LEXICON = {
    "push the block to the goal": "ادفع المكعب إلى الهدف",
    "pick up the red cup": "التقط الكوب الأحمر",
    "place the cup on the plate": "ضع الكوب على الطبق",
    "open the microwave": "افتح الميكروويف",
    "close the drawer": "أغلق الدرج",
    "move the slider forward": "حرك المنزلق للأمام",
    "turn the knob clockwise": "أدر المقبض في اتجاه عقارب الساعة",
    "pick up the can and put it in the bin": "التقط العلبة وضعها في السلة",
    "reach for the green object": "امدد يدك نحو الجسم الأخضر",
    "complete the kitchen task": "أكمل مهمة المطبخ",
}

def translate_instruction(instruction: str, mapping: Dict[str, str]) -> str:
    """Translate an English instruction using existing lexicon or fallback."""
    cleaned = instruction.strip().lower()
    if cleaned in mapping:
        return mapping[cleaned]
    # Check partial matches
    for eng, arab in mapping.items():
        if eng in cleaned:
            return arab
    return mapping.get("complete the kitchen task", "أكمل المهمة المطلوبة")

def process_dataset_instructions(
    input_file: str,
    output_file: str,
    extra_mapping_file: str = None,
):
    """
    Generate an instruction translation JSON file for FastVLA DataOrchestrator.
    """
    mapping = dict(STANDARD_ARABIC_ROBOTICS_LEXICON)
    if extra_mapping_file and os.path.exists(extra_mapping_file):
        with open(extra_mapping_file, "r", encoding="utf-8") as f:
            custom = json.load(f)
            mapping.update(custom)

    results = {}
    if input_file and os.path.exists(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            instructions = data if isinstance(data, list) else list(data.keys())
            for inst in instructions:
                results[inst] = translate_instruction(inst, mapping)
    else:
        results = mapping

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(results)} Arabic instruction translations to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Arabic Dataset Translation Tool for FastVLA")
    parser.add_argument("--input-file", type=str, default="", help="JSON file containing English instructions")
    parser.add_argument("--output-file", type=str, default="data/arabic_instructions.json", help="Output mapping path")
    parser.add_argument("--custom-lexicon", type=str, default=None, help="Custom English-to-Arabic terminology JSON")
    args = parser.parse_args()

    process_dataset_instructions(
        input_file=args.input_file,
        output_file=args.output_file,
        extra_mapping_file=args.custom_lexicon
    )

if __name__ == "__main__":
    main()

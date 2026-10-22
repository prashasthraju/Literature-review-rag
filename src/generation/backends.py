"""Text generators behind one interface: generate(messages) -> str.

Specs: "hf:<model id>" runs a local Hugging Face chat model; "anthropic:<model>"
calls the Anthropic API (needs the anthropic package and ANTHROPIC_API_KEY).
"""

import os


class TransformersGenerator:
    def __init__(self, model_id, max_new_tokens=512, quantize=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.name = f"hf:{model_id}" + (f"@{quantize}" if quantize else "")
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        extra = {}
        if quantize == "4bit":
            # 4-bit NF4 leaves room on an 8 GB GPU for the embedder and reranker.
            extra["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif quantize:
            raise ValueError(f"Unsupported quantize={quantize!r}; use '4bit' or null")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            **extra,
        )
        self.model.eval()

    def generate(self, messages, prefill=""):
        """Greedy decoding; ``prefill`` starts the reply (e.g. '{"status": "') to force the format."""
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(prompt + prefill, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return prefill + self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class AnthropicGenerator:
    def __init__(self, model, max_tokens=1024):
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY to use an anthropic: generator")
        self.name = f"anthropic:{model}"
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()

    def generate(self, messages, prefill=""):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        if prefill:
            turns.append({"role": "assistant", "content": prefill})
        response = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=turns
        )
        return prefill + "".join(block.text for block in response.content if block.type == "text")


def make_generator(spec, **kwargs):
    kind, _, name = spec.partition(":")
    if kind == "hf":
        return TransformersGenerator(name, **kwargs)
    if kind == "anthropic":
        return AnthropicGenerator(name, **kwargs)
    raise ValueError(f"Unknown generator spec {spec!r}; use hf:<model> or anthropic:<model>")

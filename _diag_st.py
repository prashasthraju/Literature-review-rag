from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
print("max_seq_length", model.max_seq_length)
print("prompts", getattr(model, "prompts", None))
print("default_prompt_name", getattr(model, "default_prompt_name", None))
print("prompt_names", getattr(model, "prompt_names", None))
cfg = getattr(model, "_model_config", None) or getattr(model, "model_card_data", None)
print("config type", type(cfg))
if hasattr(model, "get_max_seq_length"):
    print("get_max_seq_length", model.get_max_seq_length())

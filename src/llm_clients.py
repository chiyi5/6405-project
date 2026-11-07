"""
Optional LLM back-ends. Every function returns a callable llm_fn(prompt: str) -> str,
which can be passed to LLMSkillExtractor (Member B) or llm_feedback (Member F).
Set the API key as an environment variable; never hard-code keys in the notebook.
"""
import os


def openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def fn(prompt: str) -> str:
        r = client.chat.completions.create(model=model, temperature=temperature,
                                           messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content
    return fn


def anthropic_llm(model: str = "claude-haiku-5-5", temperature: float = 0.0):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def fn(prompt: str) -> str:
        r = client.messages.create(model=model, max_tokens=1024, temperature=temperature,
                                   messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    return fn


def hf_local_llm(model: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 512):
    """Free local model (GPU recommended). A T4-class GPU is enough."""
    from transformers import pipeline
    gen = pipeline("text-generation", model=model, device_map="auto")

    def fn(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        out = gen(msgs, max_new_tokens=max_new_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"]
    return fn

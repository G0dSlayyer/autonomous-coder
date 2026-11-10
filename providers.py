"""One interface in front of every LLM backend.

    p = get_provider("groq:openai/gpt-oss-20b")
    p = get_provider("ollama:qwen2.5-coder:7b")
    p.chat(messages, temperature=0.2)            # multi-turn  -> Completion
    p.generate(prompt, temperature=0.8, n=5)     # one prompt  -> [Completion] * n
"""
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class DailyCap(Exception):
    """Provider's rate limit won't reset for a long time; stop the run."""


@dataclass
class Completion:
    text: str
    tokens: int = 0


def _messages(prompt, system):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


class Provider:
    name = "base"

    def _chat(self, messages, temperature, max_tokens) -> Completion:
        raise NotImplementedError

    def chat(self, messages, temperature=0.0, max_tokens=None) -> Completion:
        """messages = [{"role": "system"|"user"|"assistant", "content": ...}]"""
        return self._chat(messages, temperature, max_tokens)

    def generate(self, prompt, system=None, temperature=0.0, n=1,
                 max_tokens=None, workers=4):
        """Single prompt -> list of n Completions (sampled in parallel)."""
        messages = _messages(prompt, system)
        call = lambda _: self._chat(messages, temperature, max_tokens)
        if n == 1:
            return [call(0)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(call, range(n)))


class GroqProvider(Provider):
    def __init__(self, model="openai/gpt-oss-20b"):
        from groq import Groq  # pip install groq
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model
        self.name = f"groq:{model}"
        self._check_model()

    def available_models(self):
        return sorted(m.id for m in self.client.models.list().data)

    def _check_model(self):
        """Fail fast with the valid names instead of 404-ing on every problem."""
        try:
            models = self.available_models()
        except Exception:
            return  # can't list (network hiccup etc.); let real calls report errors
        if self.model not in models:
            raise ValueError(
                f"Groq model '{self.model}' is not available to your API key.\n"
                "Models you can use:\n  " + "\n  ".join(models))

    def _chat(self, messages, temperature, max_tokens):
        from groq import RateLimitError
        kwargs = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if "gpt-oss" in self.model:
            kwargs["extra_body"] = {"reasoning_effort": "low"}  # fewer hidden reasoning tokens

        for _ in range(8):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except RateLimitError as e:  # free tier: wait out the per-minute token limit
                m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(e))
                wait = (int(m.group(1) or 0) * 60 + float(m.group(2)) + 1) if m else 15
                if wait > 120:
                    raise DailyCap(f"resets in ~{wait/60:.0f} min")
                time.sleep(wait)
        else:
            raise RuntimeError("Rate limit retries exhausted")

        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return Completion(text, tokens)


class OllamaProvider(Provider):
    def __init__(self, model="qwen2.5-coder:7b", host="http://localhost:11434"):
        self.model, self.host = model, host
        self.name = f"ollama:{model}"

    def _chat(self, messages, temperature, max_tokens):
        import requests  # pip install requests
        options = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        r = requests.post(
            f"{self.host}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False, "options": options},
            timeout=600,
        )
        r.raise_for_status()
        d = r.json()
        tokens = d.get("prompt_eval_count", 0) + d.get("eval_count", 0)
        return Completion(d["message"]["content"], tokens)


def get_provider(spec: str) -> Provider:
    kind, _, model = spec.partition(":")
    if kind == "groq":
        return GroqProvider(model or "openai/gpt-oss-20b")
    if kind == "ollama":
        return OllamaProvider(model or "qwen2.5-coder:7b")
    raise ValueError(f"unknown provider '{kind}' (use groq:<model> or ollama:<model>)")


def extract_code(text: str) -> str:
    """Pull Python out of an LLM reply: prefer the first fenced block with a def."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    for b in blocks:
        if "def " in b:
            return b.strip()
    return blocks[0].strip() if blocks else text.strip()


if __name__ == "__main__":
    # python providers.py  ->  list the Groq models your API key can use
    from groq import Groq
    for m in sorted(x.id for x in Groq(api_key=os.environ["GROQ_API_KEY"]).models.list().data):
        print(m)
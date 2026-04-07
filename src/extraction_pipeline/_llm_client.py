"""litellm API client replacing vLLM for vision-language model inference."""

from __future__ import annotations

import base64
import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import litellm
except ModuleNotFoundError:
    litellm = None


def _load_openrouter_api_key() -> None:
    """Load OpenRouter API key from openrouter_key file if not already set."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    key_file = Path(__file__).resolve().parents[3] / "openrouter_key"
    if key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            os.environ["OPENROUTER_API_KEY"] = key


_load_openrouter_api_key()


def image_to_data_uri(image_path: Path) -> str:
    """Convert a local image file to a base64 data URI."""
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:image/png;base64,{b64}"


def pdf_bytes_to_data_uri(data: bytes) -> str:
    """Convert PDF bytes to a base64 data URI."""
    b64 = base64.b64encode(data).decode()
    return f"data:application/pdf;base64,{b64}"


def build_visual_context(
    *,
    pdf_path: Path | None = None,
    entries: list[tuple[str, int, Path | None]],
    use_pdf: bool = True,
) -> list[dict]:
    """Build content parts for visual context with optional labels.

    Args:
        pdf_path: Source PDF path (required when use_pdf=True).
        entries: List of (label, page_number, image_path_or_None) tuples.
            label: Text label for the page (e.g. "TARGET page", ""). Empty string = no label.
            page_number: 1-indexed page number.
            image_path: Path to rendered image (used when use_pdf=False).
        use_pdf: If True, extract pages from pdf_path as a single multi-page PDF.
            If False, use individual image files.

    Returns:
        List of content part dicts for use in message content arrays.
    """
    if use_pdf and pdf_path is not None:
        from extraction_pipeline._pdf_renderer import extract_pdf_page_bytes

        parts: list[dict] = []
        page_numbers = [page_num for _, page_num, _ in entries]
        for label, page_num, _ in entries:
            if label:
                parts.append({"type": "text", "text": f"{label}: page {page_num}"})
        pdf_data = extract_pdf_page_bytes(pdf_path, page_numbers)
        uri = pdf_bytes_to_data_uri(pdf_data)
        parts.append({"type": "image_url", "image_url": {"url": uri}})
        return parts
    else:
        parts: list[dict] = []
        for label, page_num, image_path in entries:
            if label:
                parts.append({"type": "text", "text": f"{label}: page {page_num}"})
            if image_path is not None:
                parts.append(
                    {"type": "image_url", "image_url": {"url": f"file://{image_path.resolve()}"}}
                )
        return parts


def _convert_file_urls_to_base64(messages: list[dict]) -> list[dict]:
    """Convert file:// image URLs in messages to base64 data URIs."""
    converted = []
    for msg in messages:
        msg = copy.deepcopy(msg)
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "image_url"
                    and isinstance(item.get("image_url"), dict)
                ):
                    url = item["image_url"].get("url", "")
                    if url.startswith("file://"):
                        local_path = Path(url.removeprefix("file://"))
                        item["image_url"]["url"] = image_to_data_uri(
                            local_path
                        )
        converted.append(msg)
    return converted


class LLMClient:
    """litellm-based client for vision-language model inference.

    Replaces vLLM's LLM class with API-based inference via litellm.
    Supports OpenRouter and any litellm-compatible provider.
    """

    def __init__(
        self,
        model: str = "openrouter/qwen/qwen3.5-flash-02-23",
        max_concurrency: int = 8,
    ):
        self.model = model
        self.max_concurrency = max_concurrency
        self._total_cost: float = 0.0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._request_count: int = 0
        self._lock = __import__("threading").Lock()

    _RETRY_DELAYS = (16.0, 32.0)

    def _call_with_retry(self, completion_kwargs: dict):
        """Call litellm.completion with backoff on rate-limit errors (max 2 retries)."""
        if litellm is None:
            raise ModuleNotFoundError("litellm is required for extraction_pipeline. Install litellm to run the extraction pipeline.")
        for attempt in range(len(self._RETRY_DELAYS) + 1):
            try:
                return litellm.completion(**completion_kwargs)
            except (litellm.exceptions.RateLimitError, litellm.exceptions.ServiceUnavailableError):
                if attempt >= len(self._RETRY_DELAYS):
                    raise
                time.sleep(self._RETRY_DELAYS[attempt])
            except litellm.exceptions.APIError as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    if attempt >= len(self._RETRY_DELAYS):
                        raise
                    time.sleep(self._RETRY_DELAYS[attempt])
                else:
                    raise

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float | None = None,
        response_format: dict | None = None,
        n: int = 1,
        **kwargs,
    ) -> list[str]:
        """Send a single chat request. Returns list of n response texts."""
        converted_messages = _convert_file_urls_to_base64(messages)

        completion_kwargs: dict = {
            "model": self.model,
            "messages": converted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n,
        }
        if top_p is not None:
            completion_kwargs["top_p"] = top_p
        if response_format is not None:
            completion_kwargs["response_format"] = response_format
        # Pass through extra litellm kwargs (e.g. extra_body)
        completion_kwargs.update(kwargs)

        response = self._call_with_retry(completion_kwargs)

        with self._lock:
            self._request_count += 1
            cost = response._hidden_params.get("response_cost", 0) or 0
            self._total_cost += cost
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self._total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0

        return [
            choice.message.content or ""
            for choice in response.choices
        ]

    def batch_chat(
        self,
        messages_list: list[list[dict]],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float | None = None,
        response_format: dict | None = None,
        n: int = 1,
        **kwargs,
    ) -> list[list[str]]:
        """Send multiple chat requests concurrently.

        Returns list of lists, where each inner list contains n response texts.
        """
        if not messages_list:
            return []

        results: list[list[str] | None] = [None] * len(messages_list)

        def _process(idx: int, messages: list[dict]) -> tuple[int, list[str]]:
            return idx, self.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                response_format=response_format,
                n=n,
                **kwargs,
            )

        with ThreadPoolExecutor(
            max_workers=min(self.max_concurrency, len(messages_list))
        ) as executor:
            futures = [
                executor.submit(_process, idx, messages)
                for idx, messages in enumerate(messages_list)
            ]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return [r if r is not None else [""] for r in results]

    def get_usage(self) -> dict:
        """Return accumulated usage stats."""
        with self._lock:
            return {
                "total_cost": self._total_cost,
                "prompt_tokens": self._total_prompt_tokens,
                "completion_tokens": self._total_completion_tokens,
                "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
                "request_count": self._request_count,
            }

    def print_usage_summary(self, label: str = "") -> None:
        """Print a cost/usage summary to stdout.

        If the environment variable ``COST_LOG_PATH`` is set, also appends
        a JSON record to that file so the calling script can aggregate
        per-PDF totals.
        """
        u = self.get_usage()
        prefix = f"[{label}] " if label else ""
        print(
            f"{prefix}LLM usage: requests={u['request_count']}, "
            f"prompt_tokens={u['prompt_tokens']:,}, "
            f"completion_tokens={u['completion_tokens']:,}, "
            f"total_tokens={u['total_tokens']:,}, "
            f"cost=${u['total_cost']:.4f}"
        )
        cost_log = os.environ.get("COST_LOG_PATH")
        if cost_log:
            import json
            log_path = Path(cost_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"label": label, **u}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def reset_usage(self) -> None:
        """Reset accumulated usage stats."""
        with self._lock:
            self._total_cost = 0.0
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._request_count = 0

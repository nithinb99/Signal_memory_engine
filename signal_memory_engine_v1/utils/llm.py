# ============================================================================
# utils/llm.py
# ============================================================================

from fastapi import HTTPException, status


def invoke_chain(chain, prompt: str) -> str:
    """Invoke a LangChain chain safely, handling 429 / quota as 503."""
    try:
        out = chain.invoke({"query": prompt})
        if isinstance(out, dict) and "result" in out:
            return out["result"]
        if isinstance(out, str):
            return out
        return str(out)
    except Exception as e:
        detail = (
            "LLM quota/rate limit hit (OpenAI 429 / insufficient_quota). "
            "Set a valid key/billing, switch model/provider, or configure a fallback."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail
        ) from e

"""`Settings.llm_provider` is required (see config/settings.py) - the real
FastAPI app builds a real chat model at startup (app/lifespan.py) even in
tests that immediately override it via `app.dependency_overrides[get_graph]`
(the override only replaces what a request handler sees; startup still has
to succeed first). Every test in this codebase drives its own scripted
FakeToolCallingChatModel instead, so these values are never actually used to
call a real API - they just need to be present and syntactically valid so
`Settings()`/`init_chat_model(...)` construct without error. `setdefault` so
a real ANTHROPIC_API_KEY in the environment (e.g. a developer intentionally
testing against a real provider) isn't clobbered.
"""
import os

os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-not-a-real-key")

# Same idea for the Entra ID / Power BI sign-in settings (config/settings.py)
# - required for Settings() to construct, never actually used since tests
# either inject fake Power BI clients directly (see
# tests/integration/test_datasource_graph.py) or override
# app.api.get_pbi_tokens (see tests/e2e/test_api_chat.py).
os.environ.setdefault("ENTRA_TENANT_ID", "test-tenant")
os.environ.setdefault("ENTRA_CLIENT_ID", "test-client-id")
os.environ.setdefault("ENTRA_CLIENT_SECRET", "test-not-a-real-secret")
os.environ.setdefault("ENTRA_REDIRECT_URI", "http://localhost:8000/auth/callback")

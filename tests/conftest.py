import inspect

import pytest


@pytest.fixture(autouse=True)
def _allow_prompt_boxed_extra_kwargs(monkeypatch: pytest.MonkeyPatch):
    """Many tests monkeypatch main._prompt_boxed with small fakes.

    The real _prompt_boxed evolves (e.g. adding meta line args). To keep tests
    focused on behavior rather than signature, wrap any patched _prompt_boxed
    to ignore unknown kwargs.
    """

    import main as main_mod

    orig_setattr = monkeypatch.setattr

    def _setattr(obj, name, value, *args, **kwargs):
        if obj is main_mod and name == "_prompt_boxed" and callable(value):
            sig = None
            try:
                sig = inspect.signature(value)
            except Exception:
                sig = None

            if sig is not None and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):

                def _wrapper(*w_args, **w_kwargs):
                    w_kwargs.pop("state", None)
                    w_kwargs.pop("token_counter", None)
                    return value(*w_args, **w_kwargs)

                return orig_setattr(obj, name, _wrapper, *args, **kwargs)

        return orig_setattr(obj, name, value, *args, **kwargs)

    monkeypatch.setattr = _setattr
    try:
        yield
    finally:
        monkeypatch.setattr = orig_setattr

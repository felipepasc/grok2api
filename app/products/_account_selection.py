"""Shared account selection helpers for products-layer request handlers."""

from app.control.model.enums import ModeId
from app.control.model.spec import ModelSpec
from app.control.account.runtime import get_refresh_service
from app.dataplane.account.selector import current_strategy
from app.platform.config.snapshot import get_config
from app.platform.errors import AppError, ErrorKind, RateLimitError

# Random strategy has no config key for retry count; it is pinned here so that
# every retry-driven call site (chat / images / video / anthropic) sees the same
# value without introducing scattered magic numbers.
_RANDOM_MAX_RETRIES = 5


def selection_max_retries() -> int:
    """Retry count for account-swap loops, aware of the active selection strategy.

    - ``random`` strategy: fixed at :data:`_RANDOM_MAX_RETRIES` (=5).
    - ``quota`` strategy:  reads ``retry.max_retries`` (default 1), preserving
      the historical behaviour.
    """
    if current_strategy() == "random":
        return _RANDOM_MAX_RETRIES
    return int(get_config("retry.max_retries", 1))


def mode_candidates(spec: ModelSpec) -> tuple[int, ...]:
    """Return mode IDs to try for *spec* in priority order.

    Chat models using ``AUTO`` can optionally fall back to ``FAST`` and then
    ``EXPERT`` when the upstream ``auto`` quota window is exhausted but the
    account still has usable quota in the other chat windows.
    """
    primary = int(spec.mode_id)
    if (
        spec.is_chat()
        and spec.mode_id == ModeId.AUTO
        and get_config("features.auto_chat_mode_fallback", True)
    ):
        return (primary, int(ModeId.FAST), int(ModeId.EXPERT))
    return (primary,)


def normalize_account(value: str | None) -> str | None:
    """Return a cleaned explicit-account token, or None when not provided."""
    if value is None:
        return None
    token = value.strip()
    if token.startswith("sso="):
        token = token[4:]
    return token or None


async def reserve_explicit_account(
    directory,
    token: str,
    mode_id: int,
    *,
    require_quota: bool = True,
    now_s_override: int | None = None,
):
    """Reserve a caller-specified account, raising a mapped error on failure.

    Used when a request pins execution to a single account: there is no
    failover to other accounts, and an exhausted/invalid account surfaces
    the corresponding error directly to the API caller.
    """
    lease, reason = await directory.reserve_token(
        token,
        mode_id,
        require_quota=require_quota,
        now_s_override=now_s_override,
    )
    if lease is not None:
        return lease
    if reason == "not_found":
        raise AppError(
            "Specified account was not found",
            kind=ErrorKind.VALIDATION,
            code="account_not_found",
            status=404,
            details={"param": "account"},
        )
    if reason == "inactive":
        raise AppError(
            "Specified account is not active",
            kind=ErrorKind.VALIDATION,
            code="account_inactive",
            status=409,
            details={"param": "account"},
        )
    raise RateLimitError("Specified account has no remaining quota")


async def reserve_account(
    directory,
    spec: ModelSpec,
    *,
    exclude_tokens: list[str] | None = None,
    now_s_override: int | None = None,
):
    """Reserve an account and return ``(lease, selected_mode_id)``.

    Returns ``(None, original_mode_id)`` when no account is available. Under the
    random strategy no on-demand refresh fallback is attempted — upstream quota
    data is never probed.
    """
    original_mode_id = int(spec.mode_id)

    async def _try_reserve():
        for candidate_mode_id in mode_candidates(spec):
            lease = await directory.reserve(
                pool_candidates=spec.pool_candidates(),
                mode_id=candidate_mode_id,
                now_s_override=now_s_override,
                exclude_tokens=exclude_tokens,
            )
            if lease is not None:
                return lease, candidate_mode_id
        return None, original_mode_id

    lease, selected_mode_id = await _try_reserve()
    if lease is not None:
        return lease, selected_mode_id

    if current_strategy() == "random":
        return None, original_mode_id

    refresh_svc = get_refresh_service()
    if refresh_svc is not None:
        await refresh_svc.refresh_on_demand()
        lease, selected_mode_id = await _try_reserve()
        if lease is not None:
            return lease, selected_mode_id

    return None, original_mode_id

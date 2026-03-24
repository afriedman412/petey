"""
Process-wide concurrency pools for CPU and API work.

Two pools:
  - **CPU** — local parsing (pymupdf, pdfplumber, tesseract).
    Backed by a ``ProcessPoolExecutor``, sized to core count.
  - **API** — remote calls (LLM, Marker, Chandra, Mistral OCR).
    Bounded by an ``asyncio.Semaphore``, user-configurable.

Usage::

    from petey.concurrency import get_manager

    mgr = get_manager()

    # CPU-bound work
    result = await mgr.run_cpu(sync_fn, arg1, arg2)

    # API-bound work
    async with mgr.api():
        result = await async_fn()

    # Or let the manager dispatch automatically
    result = await mgr.run(fn, *args)
    # sync fn → CPU pool, async fn → API semaphore
"""
import asyncio
import concurrent.futures
import os
from typing import Any, Callable


class ConcurrencyManager:
    """Shared concurrency pools for CPU and API work."""

    def __init__(
        self,
        cpu_limit: int | None = None,
        api_limit: int = 10,
    ):
        self._cpu_limit = cpu_limit or os.cpu_count() or 4
        self._api_limit = api_limit
        self._cpu_sem: asyncio.Semaphore | None = None
        self._api_sem: asyncio.Semaphore | None = None
        self._cpu_pool: concurrent.futures.ProcessPoolExecutor | None = None
        self._loop_id: int | None = None

    def _ensure_initialized(self):
        """Lazy-init semaphores bound to the current event loop."""
        loop_id = id(asyncio.get_running_loop())
        if self._loop_id != loop_id:
            self._cpu_sem = asyncio.Semaphore(self._cpu_limit)
            self._api_sem = asyncio.Semaphore(self._api_limit)
            self._loop_id = loop_id

    @property
    def cpu_limit(self) -> int:
        return self._cpu_limit

    @property
    def api_limit(self) -> int:
        return self._api_limit

    def api(self):
        """Context manager to acquire an API slot.

        Usage::

            async with mgr.api():
                result = await some_api_call()
        """
        self._ensure_initialized()
        return self._api_sem

    def cpu_sem(self):
        """Context manager to acquire a CPU slot.

        Prefer ``run_cpu()`` which also handles the executor.
        """
        self._ensure_initialized()
        return self._cpu_sem

    def get_cpu_pool(self) -> concurrent.futures.ProcessPoolExecutor:
        """Shared ProcessPoolExecutor, sized to cpu_limit."""
        if self._cpu_pool is None:
            self._cpu_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self._cpu_limit,
            )
        return self._cpu_pool

    async def run_cpu(self, fn: Callable, *args: Any) -> Any:
        """Run a sync function in the CPU pool with concurrency control."""
        self._ensure_initialized()
        loop = asyncio.get_running_loop()
        async with self._cpu_sem:
            return await loop.run_in_executor(
                self.get_cpu_pool(), fn, *args,
            )

    async def run(self, fn: Callable, *args: Any) -> Any:
        """Dispatch automatically: async fn → API pool, sync fn → CPU pool."""
        if asyncio.iscoroutinefunction(fn):
            async with self.api():
                return await fn(*args)
        else:
            return await self.run_cpu(fn, *args)

    def configure(
        self,
        *,
        cpu_limit: int | None = None,
        api_limit: int | None = None,
    ):
        """Update limits. Semaphores are re-created on next use."""
        if cpu_limit is not None:
            self._cpu_limit = cpu_limit
        if api_limit is not None:
            self._api_limit = api_limit
        # Force re-init on next access
        self._loop_id = None
        if cpu_limit is not None and self._cpu_pool is not None:
            self._cpu_pool.shutdown(wait=False)
            self._cpu_pool = None

    def shutdown(self):
        """Clean up the process pool."""
        if self._cpu_pool is not None:
            self._cpu_pool.shutdown(wait=False)
            self._cpu_pool = None


# --- Module-level singleton ---

_default_manager: ConcurrencyManager | None = None


def get_manager() -> ConcurrencyManager:
    """Get or create the process-wide ConcurrencyManager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ConcurrencyManager()
    return _default_manager


def configure(*, cpu_limit: int | None = None, api_limit: int | None = None):
    """Configure the global ConcurrencyManager."""
    get_manager().configure(cpu_limit=cpu_limit, api_limit=api_limit)

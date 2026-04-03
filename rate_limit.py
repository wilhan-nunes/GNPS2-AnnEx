import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass

import fcntl
from flask import jsonify, request


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitRule:
    name: str
    max_requests: int
    window_seconds: int


class FileBackedSlidingWindowLimiter:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def hit(self, key: str, window_seconds: int, now: float | None = None) -> int:
        now = now or time.time()
        bucket_path = self._bucket_path(key)
        window_start = now - window_seconds

        with open(bucket_path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            raw_timestamps = handle.read().strip()
            timestamps = json.loads(raw_timestamps) if raw_timestamps else []
            timestamps = [ts for ts in timestamps if ts >= window_start]
            timestamps.append(now)

            handle.seek(0)
            handle.truncate()
            json.dump(timestamps, handle)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        return len(timestamps)

    def _bucket_path(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.storage_dir, f"{digest}.json")


class AppRateLimiter:
    ASSET_RULE = RateLimitRule("assets", max_requests=120, window_seconds=10)
    DASH_READ_RULE = RateLimitRule("dash-read", max_requests=40, window_seconds=10)
    DASH_UPDATE_RULE = RateLimitRule("dash-update", max_requests=30, window_seconds=10)
    PAGE_RULE = RateLimitRule("pages", max_requests=20, window_seconds=10)
    UNKNOWN_RULE = RateLimitRule("unknown", max_requests=4, window_seconds=10)
    NON_DASH_WRITE_RULE = RateLimitRule("non-dash-write", max_requests=3, window_seconds=10)

    PAGE_PATHS = {
        "/",
        "/dashinterface1",
        "/dashinterface1/",
        "/dashinterface2",
        "/dashinterface2/",
        "/favicon.ico",
    }
    ASSET_PREFIXES = (
        "/assets/",
        "/dashinterface2/assets/",
        "/dashinterface2/_dash-component-suites/",
    )
    DASH_READ_PATHS = {
        "/dashinterface2/_dash-layout",
        "/dashinterface2/_dash-dependencies",
    }
    DASH_UPDATE_PATH = "/dashinterface2/_dash-update-component"
    SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

    def __init__(self, app, storage_dir: str):
        self.app = app
        self.counter = FileBackedSlidingWindowLimiter(storage_dir)
        self._register_hooks()

    def _register_hooks(self) -> None:
        @self.app.before_request
        def _enforce_rate_limit():
            limit_state = self._check_request()
            if limit_state is None:
                return None

            response_body = {
                "error": "rate_limited",
                "limit": limit_state["rule"].max_requests,
                "window_seconds": limit_state["rule"].window_seconds,
                "retry_after_seconds": limit_state["rule"].window_seconds,
                "scope": limit_state["rule"].name,
            }
            logger.warning(
                "Rate limited request from %s for %s %s on rule=%s count=%s limit=%s/%ss",
                limit_state["client_id"],
                request.method,
                request.path,
                limit_state["rule"].name,
                limit_state["count"],
                limit_state["rule"].max_requests,
                limit_state["rule"].window_seconds,
            )

            if request.path.startswith("/dashinterface2/_dash-") or request.accept_mimetypes.accept_json:
                response = jsonify(response_body)
            else:
                response = jsonify(response_body)

            response.status_code = 429
            response.headers["Retry-After"] = str(limit_state["rule"].window_seconds)
            return response

    def _check_request(self):
        client_id = self._get_client_id()
        now = time.time()

        for rule in self._rules_for_request():
            counter_key = f"{rule.name}:{client_id}"
            count = self.counter.hit(counter_key, window_seconds=rule.window_seconds, now=now)
            if count > rule.max_requests:
                return {"rule": rule, "count": count, "client_id": client_id}

        return None

    def _rules_for_request(self) -> list[RateLimitRule]:
        path = request.path or "/"
        rules: list[RateLimitRule] = []

        if path == self.DASH_UPDATE_PATH:
            rules.append(self.DASH_UPDATE_RULE)
        elif path in self.DASH_READ_PATHS:
            rules.append(self.DASH_READ_RULE)
        elif path in self.PAGE_PATHS:
            rules.append(self.PAGE_RULE)
        elif path.startswith(self.ASSET_PREFIXES):
            rules.append(self.ASSET_RULE)
        else:
            rules.append(self.UNKNOWN_RULE)

        if request.method not in self.SAFE_METHODS and path != self.DASH_UPDATE_PATH:
            rules.append(self.NON_DASH_WRITE_RULE)

        return rules

    def _get_client_id(self) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        if request.access_route:
            return request.access_route[0]

        return request.remote_addr or "unknown"

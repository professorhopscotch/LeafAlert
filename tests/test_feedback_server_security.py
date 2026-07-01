"""Security tests for scripts/feedback_server.py filename handling.

These tests pin the *intended* behavior of the upload-path sanitizer: a
client-supplied filename must never be able to escape the configured output
directory. The server reduces any incoming name to its basename and confirms
the resolved path stays inside ``output_dir``; traversal attempts and absolute
paths are rejected (HTTP 400), while a normal feedback filename is accepted.

The sanitizer (``FeedbackHandler._safe_image_path``) is a bound method that
writes an HTTP response on rejection, so it is awkward to call in isolation.
We therefore drive it through a tiny fake handler that records the rejection
instead of writing to a socket. This exercises the *real* containment logic
from the module rather than a copy.

Stdlib + pytest only — no torch / coremltools / network required.

NOTE: depends on the feedback_server hardening (the ``_safe_image_path`` /
basename-containment logic). If that method is absent, the tests below xfail
with a clear message until the sibling fix lands.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "feedback_server.py"
)


def _load_feedback_server():
    """Import scripts/feedback_server.py as a module by file path.

    Returns the module, or None if it cannot be imported (e.g. an optional
    dependency is missing). The module only depends on the stdlib plus an
    optional pyobjc import that is already guarded with try/except, so this
    should succeed on a bare CI runner.
    """
    spec = importlib.util.spec_from_file_location(
        "leafalert_feedback_server", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeHandler:
    """Minimal stand-in that borrows the real ``_safe_image_path`` method.

    It records whatever ``_respond`` would have sent so the test can assert a
    rejection happened, without needing a real HTTP connection.
    """

    def __init__(self, safe_image_path):
        self.responses = []
        # Bind the unbound function from the class onto this fake instance.
        self._safe_image_path = safe_image_path.__get__(self, _FakeHandler)

    def _respond(self, code, data):
        self.responses.append((code, data))

    @property
    def rejected(self):
        return bool(self.responses)


@pytest.fixture(scope="module")
def safe_image_path_fn():
    module = _load_feedback_server()
    if module is None:
        pytest.xfail("feedback_server.py could not be imported")
    handler_cls = getattr(module, "FeedbackHandler", None)
    if handler_cls is None:
        pytest.xfail("FeedbackHandler not found in feedback_server.py")
    fn = getattr(handler_cls, "_safe_image_path", None)
    if fn is None:
        pytest.xfail(
            "FeedbackHandler._safe_image_path not present yet "
            "(awaiting feedback_server hardening)"
        )
    return fn


def _resolve(safe_image_path_fn, output_dir: Path, filename: str):
    """Run the real sanitizer and return (path_or_None, was_rejected)."""
    handler = _FakeHandler(safe_image_path_fn)
    result = handler._safe_image_path(output_dir, filename)
    return result, handler.rejected


@pytest.mark.parametrize(
    "malicious",
    [
        "../../etc/passwd",
        "/abs/path.jpg",
        "..\\x",
        "../manifest.json",
        "../../../../tmp/evil.jpg",
        "subdir/../../escape.jpg",
    ],
)
def test_malicious_filenames_are_contained_or_rejected(
    safe_image_path_fn, tmp_path, malicious
):
    """Traversal / absolute paths never resolve outside output_dir.

    Acceptable outcomes: the sanitizer rejects the name (returns None and
    records a 400), OR it collapses the name to a basename that resolves
    strictly inside output_dir. Either way the result must not escape.
    """
    output_dir = tmp_path / "feedback"
    output_dir.mkdir()

    result, rejected = _resolve(safe_image_path_fn, output_dir, malicious)

    if result is None:
        # Rejected outright — must have signaled a client error.
        assert rejected, "rejected path must send an error response"
    else:
        resolved = result.resolve()
        # Contained strictly within output_dir...
        assert resolved.is_relative_to(output_dir.resolve()), (
            f"{malicious!r} resolved to {resolved} which escapes "
            f"{output_dir.resolve()}"
        )
        # ...and reduced to just a basename (no directory components survive).
        assert result.name == Path(malicious).name


def test_normal_filename_is_accepted(safe_image_path_fn, tmp_path):
    """A well-formed feedback filename passes and lands inside output_dir."""
    output_dir = tmp_path / "feedback"
    output_dir.mkdir()

    result, rejected = _resolve(
        safe_image_path_fn, output_dir, "feedback_123.jpg"
    )

    assert not rejected, f"normal filename was rejected: {rejected}"
    assert result is not None
    assert result.name == "feedback_123.jpg"
    assert result.resolve().is_relative_to(output_dir.resolve())
    assert result.parent.resolve() == output_dir.resolve()


def test_basename_of_traversal_is_safe_basename(safe_image_path_fn, tmp_path):
    """Sanity check: '../../etc/passwd' must never write to /etc/passwd."""
    output_dir = tmp_path / "feedback"
    output_dir.mkdir()

    result, _ = _resolve(safe_image_path_fn, output_dir, "../../etc/passwd")

    if result is not None:
        assert result.resolve() != Path("/etc/passwd")
        assert result.resolve().is_relative_to(output_dir.resolve())

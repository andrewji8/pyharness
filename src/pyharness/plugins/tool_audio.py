"""Audio transcription tool plugin for PyHarness.

Provides the ``audio_transcribe`` tool: an Agent hands over an audio file
(``source`` path) or raw base64 ``data`` and receives a text transcript.

The transcription core is the **module-level** :func:`transcribe_bytes`
(rather than a method on the tool class) so the Web UI (batch 3) and any other
surface can reuse the exact same backend logic without round-tripping through
the agent loop.

Backend probing order
---------------------
1. ``OPENAI_API_KEY`` is set → OpenAI Whisper ``/v1/audio/transcriptions``
   (``whisper-1`` by default, overridable via ``PYHARNESS_WHISPER_MODEL``).
2. ``faster-whisper`` is importable → local, on-device transcription
   (lazy import; the heavy dependency is only loaded when actually used).
3. neither → :class:`TranscribeUnavailable` (the tool returns a friendly
   :class:`ToolResult` instead of crashing the agent loop).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from pluggy import HookimplMarker

from pyharness.context import SessionContext
from pyharness.schema import ToolArg, ToolResult, ToolResultStatus, ToolSpec

logger = logging.getLogger(__name__)
hookimpl = HookimplMarker("pyharness")


class TranscribeUnavailable(RuntimeError):
    """Raised when no transcription backend is configured/available."""


OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
_DEFAULT_MIME = "audio/mpeg"


# --------------------------------------------------------------------------- #
# Transcription core (module-level, reusable by web_ui in batch 3)
# --------------------------------------------------------------------------- #
async def transcribe_bytes(data: bytes, mime: str) -> str:
    """Transcribe raw audio ``data`` into text, probing available backends.

    Order: OpenAI Whisper (if ``OPENAI_API_KEY``) → faster-whisper (if
    importable) → raise :class:`TranscribeUnavailable`.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        return await _transcribe_openai(data, mime, api_key)

    if _faster_whisper_available():
        return await asyncio.to_thread(_transcribe_faster_whisper, data, mime)

    raise TranscribeUnavailable("无可用转写后端（需 OPENAI_API_KEY 或 faster-whisper）")


async def _transcribe_openai(data: bytes, mime: str, api_key: str) -> str:
    """OpenAI Whisper transcription over httpx multipart upload."""
    base_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    url = f"{base_url}/audio/transcriptions" if base_url else OPENAI_TRANSCRIBE_URL
    model = os.getenv("PYHARNESS_WHISPER_MODEL", "whisper-1")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("audio", data, mime or _DEFAULT_MIME)},
            data={"model": model},
        )
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()

    text = payload.get("text")
    if not text:
        raise RuntimeError(f"OpenAI 转写返回空文本: {payload}")
    return text


def _faster_whisper_available() -> bool:
    """Best-effort check for the (heavy) faster-whisper package."""
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


def _transcribe_faster_whisper(data: bytes, mime: str) -> str:
    """Local transcription via faster-whisper (CPU)."""
    import io

    import soundfile as sf
    from faster_whisper import WhisperModel

    buf = io.BytesIO(data)
    with sf.SoundFile(buf) as snd:
        audio = snd.read(dtype="float32")
        sample_rate = snd.samplerate

    model = WhisperModel(os.getenv("PYHARNESS_WHISPER_MODEL", "base"), device="cpu")
    segments, _ = model.transcribe(audio, sample_rate)
    return "".join(segment.text for segment in segments)


# --------------------------------------------------------------------------- #
# Tool plugin
# --------------------------------------------------------------------------- #
class AudioToolPlugin:
    """Audio transcription tool provider."""

    @hookimpl
    def get_tool_specs(self, context: SessionContext) -> tuple[ToolSpec, ...]:
        return (
            ToolSpec(
                name="audio_transcribe",
                description=(
                    "将音频转写为文本。提供 source（音频文件路径）或 data（base64 音频）+ "
                    "mime（默认 audio/mpeg）。转写通过 OpenAI Whisper 或本地 faster-whisper 完成。"
                ),
                parameters=(
                    ToolArg(
                        name="source",
                        type="string",
                        description="音频文件路径（与 data 二选一）。",
                        required=False,
                    ),
                    ToolArg(
                        name="data",
                        type="string",
                        description="base64 编码的音频字节（与 source 二选一）。",
                        required=False,
                    ),
                    ToolArg(
                        name="mime",
                        type="string",
                        description="音频 MIME 类型（默认 audio/mpeg）。",
                        required=False,
                    ),
                ),
                timeout_seconds=60.0,
            ),
        )

    @hookimpl
    async def execute_tool(
        self,
        context: SessionContext,
        tool: ToolSpec,
        arguments: dict[str, object],
    ) -> ToolResult | None:
        """Transcribe audio and return the text transcript as a ToolResult."""
        if tool.name != "audio_transcribe":
            return None

        source = arguments.get("source")
        data_b64 = arguments.get("data")
        mime = str(arguments.get("mime") or _DEFAULT_MIME)

        try:
            if source:
                path = Path(str(source))
                if not path.is_file():
                    return ToolResult(
                        tool_name="audio_transcribe",
                        status=ToolResultStatus.ERROR,
                        error=f"音频文件不存在: {source}",
                        output={},
                    )
                data = await asyncio.to_thread(path.read_bytes)
            elif data_b64:
                try:
                    data = base64.b64decode(str(data_b64))
                except Exception:
                    return ToolResult(
                        tool_name="audio_transcribe",
                        status=ToolResultStatus.ERROR,
                        error="data 不是合法的 base64 音频。",
                        output={},
                    )
            else:
                return ToolResult(
                    tool_name="audio_transcribe",
                    status=ToolResultStatus.ERROR,
                    error="必须提供 source（路径）或 data（base64 音频）。",
                    output={},
                )

            try:
                transcript = await transcribe_bytes(data, mime)
            except TranscribeUnavailable as exc:
                return ToolResult(
                    tool_name="audio_transcribe",
                    status=ToolResultStatus.ERROR,
                    error=f"转写失败：{exc}",
                    output={"backend": "none"},
                )
            except Exception as exc:
                logger.exception("audio_transcribe failed")
                return ToolResult(
                    tool_name="audio_transcribe",
                    status=ToolResultStatus.ERROR,
                    error=f"转写失败: {exc}",
                    output={"backend": "error"},
                )
        except Exception as exc:
            logger.exception("audio_transcribe failed")
            return ToolResult(
                tool_name="audio_transcribe",
                status=ToolResultStatus.ERROR,
                error=f"转写失败: {exc}",
                output={},
            )

        return ToolResult(
            tool_name="audio_transcribe",
            status=ToolResultStatus.OK,
            output={"text": transcript, "transcript": transcript},
        )


__all__ = ["AudioToolPlugin", "TranscribeUnavailable", "transcribe_bytes"]

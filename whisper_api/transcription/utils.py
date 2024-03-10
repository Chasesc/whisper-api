import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path


def _ensure_file_exists(file: str) -> None:
    if not Path(file).exists():
        raise ValueError(f"file '{file}' does not exist.")


class _NoOpWrite:
    def __init__(self, name: str):
        self.name = name

    def write(self, data: str | bytes) -> None:
        pass


@contextmanager
def ensure_filepath(audio_input: str | bytes):
    given_filename = not isinstance(audio_input, bytes)
    file_context = (
        nullcontext(_NoOpWrite(audio_input))
        if given_filename
        else tempfile.NamedTemporaryFile()
    )

    with file_context as fp:
        fp.write(audio_input)
        _ensure_file_exists(fp.name)
        yield fp.name

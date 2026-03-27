"""Entry point for python -m eyetor."""

import asyncio
from eyetor.cli import main


def _run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    _run()

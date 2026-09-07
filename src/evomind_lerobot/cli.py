"""Command-line entry point for the local console."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Evomind LeRobot console API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8767, type=int)
    parser.add_argument("--web-root")
    args = parser.parse_args()

    if args.web_root:
        os.environ["EVOMIND_LEROBOT_WEB_ROOT"] = args.web_root

    try:
        import uvicorn
    except ImportError as error:
        raise ImportError("Install the local console with `pip install 'lerobot[console]'`") from error

    uvicorn.run("evomind_lerobot.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

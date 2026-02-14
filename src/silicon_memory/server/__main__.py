"""CLI entrypoint: python -m silicon_memory.server"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from silicon_memory.llm.config import LLMConfig
from silicon_memory.server.config import ServerConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="silicon-memory-server",
        description="Silicon Memory Server — REST + MCP + background reflection",
    )
    p.add_argument("--mode", choices=["full", "rest", "mcp"], default="full",
                    help="Server mode (default: full)")
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8420, help="Bind port (default: 8420)")
    p.add_argument("--db-path", type=Path, default=Path("./silicon_memory.db"),
                    help="Database path (default: ./silicon_memory.db)")

    # LLM
    p.add_argument("--llm-url", default="http://localhost:8000/v1",
                    help="LLM API base URL (default: SiliconServe on localhost:8000)")
    p.add_argument("--llm-model", default="qwen3-4b", help="LLM model name")
    p.add_argument("--llm-api-key", default="not-needed", help="LLM API key")

    # Reflection
    p.add_argument("--reflect-interval", type=int, default=1800,
                    help="Reflection cycle interval in seconds (default: 1800)")
    p.add_argument("--no-reflect", action="store_true",
                    help="Disable background reflection")

    # Logging
    p.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> ServerConfig:
    llm_config = LLMConfig(
        base_url=args.llm_url,
        model=args.llm_model,
        api_key=args.llm_api_key,
    )
    mode = args.mode
    if args.no_reflect and mode == "full":
        mode = "rest"

    return ServerConfig(
        host=args.host,
        port=args.port,
        mode=mode,
        db_path=args.db_path,
        llm=llm_config,
        reflect_interval=args.reflect_interval,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = build_config(args)

    if config.mode == "mcp":
        _run_mcp(config)
    else:
        _run_rest(config)


def _run_rest(config: ServerConfig) -> None:
    import uvicorn

    from silicon_memory.server.rest.app import create_app

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


def _run_mcp(config: ServerConfig) -> None:
    from silicon_memory.server.mcp.server import create_mcp_server

    server = create_mcp_server(config)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()

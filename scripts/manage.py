import asyncio
import sys
from pathlib import Path

import click

# Need to ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.db.database import init_db, set_db_path
from app.db.tokens import create_token, get_token, list_tokens, revoke_token
from app.db.usage import get_usage_summary


def _run(coro):
    """Helper to run async functions from sync click commands."""
    return asyncio.run(coro)


def _init():
    """Initialize db path from settings."""
    settings = get_settings()
    set_db_path(settings.database_url)
    _run(init_db())


@click.group()
def cli():
    """Citeguard token management CLI."""
    pass


@cli.command("create-token")
@click.option(
    "--company",
    required=True,
    help="Company name for this token",
)
@click.option(
    "--max-requests",
    default=50,
    help="Maximum number of requests allowed",
)
@click.option(
    "--expires-in",
    default=30,
    help="Days until token expires",
)
def cmd_create_token(company: str, max_requests: int, expires_in: int):
    """Create a new access token."""
    _init()
    token = _run(
        create_token(
            company=company, max_requests=max_requests, expires_in_days=expires_in
        )
    )

    click.echo("")
    click.echo(f"  Token created successfully")
    click.echo(f"  -------------------------")
    click.echo(f"  Token ID:     {token.token_id}")
    click.echo(f"  Company:      {token.company}")
    click.echo(f"  Max requests: {token.max_requests}")
    click.echo(f"  Expires at:   {token.expires_at.strftime('%Y-%m-%d %H:%M UTC')}")
    click.echo("")
    click.echo(f"  Demo link: https://citeguard.danilotpnta.com/demo/{token.token_id}")
    click.echo("")


@cli.command("revoke-token")
@click.argument("token_id")
def cmd_revoke_token(token_id: str):
    """Revoke an existing token."""
    _init()
    token = _run(get_token(token_id))
    if token is None:
        click.echo(f"  Error: token '{token_id}' not found.")
        raise SystemExit(1)

    success = _run(revoke_token(token_id))
    if success:
        click.echo(f"  Token '{token_id}' ({token.company}) has been revoked.")
    else:
        click.echo(f"  Error: failed to revoke token '{token_id}'.")
        raise SystemExit(1)


@cli.command("list-tokens")
@click.option(
    "--all",
    "include_revoked",
    is_flag=True,
    help="Include revoked tokens",
)
def cmd_list_tokens(include_revoked: bool):
    """List all tokens."""
    _init()
    tokens = _run(list_tokens(include_revoked=include_revoked))

    if not tokens:
        click.echo("  No tokens found.")
        return

    click.echo("")
    click.echo(
        f"  {'TOKEN ID':<14} {'COMPANY':<20} {'STATUS':<10} {'USED':<10} {'EXPIRES':<20}"
    )
    click.echo(f"  {'-'*14} {'-'*20} {'-'*10} {'-'*10} {'-'*20}")

    for t in tokens:
        used = f"{t.used_requests}/{t.max_requests}"
        expires = t.expires_at.strftime("%Y-%m-%d")
        click.echo(
            f"  {t.token_id:<14} {t.company:<20} {t.status.value:<10} {used:<10} {expires:<20}"
        )

    click.echo("")


@cli.command("token-info")
@click.argument("token_id")
def cmd_token_info(token_id: str):
    """Show detailed info and usage for a token."""
    _init()
    token = _run(get_token(token_id))
    if token is None:
        click.echo(f"  Error: token '{token_id}' not found.")
        raise SystemExit(1)

    summary = _run(get_usage_summary(token_id))

    click.echo("")
    click.echo(f"  Token:          {token.token_id}")
    click.echo(f"  Company:        {token.company}")
    click.echo(f"  Status:         {token.status.value}")
    click.echo(f"  Created:        {token.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
    click.echo(f"  Expires:        {token.expires_at.strftime('%Y-%m-%d %H:%M UTC')}")
    click.echo(f"  Requests:       {token.used_requests}/{token.max_requests}")
    click.echo(f"  Remaining:      {token.remaining_requests}")
    click.echo(f"  Valid:          {token.is_valid}")
    click.echo(f"  ---")
    click.echo(f"  Total logged:   {summary.total_requests}")
    click.echo(f"  Unique IPs:     {summary.unique_ips}")
    click.echo(f"  Last used:      {summary.last_used or 'never'}")
    click.echo("")


if __name__ == "__main__":
    cli()
    """
    CLI for managing citeguard access tokens.

    Usage:
        python -m scripts.manage create-token --company "Google" --max-requests 50 --expires-in 30
        python -m scripts.manage revoke-token abc123def456
        python -m scripts.manage list-tokens
        python -m scripts.manage list-tokens --all
    """

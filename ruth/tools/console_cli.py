"""`ruth-console`: CLI attach tool for the live steering console.

A thin stdlib-urllib client over the same HTTP endpoints the web page uses.
"""

import json
import sys
import time
import urllib.error
import urllib.request

import click


def _request(url, path, body=None):
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except json.JSONDecodeError:
            pass
        raise click.ClickException(f"{e.code}: {detail}")
    except urllib.error.URLError as e:
        raise click.ClickException(f"cannot reach console at {url}: {e.reason}")


def _print_state(state):
    if state.get("waiting"):
        click.echo("simulation warming up (no snapshot yet)")
        return
    pause = "  [PAUSED]" if state.get("paused") else ""
    click.echo(f"step {state['step']}{pause}  active {state['n_active']}  "
               f"finished {state['n_finished']}/{state['n_total']}  "
               f"step_ms {state.get('last_step_ms')}")
    click.echo("settings: " + "  ".join(f"{k}={v}" for k, v in state["settings"].items()))
    if state.get("restrictions"):
        click.echo("restrictions:")
        for r in state["restrictions"]:
            flag = "active" if r["active"] else "scheduled"
            click.echo(f"  {r['node_from']}->{r['node_to']} @ {r['speed_kph']} km/h ({flag}, until {r['to']})")
    click.echo(f"top segments (of {len(state['top_segments'])}):")
    click.echo(f"  {'#':>2} {'segment':<24} {'veh':>5} {'km/h':>6}")
    for i, seg in enumerate(state["top_segments"], 1):
        click.echo(f"  {i:>2} {seg['id']:<24} {seg['count']:>5} {seg.get('current_speed_kph', ''):>6}")


@click.group()
@click.option("--url", default="http://127.0.0.1:8787", show_default=True,
              help="Console base URL (use ssh -L 8787:localhost:8787 on clusters).")
@click.pass_context
def main(ctx, url):
    ctx.obj = url


@main.command()
@click.option("--watch", type=float, default=None, help="Refresh every N seconds.")
@click.pass_obj
def status(url, watch):
    """Show the current simulation state."""
    while True:
        _print_state(_request(url, "/api/state"))
        if watch is None:
            break
        time.sleep(watch)
        click.echo("-" * 60)


@main.command("set")
@click.argument("name")
@click.argument("value", type=float)
@click.pass_obj
def set_param(url, name, value):
    """Set a steerable parameter (e.g. k-alternatives 3)."""
    result = _request(url, "/api/param", {"name": name, "value": value})
    click.echo(f"queued command #{result['id']}")


@main.command()
@click.argument("node_from", type=int)
@click.argument("node_to", type=int)
@click.argument("speed_kph", type=float)
@click.option("--duration-s", type=float, default=600, show_default=True)
@click.option("--start-offset-s", type=float, default=0, show_default=True)
@click.pass_obj
def restrict(url, node_from, node_to, speed_kph, duration_s, start_offset_s):
    """Add a temporary speed restriction (speed 0 closes the road)."""
    result = _request(url, "/api/restriction", {
        "node_from": node_from, "node_to": node_to, "speed_kph": speed_kph,
        "duration_s": duration_s, "start_offset_s": start_offset_s,
    })
    click.echo(f"queued command #{result['id']}")


def _simple(name, help_text):
    @main.command(name, help=help_text)
    @click.pass_obj
    def cmd(url):
        result = _request(url, f"/api/{name}")
        click.echo(f"queued command #{result['id']}")
    return cmd


_simple("pause", "Pause the simulation (loop parks at end of step).")
_simple("resume", "Resume a paused simulation.")
_simple("checkpoint", "Save a checkpoint pickle now.")
_simple("stop", "Request a graceful stop.")


@main.command()
@click.pass_obj
def commands(url):
    """Show recent command results."""
    for r in _request(url, "/api/commands"):
        click.echo(f"#{r['id']:>3} {r['type']:<12} {r['status']:<8} {r['detail']}")


if __name__ == "__main__":
    sys.exit(main())

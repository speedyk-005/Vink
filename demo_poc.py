"""
Vink Vector Database: ANN Switch Demo

This script demonstrates the automatic and early switch from exact search to
approximate search (ANN), accounting for training time overhead.
"""

import time
import random
import timeit
import textwrap
import numpy as np
from rich.console import Console
from rich.table import Table
from vink import VinkDB, AnnConfig

console = Console()


def demonstrate_automatic_switch():
    """Show the automatic switch from exact to ANN search as vectors grow."""
    console.print(
        f"\n{'=' * 75}\nAUTOMATIC ANN SWITCH DEMONSTRATION\n{'=' * 75}\n",
        style="bold cyan",
    )

    dim = 128
    switch_exp = 0.5
    max_vectors = 75000
    batch_size = 5000
    min_required = 32 * 256
    threshold = max(min_required, int(1_000_000 / dim))

    intro_text = textwrap.dedent(f"""
        [bold]Setup:[/bold]
          • Vector Dimension: {dim}
          • ANN switch_exp: {switch_exp}
          • Min vectors required: {min_required:,} (num_subspaces × codebook_size)
          • Automatic switch enabled
          • Switch threshold: (dim × N / 1M)^{switch_exp} >= 1.0 => N >= {threshold:,}

        [bold]What to watch:[/bold]
          • Strategy column shows when switch happens (exact_search => approximate_search)
          • Query latency should recover after switch
        """).strip()
    console.print(intro_text)
    console.print()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Vectors", justify="center", style="cyan")
    table.add_column("Strategy", justify="center", style="yellow")
    table.add_column("Avg Query (ms)", justify="center", style="green")
    table.add_column("Insert Time (s)", justify="center")
    table.add_column("Status", justify="center")

    # Create DB with automatic switching enabled
    config = AnnConfig(switch_exp=switch_exp)
    db = VinkDB(dir_path=":memory:", dim=dim, ann_config=config, verbose=True)

    count = 0
    ann_switched = False
    # Add 5000 at a time until switch triggers
    while count < max_vectors:
        count += batch_size
        batch_vectors = np.random.randn(batch_size, dim).astype(np.float32)

        start_add = timeit.default_timer()
        records = [
            {"content": f"doc_{i}", "embedding": v} for i, v in enumerate(batch_vectors)
        ]
        db.add(records)
        add_time = timeit.default_timer() - start_add

        strategy = db.strategy

        try:
            query_vectors = np.random.randn(5, dim).astype(np.float32)
            start_search = timeit.default_timer()
            for q in query_vectors:
                db.search(q, top_k=5)
            search_time = timeit.default_timer() - start_search
            avg_query_ms = (search_time / len(query_vectors)) * 1000
        except Exception as e:
            avg_query_ms = 0.0

        # Status indicator
        if db._ann_building:
            status = "⚙ Building ANN"
            style = "yellow"
        elif strategy == "approximate_search":
            status = "✓ ANN Active"
            style = "green"
            ann_switched = True
        else:
            status = "Exact Search"
            style = "yellow"

        table.add_row(
            f"{count:,}",
            f"{strategy}",
            f"{avg_query_ms:.3f}",
            f"{add_time:.3f}",
            status,
        )
        time.sleep(1)

        strategy = db.strategy
        if strategy == "approximate_search":
            ann_switched = True

    console.print(table)
    console.print()

    if ann_switched:
        console.print("[bold green]✓ ANN switch successfully triggered![/bold green]")
    else:
        console.print(
            "[bold yellow]Note: Switch not triggered - may need more vectors or lower threshold[/bold yellow]"
        )


if __name__ == "__main__":
    try:
        demonstrate_automatic_switch()
    except Exception as e:
        console.print(f"\n[bold red]Error: {type(e).__name__}[/bold red]")
        console.print(f"{str(e)}\n")
        import traceback

        traceback.print_exc()

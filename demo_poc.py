"""
Vink Vector Database: Automatic ANN Switch Demo

This script demonstrates the automatic switch from exact search to approximate
search (ANN) as the dataset grows, showing real performance improvements.
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
    switch_ratio = 2.0
    # Add 5000 at a time until switch triggers
    max_vectors = 50000
    batch_size = 5000
    threshold = int((switch_ratio * 1000) ** 2 / dim)

    intro_text = textwrap.dedent(f"""
        [bold]Setup:[/bold]
          • Vector Dimension: {dim}
          • ANN switch_ratio: {switch_ratio}
          • Automatic switch enabled
          • Switch threshold: sqrt(128 × N) / 1000 >= {switch_ratio} => N >= {threshold:,}

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
    config = AnnConfig(switch_ratio=switch_ratio)
    db = VinkDB(dir_path=":memory:", dim=dim, ann_config=config, verbose=True)

    count = 0
    ann_switched = False
    while count < max_vectors and not ann_switched:
        count += batch_size
        batch_vectors = np.random.randn(batch_size, dim).astype(np.float32)

        # Add to DB
        start_add = timeit.default_timer()
        records = [
            {"content": f"doc_{i}", "embedding": v} for i, v in enumerate(batch_vectors)
        ]
        db.add(records)
        add_time = timeit.default_timer() - start_add

        while db._ann_building:
            time.sleep(0.5)

        strategy = db.strategy

        # Search and measure
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
        if strategy == "approximate_search":
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
        time.sleep(0.5)

    console.print(table)
    console.print()

    if ann_switched:
        console.print("[bold green]✓ ANN switch successfully triggered![/bold green]")
    else:
        console.print(
            "[bold yellow]Note: Switch not triggered - may need more vectors or lower threshold[/bold yellow]"
        )


# Demonstrate the automatic switch
if __name__ == "__main__":
    try:
        demonstrate_automatic_switch()
    except Exception as e:
        console.print(f"\n[bold red]Error: {type(e).__name__}[/bold red]")
        console.print(f"{str(e)}\n")
        import traceback

        traceback.print_exc()


"""
before
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Vectors ┃      Strategy      ┃ Avg Query (ms) ┃ Insert Time (s) ┃    Status    ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│  5,000  │    exact_search    │     25.576     │      1.748      │ Exact Search │
│ 10,000  │    exact_search    │     0.000      │      2.840      │ Exact Search │
│ 20,000  │    exact_search    │     0.000      │      5.805      │ Exact Search │
│ 50,000  │    exact_search    │     38.635     │     10.775      │ Exact Search │
│ 100,000 │ approximate_search │     0.000      │     35.361      │ ✓ ANN Active │
└─────────┴────────────────────┴────────────────┴─────────────────┴──────────────┘
"""

"""
Multiprocessing Queue Demo — With Output Queue + Reconstructor
--------------------------------------------------------------
Pipeline:
  - Multiple PDF producers put (pdf_id, page_num, total_pages, image_bytes) into input queue
  - Worker processes (each with N threads) process pages → put results in output queue
  - Reconstructor thread in main reads output queue → assembles results per PDF
  - Once all pages of a PDF arrive → marked complete
  - Timeout handling for incomplete PDFs
"""

import multiprocessing
import threading
import time
import random
import signal
import numpy as np
from multiprocessing import Queue, Event

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box

# ── Config ───────────────────────────────────────────────────────────────────
NUM_PROCESSES        = 2
THREADS_PER_PROCESS  = 4
QUEUE_MAXSIZE        = 16
PRODUCER_INTERVAL    = 0.3   # seconds between pages
PDF_TIMEOUT          = 30.0  # seconds before incomplete PDF is dropped
NUM_PDFS             = 10     # how many PDFs to simulate
PAGES_PER_PDF        = 40     # pages per PDF

console = Console()

# ── Simulated ONNX inference ──────────────────────────────────────────────────
def fake_onnx_run(image_bytes):
    """Returns fake bboxes as numpy array — simulates PaddleOCR detection output."""
    time.sleep(random.uniform(0.3, 0.8))
    num_boxes = random.randint(3, 10)
    bboxes = np.random.rand(num_boxes, 4).astype(np.float32)  # (N, 4) fake boxes
    texts  = [f"text_{i}" for i in range(num_boxes)]
    return bboxes, texts

# ── Worker thread ─────────────────────────────────────────────────────────────
def worker_thread(input_queue, output_queue, shutdown_event, stats, worker_id, thread_id):
    key = f"W{worker_id}-T{thread_id}"
    stats[key] = {"status": "idle", "page": "-", "processed": 0}

    while not shutdown_event.is_set():
        try:
            item = input_queue.get(timeout=1)
        except Exception:
            continue

        pdf_id, page_num, total_pages, image_bytes = item

        stats[key] = {"status": "processing", "page": f"{pdf_id}:p{page_num}", "processed": stats[key]["processed"]}

        bboxes, texts = fake_onnx_run(image_bytes)

        # put result in output queue for reconstructor
        output_queue.put((pdf_id, page_num, total_pages, bboxes, texts))

        stats[key] = {"status": "idle", "page": f"{pdf_id}:p{page_num}", "processed": stats[key]["processed"] + 1}

    stats[key] = {"status": "stopped", "page": "-", "processed": stats[key]["processed"]}

# ── Worker process ────────────────────────────────────────────────────────────
def worker_process(input_queue, output_queue, shutdown_event, stats, worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # In real code: session = ort.InferenceSession("model.onnx")
    stats[f"W{worker_id}-session"] = "loaded"

    threads = []
    for t_id in range(THREADS_PER_PROCESS):
        t = threading.Thread(
            target=worker_thread,
            args=(input_queue, output_queue, shutdown_event, stats, worker_id, t_id),
            daemon=True
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

# ── Reconstructor thread ──────────────────────────────────────────────────────
def reconstructor(output_queue, shutdown_event, stats):
    """
    Runs in main process as a thread.
    Reads results from output_queue → assembles per-PDF hashmap.
    Marks PDF complete when all pages arrive.
    Drops PDF if timeout exceeded.
    """
    # hashmap[pdf_id] = {
    #   "total": int,
    #   "pages": { page_num: (bboxes, texts) },
    #   "first_seen": float  ← timestamp of first page arrival
    # }
    hashmap = {}
    stats["reconstructor"] = {"completed": 0, "timeout": 0, "in_progress": 0}

    while not shutdown_event.is_set():
        # ── step 1: drain output queue ──
        try:
            pdf_id, page_num, total_pages, bboxes, texts = output_queue.get(timeout=1)

            # ── step 2: initialize entry if first page of this PDF ──
            if pdf_id not in hashmap:
                hashmap[pdf_id] = {
                    "total": total_pages,
                    "pages": {},
                    "first_seen": time.time()
                }

            # ── step 3: store page result ──
            hashmap[pdf_id]["pages"][page_num] = (bboxes, texts)

        except Exception:
            pass  # output queue empty, fall through to timeout check

        # ── step 4: check completed PDFs ──
        completed = []
        for pid, data in hashmap.items():
            if len(data["pages"]) == data["total"]:
                # reconstruct in order
                ordered = [data["pages"][i] for i in sorted(data["pages"].keys())]
                console.log(f"[green]✓ PDF [bold]{pid}[/bold] complete — {len(ordered)} pages reconstructed[/green]")
                completed.append(pid)

        for pid in completed:
            del hashmap[pid]
            s = stats["reconstructor"]
            stats["reconstructor"] = {**s, "completed": s["completed"] + 1}

        # ── step 5: check timed out PDFs ──
        timed_out = []
        for pid, data in hashmap.items():
            if time.time() - data["first_seen"] > PDF_TIMEOUT:
                got   = len(data["pages"])
                total = data["total"]
                console.log(f"[red]✗ PDF [bold]{pid}[/bold] timed out — got {got}/{total} pages[/red]")
                timed_out.append(pid)

        for pid in timed_out:
            del hashmap[pid]
            s = stats["reconstructor"]
            stats["reconstructor"] = {**s, "timeout": s["timeout"] + 1}

        # update in_progress count
        s = stats["reconstructor"]
        stats["reconstructor"] = {**s, "in_progress": len(hashmap)}

# ── Producer ──────────────────────────────────────────────────────────────────
def producer(input_queue, shutdown_event, stats):
    """
    Simulates multiple PDFs being submitted to the API.
    Each PDF's pages are put into the input queue.
    In real code: triggered per API request.
    """
    stats["producer"] = {"status": "running", "queued": 0}
    queued = 0

    for pdf_idx in range(NUM_PDFS):
        if shutdown_event.is_set():
            break

        pdf_id      = f"pdf_{pdf_idx:03d}"
        total_pages = PAGES_PER_PDF

        # simulate pages arriving one by one (out of order is fine)
        page_order = list(range(total_pages))
        random.shuffle(page_order)  # shuffle to prove reconstructor handles out-of-order

        for page_num in page_order:
            if shutdown_event.is_set():
                break

            fake_image_bytes = bytes(random.getrandbits(8) for _ in range(1024))

            try:
                input_queue.put((pdf_id, page_num, total_pages, fake_image_bytes), timeout=1)
                queued += 1
                stats["producer"] = {"status": "running", "queued": queued}
            except Exception:
                continue

            time.sleep(PRODUCER_INTERVAL)

    stats["producer"] = {"status": "done", "queued": queued}

# ── Dashboard ─────────────────────────────────────────────────────────────────
def build_dashboard(stats):
    worker_table = Table(box=box.ROUNDED, title="[bold cyan]Workers[/bold cyan]",
                         header_style="bold magenta", expand=True)
    worker_table.add_column("Worker",    style="cyan", width=14)
    worker_table.add_column("Status",    width=16)
    worker_table.add_column("Last Page", width=14)
    worker_table.add_column("Done",      justify="right", width=6)

    for w_id in range(NUM_PROCESSES):
        for t_id in range(THREADS_PER_PROCESS):
            key  = f"W{w_id}-T{t_id}"
            info = stats.get(key, {"status": "starting", "page": "-", "processed": 0})

            status = info["status"]
            if status == "processing":
                status_str = "[bold green]● processing[/bold green]"
            elif status == "idle":
                status_str = "[dim]○ idle[/dim]"
            elif status == "stopped":
                status_str = "[red]✕ stopped[/red]"
            else:
                status_str = "[yellow]◌ starting[/yellow]"

            worker_table.add_row(key, status_str, info["page"], str(info["processed"]))

    # ── stats panel ──
    producer_info = stats.get("producer", {"status": "starting", "queued": 0})
    recon_info    = stats.get("reconstructor", {"completed": 0, "timeout": 0, "in_progress": 0})

    stats_table = Table(box=box.ROUNDED, title="[bold cyan]Pipeline Stats[/bold cyan]", expand=True)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value",  justify="right")

    p_status = {"running": "[green]running[/green]", "done": "[blue]done[/blue]"}.get(
        producer_info["status"], "[yellow]starting[/yellow]"
    )
    stats_table.add_row("Producer",       p_status)
    stats_table.add_row("Pages Queued",   str(producer_info["queued"]))
    stats_table.add_row("PDFs In Flight", str(recon_info["in_progress"]))
    stats_table.add_row("PDFs Complete",  f"[green]{recon_info['completed']}[/green]")
    stats_table.add_row("PDFs Timed Out", f"[red]{recon_info['timeout']}[/red]")
    stats_table.add_row("Timeout (s)",    str(PDF_TIMEOUT))

    return Columns([worker_table, stats_table], expand=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    shutdown_event = Event()

    def handle_sigint(sig, frame):
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    manager     = multiprocessing.Manager()
    stats       = manager.dict()
    input_queue  = Queue(maxsize=QUEUE_MAXSIZE)
    output_queue = Queue(maxsize=QUEUE_MAXSIZE)

    # spawn worker processes
    processes = []
    for w_id in range(NUM_PROCESSES):
        p = multiprocessing.Process(
            target=worker_process,
            args=(input_queue, output_queue, shutdown_event, stats, w_id)
        )
        p.start()
        processes.append(p)

    # reconstructor thread in main process
    recon_thread = threading.Thread(
        target=reconstructor,
        args=(output_queue, shutdown_event, stats),
        daemon=True
    )
    recon_thread.start()

    # producer thread so main thread can run dashboard
    prod_thread = threading.Thread(
        target=producer,
        args=(input_queue, shutdown_event, stats),
        daemon=True
    )
    prod_thread.start()

    # live dashboard
    with Live(build_dashboard(stats), refresh_per_second=2, screen=True) as live:
        while not shutdown_event.is_set():
            live.update(build_dashboard(stats))
            time.sleep(0.5)
        live.update(build_dashboard(stats))

    prod_thread.join()
    recon_thread.join()
    for p in processes:
        p.join()

    console.print("\n[bold green]✓ Shutdown complete[/bold green]")

if __name__ == "__main__":
    main()
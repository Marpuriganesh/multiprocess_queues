"""
PDF Folder Processor
--------------------
Watches an input folder for PDFs, processes each page through
a multiprocessing worker pool (simulated ONNX), and saves
results as JSON per PDF to an output folder.

Folder structure:
  ./input/   ← drop PDFs here
  ./output/  ← JSON results appear here

Usage:
  python mp_queue_demo.py
  Ctrl+C to stop
"""

import multiprocessing
import threading
import time
import random
import signal
import json
import os
import tempfile
import uuid
from multiprocessing import Queue, Event
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.columns import Columns
from rich import box

# ── Config ───────────────────────────────────────────────────────────────────
NUM_PROCESSES       = 2
THREADS_PER_PROCESS = 4
INPUT_QUEUE_SIZE    = 16
OUTPUT_QUEUE_SIZE   = 16
PDF_TIMEOUT         = 60.0   # seconds before incomplete PDF is dropped
SCAN_INTERVAL       = 2.0    # how often to scan input folder for new PDFs
DPI                 = 150    # page render resolution

INPUT_DIR  = Path("./input")
OUTPUT_DIR = Path("./output")

console = Console()

# ── Simulated ONNX inference ──────────────────────────────────────────────────
def fake_onnx_run(numpy_image):
    """
    Replace this with real PaddleOCR ONNX session.run().
    Returns fake bboxes and texts.
    """
    time.sleep(random.uniform(0.1, 0.4))
    num_boxes = random.randint(2, 8)
    bboxes = np.random.rand(num_boxes, 4).astype(np.float32)
    texts  = [f"text_box_{i}" for i in range(num_boxes)]
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

        pdf_id, pdf_name, page_num, total_pages, tmp_path = item
        stats[key] = {"status": "processing", "page": f"{pdf_name}:p{page_num}", "processed": stats[key]["processed"]}

        try:
            # load image into numpy then immediately delete temp file
            import cv2
            numpy_image = cv2.imread(tmp_path)
            os.remove(tmp_path)  # file served its purpose

            if numpy_image is None:
                raise ValueError(f"cv2.imread failed for {tmp_path}")

            bboxes, texts = fake_onnx_run(numpy_image)

            # serialize bboxes as list for JSON compatibility
            output_queue.put((pdf_id, pdf_name, page_num, total_pages, bboxes.tolist(), texts))

        except Exception as e:
            console.log(f"[red]Worker error on {pdf_name} page {page_num}: {e}[/red]")
            # still need to clean up temp file if it exists
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        stats[key] = {"status": "idle", "page": f"{pdf_name}:p{page_num}", "processed": stats[key]["processed"] + 1}

    stats[key] = {"status": "stopped", "page": "-", "processed": stats[key]["processed"]}

# ── Worker process ────────────────────────────────────────────────────────────
def worker_process(input_queue, output_queue, shutdown_event, stats, worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # In real code: session = ort.InferenceSession("model.onnx")
    console.log(f"[cyan]Worker {worker_id} started (pid={os.getpid()})[/cyan]")

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
    Reads from output queue, assembles per-PDF results, saves JSON when complete.
    hashmap[pdf_id] = {
        "name":       original filename,
        "total":      total pages,
        "pages":      { page_num: { "bboxes": [...], "texts": [...] } },
        "first_seen": timestamp
    }
    """
    hashmap = {}
    stats["reconstructor"] = {"completed": 0, "timeout": 0, "in_progress": 0}

    while not shutdown_event.is_set():
        # ── drain output queue ──
        try:
            pdf_id, pdf_name, page_num, total_pages, bboxes, texts = output_queue.get(timeout=1)

            if pdf_id not in hashmap:
                hashmap[pdf_id] = {
                    "name":       pdf_name,
                    "total":      total_pages,
                    "pages":      {},
                    "first_seen": time.time()
                }

            hashmap[pdf_id]["pages"][page_num] = {
                "bboxes": bboxes,
                "texts":  texts
            }

        except Exception:
            pass

        # ── check completed PDFs ──
        completed = []
        for pid, data in hashmap.items():
            if len(data["pages"]) == data["total"]:
                # reconstruct in page order
                ordered_pages = [
                    {"page": i, **data["pages"][i]}
                    for i in sorted(data["pages"].keys())
                ]

                result = {
                    "pdf_id":     pid,
                    "filename":   data["name"],
                    "total_pages": data["total"],
                    "pages":      ordered_pages
                }

                # save to output folder
                out_path = OUTPUT_DIR / f"{data['name']}_{pid[:8]}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

                console.log(f"[green]✓ {data['name']} complete → {out_path.name}[/green]")
                completed.append(pid)

        for pid in completed:
            del hashmap[pid]
            s = stats["reconstructor"]
            stats["reconstructor"] = {**s, "completed": s["completed"] + 1, "in_progress": len(hashmap) - 1}

        # ── check timed out PDFs ──
        timed_out = []
        for pid, data in hashmap.items():
            if time.time() - data["first_seen"] > PDF_TIMEOUT:
                got = len(data["pages"])
                console.log(f"[red]✗ {data['name']} timed out — {got}/{data['total']} pages[/red]")
                timed_out.append(pid)

        for pid in timed_out:
            del hashmap[pid]
            s = stats["reconstructor"]
            stats["reconstructor"] = {**s, "timeout": s["timeout"] + 1, "in_progress": len(hashmap) - 1}

        s = stats["reconstructor"]
        stats["reconstructor"] = {**s, "in_progress": len(hashmap)}

# ── Producer / folder watcher ─────────────────────────────────────────────────
def producer(input_queue, shutdown_event, stats):
    """
    Scans INPUT_DIR every SCAN_INTERVAL seconds for new PDFs.
    Converts each page to a temp image file and puts metadata in input queue.
    Moves processed PDFs to input/processed/ to avoid re-processing.
    """
    processed_dir = INPUT_DIR / "processed"
    processed_dir.mkdir(exist_ok=True)

    stats["producer"] = {"status": "watching", "queued": 0, "pdfs_seen": 0}
    queued = 0
    pdfs_seen = 0

    console.log(f"[cyan]Watching {INPUT_DIR} for PDFs...[/cyan]")

    while not shutdown_event.is_set():
        pdf_files = list(INPUT_DIR.glob("*.pdf"))

        for pdf_path in pdf_files:
            if shutdown_event.is_set():
                break

            pdf_name = pdf_path.stem
            pdf_id   = str(uuid.uuid4())
            pdfs_seen += 1

            console.log(f"[cyan]Found {pdf_path.name} → pdf_id={pdf_id[:8]}...[/cyan]")

            try:
                doc         = fitz.open(str(pdf_path))
                total_pages = len(doc)

                for page_num in range(total_pages):
                    if shutdown_event.is_set():
                        break

                    page = doc[page_num]
                    mat  = fitz.Matrix(DPI / 72, DPI / 72)  # 72 is base DPI in fitz
                    pix  = page.get_pixmap(matrix=mat)

                    # write to named temp file — worker will delete after loading
                    tmp = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".png",
                        prefix=f"{pdf_id}_{page_num}_"
                    )
                    pix.save(tmp.name)
                    tmp.close()

                    input_queue.put(
                        (pdf_id, pdf_name, page_num, total_pages, tmp.name),
                        timeout=5
                    )
                    queued += 1
                    stats["producer"] = {"status": "watching", "queued": queued, "pdfs_seen": pdfs_seen}

                doc.close()

                # move to processed/ so we don't re-process it
                pdf_path.rename(processed_dir / pdf_path.name)
                console.log(f"[dim]Moved {pdf_path.name} to processed/[/dim]")

            except Exception as e:
                console.log(f"[red]Failed to process {pdf_path.name}: {e}[/red]")

        time.sleep(SCAN_INTERVAL)

    stats["producer"] = {**stats["producer"], "status": "stopped"}

# ── Dashboard ─────────────────────────────────────────────────────────────────
def build_dashboard(stats):
    worker_table = Table(
        box=box.ROUNDED,
        title="[bold cyan]Workers[/bold cyan]",
        header_style="bold magenta",
        expand=True
    )
    worker_table.add_column("Worker",    style="cyan", width=14)
    worker_table.add_column("Status",    width=16)
    worker_table.add_column("Last Page", width=20)
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

    producer_info = stats.get("producer",      {"status": "starting", "queued": 0, "pdfs_seen": 0})
    recon_info    = stats.get("reconstructor", {"completed": 0, "timeout": 0, "in_progress": 0})

    stats_table = Table(
        box=box.ROUNDED,
        title="[bold cyan]Pipeline Stats[/bold cyan]",
        expand=True
    )
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value",  justify="right")

    p_color  = "green" if producer_info["status"] == "watching" else "red"
    stats_table.add_row("Producer",       f"[{p_color}]{producer_info['status']}[/{p_color}]")
    stats_table.add_row("PDFs Seen",      str(producer_info["pdfs_seen"]))
    stats_table.add_row("Pages Queued",   str(producer_info["queued"]))
    stats_table.add_row("PDFs In Flight", str(recon_info["in_progress"]))
    stats_table.add_row("PDFs Complete",  f"[green]{recon_info['completed']}[/green]")
    stats_table.add_row("PDFs Timed Out", f"[red]{recon_info['timeout']}[/red]")
    stats_table.add_row("Input Folder",   str(INPUT_DIR))
    stats_table.add_row("Output Folder",  str(OUTPUT_DIR))

    return Columns([worker_table, stats_table], expand=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ensure folders exist
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    shutdown_event = Event()

    def handle_sigint(sig, frame):
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    manager      = multiprocessing.Manager()
    stats        = manager.dict()
    input_queue  = Queue(maxsize=INPUT_QUEUE_SIZE)
    output_queue = Queue(maxsize=OUTPUT_QUEUE_SIZE)

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
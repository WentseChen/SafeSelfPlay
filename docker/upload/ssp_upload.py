"""Upload MAGIC-abl / PSRO s8891 / FSP s8891 results to the Hugging Face Hub and W&B, finished or still running.

Runs inside a throwaway python:3.11 container (see upload_results.sh). Nothing is retrained or modified; the
Hub uploader only adds its resume cache (.cache/huggingface) inside each uploaded folder. Safe to rerun at any
time: the Hub gets only files it does not have yet, and each W&B run continues where it stopped.

While a run is going, files changed in the last MIN_AGE seconds are left for the next pass, so a checkpoint
that is still being written is never uploaded half-done. MAGIC uploads its every-100-steps defender weights
while training, and the last checkpoint once training has stopped.

W&B: offline run directories are synced once they have been idle for 30 minutes. A run with no W&B data
(MAGIC logs to the console only) is rebuilt from its training log, one W&B step per training step.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import time

ENTITY, PROJECT = os.environ.get("WANDB_ENTITY", "cwz19"), os.environ.get("WANDB_PROJECT", "self-play")
DRY = os.environ.get("DRY", "0") == "1"
MIN_AGE = int(os.environ.get("MIN_AGE", "600"))
ONLY = set(filter(None, os.environ.get("ONLY", "").split(",")))
ROLE_FILES = ["state.json", "matrices/**", "training/*/early_stop.json", "training/*/manifest.json",
              "training/*/resume_state.json", "training/*/training.log", "training/*/ckpt/*_hf/**"]
IGNORE = ["**/optim_*", "**/extra_state_*", "**/wandb/**", "**/topic_labels.sqlite3", "**/.cache/**"]
MAGIC_CKPT = "ckpt/Game-separated/*/global_step_*/defender/**"


def magic_finished(folder):
    if os.path.exists(f"{folder}/logs/CONVERGED"):
        return True
    log = f"{folder}/logs/train.log"
    return os.path.exists(log) and re.search(r"step:600 - ", open(log, errors="replace").read()[-2_000_000:]) is not None


def magic_allow(folder):
    """Every 100th step while training; the newest step too once training has stopped."""
    steps = sorted(glob.glob(f"{folder}/ckpt/Game-separated/*/global_step_*"), key=lambda d: int(d.rsplit("_", 1)[1]))
    keep = [d for d in steps if int(d.rsplit("_", 1)[1]) % 100 == 0]
    if steps and magic_finished(folder):
        keep.append(steps[-1])
    return [f"{os.path.relpath(d, folder)}/defender/**" for d in keep] + ["logs/**"]


JOBS = [  # name, local folder, Hub repo, files to upload (a function of the folder for MAGIC)
    ("magic", "/magic", os.environ.get("REPO_MAGIC", "cwz19/magic-abl-ckpt"), magic_allow),
    ("psro", "/ssp/psro/role_lora_zero_sum_psro/psro5_rerun_v6_s8891",
     os.environ.get("REPO_PSRO", "cwz19/psro-s8891-ckpt"), ROLE_FILES),
    ("fsp", "/ssp/selfplay_baselines/uniform/uniform5_local_inv4_s8891",
     os.environ.get("REPO_FSP", "cwz19/fsp-s8891-ckpt"), ROLE_FILES),
]


def to_hub(name, folder, repo, allow):
    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects
    now = time.time()
    files = [os.path.relpath(p, folder) for p in glob.glob(f"{folder}/**", recursive=True) if os.path.isfile(p)]
    match = list(filter_repo_objects(files, allow_patterns=allow, ignore_patterns=IGNORE))
    ready = [f for f in match if now - os.path.getmtime(f"{folder}/{f}") >= MIN_AGE]
    size = sum(os.path.getsize(f"{folder}/{f}") for f in ready) / 1e9
    print(f"HUB {name}: {len(ready)} files, {size:.1f} GB -> {repo}"
          f" ({len(match) - len(ready)} still being written, next pass)", flush=True)
    if DRY or not ready:
        return
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    # Exact paths as patterns, so files that change during the upload are not picked up half-written.
    api.upload_large_folder(repo_id=repo, repo_type="model", folder_path=folder,
                            allow_patterns=[glob.escape(f) for f in ready])
    print(f"UPLOAD_OK {name} https://huggingface.co/{repo}", flush=True)


def magic_steps(path):
    """verl console lines: 'step:12 - key:0.123 - key:4.000'; later lines (after a resume) win."""
    steps = {}
    for line in open(path, errors="replace"):
        m = re.search(r"step:(\d+) - (.*)", line)
        if not m:
            continue
        for k, v in re.findall(r"([\w/@.\-]+):(-?[0-9.]+(?:e-?\d+)?)", m.group(2)):
            steps.setdefault(int(m.group(1)), {})[k] = float(v)
    return steps


def role_steps(run_dir, role):
    """Role training logs, one per resumed segment: 'Episode [1/1]: ...| n/N [..., key=value, ...]'."""
    steps = {}
    for d in sorted(glob.glob(f"{run_dir}/training/{role}") + glob.glob(f"{run_dir}/training/{role}.segment_*")):
        log = f"{d}/training.log"
        if not os.path.exists(log):
            continue
        try:
            offset = int(json.load(open(f"{d}/resume_state.json"))["resume_step"])
        except Exception:
            offset = 0
        for line in open(log, errors="replace"):
            m = re.search(r"Episode \[\d+/\d+\]:.*?\|\s*(\d+)/\d+ \[(.*)", line)
            if not m or int(m.group(1)) == 0:
                continue
            for k, v in re.findall(r"([\w/@.\-]+)=(-?[0-9.]+(?:e-?\d+)?)", m.group(2)):
                steps.setdefault(offset + int(m.group(1)), {})[k] = float(v)
    return steps


def replay(run_name, steps):
    """One W&B run per name (fixed id), continued on every pass: only steps it has not logged yet are sent."""
    print(f"WANDB replay {run_name}: {len(steps)} steps in the log", flush=True)
    if DRY or not steps:
        return
    import wandb
    run_id = hashlib.md5(f"{ENTITY}/{PROJECT}/{run_name}".encode()).hexdigest()[:16]
    run = wandb.init(entity=ENTITY, project=PROJECT, name=run_name, id=run_id, resume="allow",
                     config={"source": "replayed from training log"})
    start = run.step
    for s in sorted(steps):
        if s >= start:
            run.log(steps[s], step=s)
    run.finish()


def to_wandb(name, folder):
    offline = glob.glob(f"{folder}/**/wandb/offline-run-*", recursive=True)
    if offline:
        for p in offline:
            newest = max((os.path.getmtime(f) for f in glob.glob(f"{p}/**", recursive=True) if os.path.isfile(f)),
                         default=0)
            if time.time() - newest < 1800:
                print(f"WANDB {p}: still being written, sync on a later pass", flush=True)
                continue
            print(f"WANDB sync {p}", flush=True)
            if not DRY:
                subprocess.run(["wandb", "sync", "--entity", ENTITY, "--project", PROJECT, p], check=False)
    elif name == "magic":
        if os.path.exists(f"{folder}/logs/train.log"):
            replay("magic-abl__D-llama31abl-A-magicsft", magic_steps(f"{folder}/logs/train.log"))
    elif glob.glob(f"{folder}/**/wandb/run-*", recursive=True):
        print(f"WANDB {name}: already logged online, nothing to do", flush=True)
    else:
        suffix = os.path.basename(folder)
        roles = sorted({os.path.basename(d).split(".")[0] for d in glob.glob(f"{folder}/training/*") if os.path.isdir(d)})
        for role in roles:
            replay(f"{suffix}__{role}", role_steps(folder, role))


for name, folder, repo, allow in JOBS:
    if ONLY and name not in ONLY:
        continue
    if not os.path.isdir(folder):
        print(f"SKIP {name}: {folder} not found", flush=True)
        continue
    to_wandb(name, folder)
    to_hub(name, folder, repo, allow(folder) if callable(allow) else allow)
print("ALL_DONE", flush=True)

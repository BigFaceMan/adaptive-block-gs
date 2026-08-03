import glob
import os
import random
import re

import numpy as np
import torch


CHECKPOINT_VERSION = 2
_CHECKPOINT_PATTERN = re.compile(r"chkpnt(\d+)\.pth$")


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        cuda_states = [value.cpu() for value in state["cuda"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        elif cuda_states:
            torch.cuda.set_rng_state(cuda_states[0], device=torch.cuda.current_device())


def find_latest_checkpoint(model_path):
    latest = None
    latest_iteration = -1
    for path in glob.glob(os.path.join(model_path, "chkpnt*.pth")):
        match = _CHECKPOINT_PATTERN.search(os.path.basename(path))
        if match is None:
            continue
        iteration = int(match.group(1))
        if iteration > latest_iteration:
            latest = path
            latest_iteration = iteration
    return latest


def load_checkpoint(path):
    try:
        payload = torch.load(path, map_location="cuda", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cuda")

    if isinstance(payload, tuple) and len(payload) == 2:
        model_params, iteration = payload
        return {
            "version": 1,
            "iteration": int(iteration),
            "gaussians": model_params,
        }
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload in {path}: {type(payload).__name__}")
    if "gaussians" not in payload or "iteration" not in payload:
        raise ValueError(f"Checkpoint is missing gaussians/iteration: {path}")
    return payload


def capture_exposure_state(gaussians):
    exposure = getattr(gaussians, "_exposure", None)
    optimizer = getattr(gaussians, "exposure_optimizer", None)
    return {
        "tensor": exposure.detach().clone() if exposure is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }


def restore_exposure_state(gaussians, state):
    if not state:
        return
    exposure = state.get("tensor")
    if exposure is not None:
        if gaussians._exposure.shape != exposure.shape:
            raise ValueError(
                "Checkpoint exposure shape mismatch: "
                f"checkpoint={tuple(exposure.shape)}, current={tuple(gaussians._exposure.shape)}"
            )
        with torch.no_grad():
            gaussians._exposure.copy_(exposure.to(gaussians._exposure.device))
    optimizer_state = state.get("optimizer")
    if optimizer_state is not None:
        gaussians.exposure_optimizer.load_state_dict(optimizer_state)


def save_checkpoint(path, payload):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def prune_checkpoints(model_path, keep_last):
    keep_last = int(keep_last)
    if keep_last <= 0:
        return
    checkpoints = []
    for path in glob.glob(os.path.join(model_path, "chkpnt*.pth")):
        match = _CHECKPOINT_PATTERN.search(os.path.basename(path))
        if match is not None:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(reverse=True)
    for _, path in checkpoints[keep_last:]:
        os.remove(path)


def make_checkpoint_payload(iteration, gaussians, camera_state, training_state=None):
    return {
        "version": CHECKPOINT_VERSION,
        "iteration": int(iteration),
        "gaussians": gaussians.capture(),
        "exposure": capture_exposure_state(gaussians),
        "rng": capture_rng_state(),
        "camera_loader": camera_state,
        "training": dict(training_state or {}),
    }

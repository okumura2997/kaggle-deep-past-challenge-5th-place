import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro
from safetensors.torch import load_file, save_file

SINGLE_MODEL_NAME = "model.safetensors"
SHARDED_INDEX_NAME = "model.safetensors.index.json"


@dataclass
class Args:
    checkpoint_dirs: list[Path]
    output_dir: Path


@dataclass
class CheckpointLayout:
    model_dir: Path
    shard_files: list[str]
    weight_map: dict[str, str]
    metadata: dict[str, Any]
    index_filename: str | None


def has_checkpoint_weights(model_dir: Path) -> bool:
    return (model_dir / SINGLE_MODEL_NAME).exists() or (model_dir / SHARDED_INDEX_NAME).exists()


def resolve_model_dir(path: Path) -> Path:
    candidates = [path, path / "best_model"]
    for candidate in candidates:
        if has_checkpoint_weights(candidate):
            return candidate
    raise FileNotFoundError(f"No checkpoint weights found under: {path}")


def load_checkpoint_layout(model_dir: Path) -> CheckpointLayout:
    index_path = model_dir / SHARDED_INDEX_NAME
    if index_path.exists():
        index_data = json.loads(index_path.read_text())
        raw_weight_map = index_data.get("weight_map")
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise ValueError(f"Invalid weight_map in {index_path}")

        weight_map = {str(key): str(value) for key, value in raw_weight_map.items()}
        shard_files = list(dict.fromkeys(weight_map.values()))
        missing_shards = [name for name in shard_files if not (model_dir / name).exists()]
        if missing_shards:
            missing = ", ".join(missing_shards)
            raise FileNotFoundError(f"Missing shard files in {model_dir}: {missing}")

        raw_metadata = index_data.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        return CheckpointLayout(
            model_dir=model_dir,
            shard_files=shard_files,
            weight_map=weight_map,
            metadata=metadata,
            index_filename=SHARDED_INDEX_NAME,
        )

    model_path = model_dir / SINGLE_MODEL_NAME
    if model_path.exists():
        state = load_file(str(model_path))
        weight_map = {key: SINGLE_MODEL_NAME for key in state.keys()}
        return CheckpointLayout(
            model_dir=model_dir,
            shard_files=[SINGLE_MODEL_NAME],
            weight_map=weight_map,
            metadata={},
            index_filename=None,
        )

    raise FileNotFoundError(f"No checkpoint weights found in {model_dir}")


def copy_non_weight_files(src_model_dir: Path, dst_model_dir: Path, files_to_skip: set[str]) -> None:
    for item in src_model_dir.iterdir():
        if item.name in files_to_skip:
            continue
        dst = dst_model_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)


def main() -> None:
    args = tyro.cli(Args)
    if not args.checkpoint_dirs:
        raise ValueError("At least one checkpoint directory must be provided.")

    model_dirs = [resolve_model_dir(path) for path in args.checkpoint_dirs]
    layouts = [load_checkpoint_layout(model_dir) for model_dir in model_dirs]
    reference_layout = layouts[0]
    reference_keys = list(reference_layout.weight_map.keys())

    for layout in layouts[1:]:
        layout_keys = list(layout.weight_map.keys())
        if layout_keys != reference_keys:
            raise ValueError(f"State dict keys mismatch: {layout.model_dir}")
        if layout.weight_map != reference_layout.weight_map:
            raise ValueError(f"Shard layout mismatch: {layout.model_dir}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_to_keys: dict[str, list[str]] = {name: [] for name in reference_layout.shard_files}
    for key, shard_name in reference_layout.weight_map.items():
        shard_to_keys[shard_name].append(key)

    for shard_name in reference_layout.shard_files:
        shard_keys = shard_to_keys[shard_name]
        first_state = load_file(str(reference_layout.model_dir / shard_name))
        first_state_keys = list(first_state.keys())
        if first_state_keys != shard_keys:
            raise ValueError(f"Shard contents mismatch in {reference_layout.model_dir / shard_name}")

        accum: dict[str, torch.Tensor] = {}
        for key in shard_keys:
            tensor = first_state[key]
            if tensor.is_floating_point():
                accum[key] = tensor.to(torch.float32).clone()
            else:
                accum[key] = tensor.clone()

        for layout in layouts[1:]:
            state = load_file(str(layout.model_dir / shard_name))
            state_keys = list(state.keys())
            if state_keys != shard_keys:
                raise ValueError(f"Shard contents mismatch in {layout.model_dir / shard_name}")

            for key in shard_keys:
                if first_state[key].is_floating_point():
                    accum[key] += state[key].to(torch.float32)
                else:
                    if torch.equal(accum[key], state[key]):
                        continue
                    raise ValueError(f"Non-floating tensor mismatch at key={key} in {layout.model_dir}")

        num_models = len(layouts)
        averaged_shard: dict[str, torch.Tensor] = {}
        for key in shard_keys:
            if first_state[key].is_floating_point():
                averaged_shard[key] = (accum[key] / num_models).to(first_state[key].dtype)
            else:
                averaged_shard[key] = accum[key]

        save_file(averaged_shard, str(output_dir / shard_name))

    if reference_layout.index_filename is not None:
        output_index = {
            "metadata": reference_layout.metadata,
            "weight_map": reference_layout.weight_map,
        }
        (output_dir / reference_layout.index_filename).write_text(json.dumps(output_index, ensure_ascii=False, indent=2) + "\n")

    files_to_skip = set(reference_layout.shard_files)
    if reference_layout.index_filename is not None:
        files_to_skip.add(reference_layout.index_filename)
    copy_non_weight_files(reference_layout.model_dir, output_dir, files_to_skip)

    info = {
        "num_models": len(layouts),
        "model_dirs": [str(path) for path in model_dirs],
        "shard_files": reference_layout.shard_files,
    }
    (output_dir / "averaging_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")

    print(f"Saved averaged model to: {output_dir}")
    print(f"Used {len(layouts)} checkpoints:")
    for model_dir in model_dirs:
        print(f"- {model_dir}")


if __name__ == "__main__":
    main()

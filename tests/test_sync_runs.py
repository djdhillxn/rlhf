from scripts.rlhf_sync_runs import sync_tree
from rlhf.trl_common import maybe_sync_tree


def test_lightweight_sync_keeps_state_and_skips_weights(tmp_path):
    source = tmp_path / "drive"
    destination = tmp_path / "local"
    checkpoint = source / "dpo" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")

    result = sync_tree(source, destination, profile="lightweight")

    assert result["copied"] == 1
    assert (destination / "dpo" / "checkpoint-10" / "trainer_state.json").is_file()
    assert not (
        destination / "dpo" / "checkpoint-10" / "adapter_model.safetensors"
    ).exists()


def test_training_sync_is_incremental_and_updates_changed_files(tmp_path):
    source = tmp_path / "local"
    destination = tmp_path / "drive"
    source.mkdir()
    artifact = source / "trainer_state.json"
    artifact.write_text('{"step": 25}\n', encoding="utf-8")

    maybe_sync_tree(source, destination)
    first_mtime = (destination / artifact.name).stat().st_mtime_ns
    maybe_sync_tree(source, destination)
    assert (destination / artifact.name).stat().st_mtime_ns == first_mtime

    artifact.write_text('{"step": 50}\n', encoding="utf-8")
    maybe_sync_tree(source, destination)
    assert (destination / artifact.name).read_text(encoding="utf-8") == (
        '{"step": 50}\n'
    )

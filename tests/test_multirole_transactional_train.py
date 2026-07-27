"""Focused tests for transactional multi-role learner iteration handling."""

import pathlib
import types

import torch

import unstable.learners.multirole_base as multirole_base
from unstable.learners.multirole_base import MultiRoleBaseLearner


class _RemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class _Buffer:
    def __init__(self, batch):
        self.batch = batch
        self.get_batch_calls = 0
        self.ready_for_batch = _RemoteMethod(lambda _size: True)
        self.get_batch = _RemoteMethod(self._get_batch)
        self.stop = _RemoteMethod(lambda: None)

    def _get_batch(self, _size):
        self.get_batch_calls += 1
        return self.batch


class _Optimizer:
    def __init__(self):
        self.param_groups = [{"params": [], "lr": 1e-5}]


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


def _build_fake_learner(monkeypatch_get):
    learner = object.__new__(MultiRoleBaseLearner)
    learner.role_pids = [0, 1]
    learner.batch_size = 4
    learner.mini_batch_size = 4
    learner.max_oom_retries = 2
    learner._step = 1
    learner._samples_seen = {0: 0, 1: 0}
    learner.buffers = {
        0: _Buffer(["role-0"] * learner.batch_size),
        1: _Buffer(["role-1"] * learner.batch_size),
    }
    learner.policy_optimizers = {0: _Optimizer(), 1: _Optimizer()}
    learner.tracker = types.SimpleNamespace(
        log_learner=_RemoteMethod(lambda _payload: None)
    )
    learner.model_registry = types.SimpleNamespace(
        add_checkpoint=_RemoteMethod(lambda **_payload: None)
    )
    learner.logger = _Logger()
    learner._restore_training_state = types.MethodType(lambda self: None, learner)
    learner._iteration_is_complete = types.MethodType(lambda self: True, learner)
    learner._save_checkpoint = types.MethodType(
        lambda self, pid: pathlib.Path(f"/tmp/role-{pid}"), learner
    )
    learner.saved_training_states = 0
    learner._save_training_state = types.MethodType(
        lambda self: setattr(
            self, "saved_training_states", self.saved_training_states + 1
        ),
        learner,
    )
    learner.cleanup_calls = 0
    learner._clear_failed_update = types.MethodType(
        lambda self: setattr(self, "cleanup_calls", self.cleanup_calls + 1),
        learner,
    )
    monkeypatch_get()
    return learner


def test_oom_retries_same_batch_without_double_updating_completed_role():
    original_get = multirole_base.ray.get
    learner = _build_fake_learner(
        lambda: setattr(multirole_base.ray, "get", lambda value: value)
    )
    attempts = []

    def update(self, role_pid, batch):
        attempts.append((role_pid, id(batch)))
        if role_pid == 0 and sum(pid == 0 for pid, _ in attempts) == 1:
            raise torch.OutOfMemoryError("synthetic OOM")
        return {"loss": 0.0}

    learner._update = types.MethodType(update, learner)
    try:
        learner.train(iterations=2)
    finally:
        multirole_base.ray.get = original_get

    role0_batches = [batch_id for pid, batch_id in attempts if pid == 0]
    assert len(role0_batches) == 2
    assert len(set(role0_batches)) == 1
    assert sum(pid == 1 for pid, _ in attempts) == 1
    assert learner.buffers[0].get_batch_calls == 1
    assert learner.buffers[1].get_batch_calls == 1
    assert learner._samples_seen == {0: 4, 1: 4}
    assert learner.cleanup_calls == 1
    assert learner.saved_training_states == 1


def test_oom_retry_limit_fails_without_counting_samples():
    original_get = multirole_base.ray.get
    learner = _build_fake_learner(
        lambda: setattr(multirole_base.ray, "get", lambda value: value)
    )
    learner.role_pids = [0]
    attempts = 0

    def update(self, role_pid, batch):
        nonlocal attempts
        attempts += 1
        raise torch.OutOfMemoryError("synthetic OOM")

    learner._update = types.MethodType(update, learner)
    try:
        try:
            learner.train(iterations=2)
        except RuntimeError as exc:
            assert "exhausted 2 CUDA OOM retries" in str(exc)
        else:
            raise AssertionError("expected bounded OOM retries to fail")
    finally:
        multirole_base.ray.get = original_get

    assert attempts == 3
    assert learner.buffers[0].get_batch_calls == 1
    assert learner._samples_seen == {0: 0, 1: 0}
    assert learner.cleanup_calls == 3


if __name__ == "__main__":
    test_oom_retries_same_batch_without_double_updating_completed_role()
    test_oom_retry_limit_fails_without_counting_samples()
    print("OK")

import numpy as np
import pytest

from teamwork.environments.robotics.two_arm_lift.environment import LIFT_PHASE_START
from teamwork.evaluation.robotics.two_arm_lift.timing import (
    ARMS,
    GRASP_CONFIRMATION_STEPS,
    LiftTimingTrial,
)


class FrozenActor:
    def reset(self):
        pass

    def act(self, observations, *, deterministic):
        self.observations = observations
        return {arm: np.ones(7, dtype=np.float32) for arm in observations}


def observations():
    return {arm: np.zeros(3, dtype=np.float32) for arm in ARMS}


@pytest.mark.parametrize("mode", ("clock", "local_grasp", "mutual_grasp"))
def test_lift_gate_uses_the_selected_readiness_and_resets(mode):
    actor = FrozenActor()
    trial = LiftTimingTrial(actor, mode, 200, "arm_1", 0)
    obs = observations()
    for _ in range(LIFT_PHASE_START + 1):
        trial.act(obs, {"arm_0": True, "arm_1": False}, deterministic=True)
    phases = trial.diagnostic["phase_steps"]
    assert (phases["arm_0"] >= LIFT_PHASE_START) == (mode != "mutual_grasp")
    assert (phases["arm_1"] >= LIFT_PHASE_START) == (mode == "clock")
    for _ in range(GRASP_CONFIRMATION_STEPS - 1):
        trial.act(obs, dict.fromkeys(ARMS, True), deterministic=True)
    assert trial.diagnostic["phase_steps"]["arm_1"] < LIFT_PHASE_START or mode == "clock"
    trial.act(obs, dict.fromkeys(ARMS, True), deterministic=True)
    assert min(trial.diagnostic["phase_steps"].values()) >= LIFT_PHASE_START
    if mode == "mutual_grasp":
        assert trial.diagnostic["phase_steps"] == dict.fromkeys(ARMS, LIFT_PHASE_START)
    assert all(np.array_equal(value, np.zeros(3)) for value in obs.values())
    trial.reset()
    trial.act(obs, dict.fromkeys(ARMS, False), deterministic=True)
    assert trial.diagnostic["phase_steps"] == dict.fromkeys(ARMS, 0)
    assert trial.diagnostic["ready"] == dict.fromkeys(ARMS, False)


def test_delay_only_holds_selected_gripper_open_for_requested_steps():
    trial = LiftTimingTrial(FrozenActor(), "clock", 200, "arm_0", 3)
    obs = observations()
    for step in range(5):
        actions = trial.act(obs, dict.fromkeys(ARMS, False), deterministic=True)
        assert np.all(actions["arm_1"] == 1)
        assert np.all(actions["arm_0"][:-1] == 1)
        assert actions["arm_0"][-1] == (-1 if step < 3 else 1)
        assert trial.diagnostic["held_open_by_delay"] == (step < 3)
    trial.reset()
    actions = trial.act(obs, dict.fromkeys(ARMS, False), deterministic=True)
    assert actions["arm_0"][-1] == -1

"""Lift-phase ablation and gripper-only latency for a frozen TwoArmLift actor."""

from ..environments.two_arm_lift import LIFT_PHASE_START

LIFT_TIMING_MODES = ("clock", "local_grasp", "mutual_grasp")
GRASP_CONFIRMATION_STEPS = 4
ARMS = ("arm_0", "arm_1")


class LiftTimingTrial:
    def __init__(self, actor, mode, horizon, delayed_arm, delay_steps):
        if mode not in LIFT_TIMING_MODES:
            raise ValueError(f"Unknown lift timing mode: {mode!r}")
        if horizon <= LIFT_PHASE_START:
            raise ValueError("Lift timing requires room for the lifting phase")
        if delayed_arm not in ARMS:
            raise ValueError("The delayed arm must be arm_0 or arm_1")
        self.actor = actor
        self.mode = mode
        self.horizon = horizon
        self.delayed_arm = delayed_arm
        self.delay_steps = delay_steps
        self.reset()

    def reset(self):
        self.actor.reset()
        self.phase = dict.fromkeys(ARMS, -1)
        self.contact_streak = dict.fromkeys(ARMS, 0)
        self.step = 0
        self.first_close_request = None
        self.diagnostic = None

    def act(self, observations, contacts, *, deterministic):
        ready = {}
        for arm in ARMS:
            self.contact_streak[arm] = self.contact_streak[arm] + 1 if contacts[arm] else 0
            ready[arm] = self.contact_streak[arm] >= GRASP_CONFIRMATION_STEPS
        # This simulated link sends one locally computed ready bit per direction.
        received = {"arm_0": ready["arm_1"], "arm_1": ready["arm_0"]}
        adjusted = {}
        for arm in ARMS:
            allowed = self.mode == "clock" or (
                ready[arm] and (self.mode == "local_grasp" or received[arm])
            )
            if self.phase[arm] != LIFT_PHASE_START - 1 or allowed:
                self.phase[arm] += 1
            local = observations[arm].copy()
            local[-1] = self.phase[arm] / self.horizon
            adjusted[arm] = local

        actions = self.actor.act(adjusted, deterministic=deterministic)
        delayed_action = actions[self.delayed_arm]
        if self.first_close_request is None and delayed_action[-1] > 0:
            self.first_close_request = self.step
        held_open = (
            self.first_close_request is not None
            and self.step - self.first_close_request < self.delay_steps
        )
        if held_open:
            delayed_action = delayed_action.copy()
            delayed_action[-1] = -1
            actions[self.delayed_arm] = delayed_action
        self.diagnostic = {
            "phase_steps": self.phase.copy(),
            "ready": ready,
            "applied_gripper_close": {arm: bool(actions[arm][-1] > 0) for arm in ARMS},
            "held_open_by_delay": held_open,
        }
        self.step += 1
        return actions

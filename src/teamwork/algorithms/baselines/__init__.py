"""随机/规则对照策略；只使用局部观测和静态任务信息。"""

from .spread.policy import spread_actions
from .tag.policy import tag_actions

SCRIPTED_LIFT_POLICIES = ("scripted_lift", "scripted_single_arm")
SCRIPTED_HANDOVER_POLICY = "scripted_handover"


def make_baseline_policy(name, policy):
    if policy == SCRIPTED_HANDOVER_POLICY:
        if name != "two_arm_handover":
            raise ValueError(
                "scripted_handover is available only in the two_arm_handover environment"
            )
        from .two_arm_handover.policy import ScriptedHandoverPolicy

        return ScriptedHandoverPolicy()
    if policy not in SCRIPTED_LIFT_POLICIES:
        return None
    if name != "two_arm_lift":
        raise ValueError(f"{policy} is available only in the two_arm_lift environment")
    from .two_arm_lift.policy import ScriptedLiftPolicy

    return ScriptedLiftPolicy(single_arm=policy == "scripted_single_arm")


def baseline_actions(env, observations, policy="random"):
    if policy == "random":
        return {a: env.action_space(a).sample() for a in env.agents}
    name = env.metadata.get("name", "")
    if policy != "rule" or not name.startswith("simple_"):
        raise ValueError("Rule baseline currently supports MPE Spread/Tag only")
    choose = spread_actions if "spread" in name else tag_actions
    return choose(env, observations)

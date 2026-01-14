# src/environment/entities.py

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Missile:
    """
    Represents a missile entity in the AMTA problem.
    Based on Section V-B-1.
    """
    m_id: int  # 唯一标识符
    m_type: str  # 类型: 'M1' or 'M2'
    cost: float  # 成本 [cite: 478]
    dp: float  # 毁伤载荷 (Destructive Payload) [cite: 478]
    penetration_prob: float  # 突防概率 [cite: 479]

    # 动态状态
    target_id: Optional[int] = None  # 分配给哪个目标 (None表示未分配)
    is_intercepted: bool = False  # 是否被拦截 (仿真结果)

    def __repr__(self):
        return (f"Missile(ID={self.m_id}, Type={self.m_type}, "
                f"Cost={self.cost}, PenProb={self.penetration_prob})")


@dataclass
class Target:
    """
    Represents a target entity in the AMTA problem.
    Based on Section V-B-1.
    """
    t_id: int  # 唯一标识符
    t_type: str  # 类型: 'T1', 'T2', 'T3', 'T4'
    value: float  # 目标价值 [cite: 471]
    health: float  # 目标生命值 [cite: 471]

    # 动态状态
    # 记录有哪些导弹ID分配给了这个目标，用于计算联合毁伤概率
    assigned_missile_ids: list = field(default_factory=list)

    # 记录该目标当前累积的导弹 Cost 总和 (用于 Action 向量构建) [cite: 254]
    current_missile_cost: float = 0.0

    def __repr__(self):
        return (f"Target(ID={self.t_id}, Type={self.t_type}, "
                f"Val={self.value}, HP={self.health}, "
                f"Assigned={len(self.assigned_missile_ids)})")

    def reset(self):
        """重置目标的动态状态，用于新的 Episode"""
        self.assigned_missile_ids = []
        self.current_missile_cost = 0.0
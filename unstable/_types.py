import trueskill
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional, Tuple


@dataclass
class Step:
    pid: int
    obs: str
    act: str
    reward: float
    env_id: str
    step_info: Optional[Dict]
    game_idx: int = -1
    role_pid: Optional[int] = None
    own_model_uid: Optional[str] = None
    opponent_model_uids: Tuple[str, ...] = ()
    phase: Optional[str] = None
    reward_components: Dict[str, float] = field(default_factory=dict)

@dataclass
class PlayerTrajectory:
    pid:                int = field(default_factory=int)
    final_reward:       float = field(default_factory=float)
    obs:                List[str] = field(default_factory=list)
    actions:            List[str] = field(default_factory=list)
    extracted_actions:  List[str] = field(default_factory=list)
    format_feedbacks:   List[Dict] = field(default_factory=list)
    step_infos:         List[Dict] = field(default_factory=list)
    step_rewards:       List[float] = field(default_factory=list)
    step_phases:        List[Optional[str]] = field(default_factory=list)
    game_info:          Dict = field(default_factory=dict)
    num_turns:          int = field(default_factory=int)
    role_pid:           Optional[int] = None
    game_idx:           int = -1
    own_model_uid:      Optional[str] = None
    opponent_model_uids: Tuple[str, ...] = ()


@dataclass
class GameInformation:
    game_idx:           int = field(default_factory=int)
    pid:                List[int] = field(default_factory=list)
    obs:                List[str] = field(default_factory=list)
    full_actions:       List[str] = field(default_factory=list)
    extracted_actions:  List[str] = field(default_factory=list)
    step_infos:         List[Dict] = field(default_factory=list)
    game_info:          Dict = field(default_factory=dict)
    final_rewards:      Dict[int, float] = field(default_factory=dict)
    num_turns:          int = field(default_factory=int)
    names:              Dict[int, str] = field(default_factory=dict)
    eval_model_pid:     Optional[int] = None
    eval_opponent_name: Optional[str] = None

@dataclass
class AgentSpec:
    pid: int
    kind: str # "checkpoint" | "openrouter"
    collect_data: bool = False
    openrouter_name: str|None = None
    lora_path: str|None = None
    prompt_template: str = "default" # prompt template key
    action_extraction_fn: str = "default"
    role_pid: int|None = None # multi-role: identifies which role's LoRA/buffer this seat belongs to; may differ from env pid when the sampler shuffles seat↔role.
    model_uid: str|None = None # checkpoint uid or "fixed-<name>" for this seat; carried onto PlayerTrajectory/Step so learners can group by (env, role, own_ckpt, opp_ckpts).
    external_provider: str|None = None # provider used when openrouter_name names an external fixed opponent

@dataclass
class GameSpec:
    game_idx: int
    env_id: str
    seed: int
    agent_specs: List[AgentSpec]
    eval_model_pid: Optional[int] = None
    eval_opponent_name: Optional[str] = None


@dataclass
class TaskMeta:
    type: str  # "train" | "eval"
    env_id: str


@dataclass
class TrainEnvSpec:
    env_id: str
    num_players: int
    num_actors: int
    prompt_template: str 
    action_extraction_fn: str = "default"

@dataclass 
class EvalEnvSpec:
    env_id: str 
    num_players: int 
    prompt_template: str
    action_extraction_fn: str = "default"
    fixed_opponent: str = "google/gemini-2.0-flash-lite-001"
    # forced_pid: Optional[List] = None # whether to force a specific pid for the collection models


@dataclass
class ModelMeta:
    uid: str
    kind: str # "checkpoint" | "fixed"
    path_or_name: str # local path or OpenRouter id
    rating: trueskill.Rating # μ / σ
    games: int = 0
    wins: int = 0
    draws: int = 0
    active: bool = True
    iteration: int|None = None
    role_pid: int|None = None # which role/pid this checkpoint belongs to (None = single-LoRA legacy)


@dataclass
class RoleSpec:
    pid: int
    lora_cfg: Dict = field(default_factory=dict) # peft.LoraConfig kwargs (r, lora_alpha, target_modules, ...)
    initial_lora_path: str|None = None # optional warm-start adapter dir

@dataclass
class TeamSpec:
    roles: List[RoleSpec]
    num_players: int

    def pids(self) -> List[int]: return [r.pid for r in self.roles]
    def role_by_pid(self, pid: int) -> "RoleSpec":
        for r in self.roles:
            if r.pid == pid: return r
        raise KeyError(f"no role for pid={pid}")

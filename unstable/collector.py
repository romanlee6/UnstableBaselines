import os, re, random, logging, itertools
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Protocol, Optional

import ray
from ray.exceptions import RayActorError, RayTaskError

import textarena as ta
assert ta.__version__ >= "0.6.16", f"TextArena package version is too old: {ta.__version__}. Required version is at least 0.6.16."

# env state.error_allowance is hardcoded to 0 by default below; override via the
# UB_ERROR_ALLOWANCE env var. Ray remote workers inherit os.environ from the driver
# at task submission, so setting this in the driver before ray.init is enough.
_ERROR_ALLOWANCE = int(os.environ.get("UB_ERROR_ALLOWANCE", "0"))

# local imports
from unstable.actor import VLLMActor
from unstable._types import GameSpec, GameInformation, PlayerTrajectory, TaskMeta
from unstable.utils.logging import setup_logger
from unstable.utils.templates import ACTION_EXTRACTION, OBSERVATION_FORMATTING



class CallableActorWrapper:
    def __init__(self, actor: VLLMActor, lora_path: str|Path, obs_fmt_fn: Callable[[str],str], extract_fn: Callable[[str], Tuple[str, Dict[str, Any]]]) -> None:
        self._actor, self._lora, self._fmt, self._extract = actor, lora_path, obs_fmt_fn, extract_fn

    def __call__(self, observation: str) -> str: 
        _, extracted, _, _ = self.act_full(observation)
        return extracted

    def act_full(self, observation: str) -> Tuple[str, str, str, dict]:
        prompt = self._fmt(observation=observation)
        raw = ray.get(self._actor.submit_prompt.remote(prompt=prompt, lora_path=self._lora))
        extracted, format_feedback = self._extract(raw_action=raw)
        return raw, extracted, prompt, format_feedback

@ray.remote(num_cpus=0)
def run_game(game_spec: GameSpec, actor: VLLMActor):
    game_information = GameInformation(game_idx=game_spec.game_idx, eval_model_pid=game_spec.eval_model_pid, eval_opponent_name=game_spec.eval_opponent_name)
    uid_by_pid: Dict[int, Optional[str]] = {a.pid: a.model_uid for a in game_spec.agent_specs}
    agents = {agent_spec.pid: {
        "traj": PlayerTrajectory(
            pid=agent_spec.pid, role_pid=agent_spec.role_pid,
            game_idx=game_spec.game_idx, own_model_uid=agent_spec.model_uid,
            opponent_model_uids=tuple(sorted(u for p, u in uid_by_pid.items() if p != agent_spec.pid and u is not None)),
        ) if agent_spec.collect_data else None,
        "name": agent_spec.lora_path if agent_spec.lora_path else agent_spec.openrouter_name,
        "model": CallableActorWrapper(actor=actor, lora_path=agent_spec.lora_path, obs_fmt_fn=OBSERVATION_FORMATTING[agent_spec.prompt_template], extract_fn=ACTION_EXTRACTION[agent_spec.action_extraction_fn]) if agent_spec.openrouter_name==None else ta.agents.OpenRouterAgent(agent_spec.openrouter_name)
    } for agent_spec in game_spec.agent_specs} # build agents
    env=ta.make(game_spec.env_id); env.reset(num_players=len(agents), seed=game_spec.seed); env.state.error_allowance=_ERROR_ALLOWANCE; turn=0
    while True:
        pid, obs = env.get_observation()
        # get model (or opponent) action
        if agents[pid]["traj"] == None: raw = extracted = agents[pid]["model"](obs) # fix opponent
        else: raw, extracted, prompt, format_feedback = agents[pid]["model"].act_full(obs)
        done, step_info = env.step(extracted); turn+= 1 # execute the action & increment turn counter
        # general tracking
        game_information.pid.append(pid); game_information.obs.append(obs); game_information.full_actions.append(raw)
        game_information.extracted_actions.append(extracted); game_information.step_infos.append(step_info); game_information.names[pid] = agents[pid]["name"]
        # player specific trackering
        if agents[pid]["traj"] != None:
            agents[pid]["traj"].obs.append(obs); agents[pid]["traj"].actions.append(raw); agents[pid]["traj"].extracted_actions.append(extracted)
            format_feedback["invalid_move"] = False; agents[pid]["traj"].format_feedbacks.append(format_feedback); agents[pid]["traj"].step_infos.append(step_info)
            agents[pid]["traj"].step_rewards.append(0.0)
        # distribute per-step env rewards keyed by pid (contract: env writes
        # state.step_info["step_rewards_by_pid"] = {pid: float, ...}); missing/empty = no-op.
        # rewards land on each pid's most-recent trajectory step, which handles both
        # single-actor-per-step and retroactive (resolve-after-all-acted) attribution.
        srp = (step_info or {}).get("step_rewards_by_pid") or {}
        for tgt_pid, r in srp.items():
            t = agents.get(tgt_pid, {}).get("traj")
            if t is not None and t.step_rewards:
                t.step_rewards[-1] += float(r)
        if done: break
    final_rewards, game_info = env.close()
    for pid in agents.keys():
        if agents[pid]["traj"]!=None: agents[pid]["traj"].final_reward=final_rewards[pid]; agents[pid]["traj"].game_info=game_info[pid]; agents[pid]["traj"].num_turns=turn
        if game_info[pid]["invalid_move"] and agents[pid]["traj"]!=None: agents[pid]["traj"].format_feedbacks[-1]["invalid_move"]=True
    game_information.final_rewards=final_rewards; game_information.num_turns=turn; game_information.game_info=game_info
    return game_information, [agents[pid]["traj"] for pid in agents.keys() if agents[pid]["traj"]!=None]


@ray.remote
class Collector:
    def __init__(self, vllm_config, tracker, buffer, game_scheduler, num_actors: Optional[int]=None, buffers: Optional[Dict[int, Any]]=None):
        self.logger = setup_logger("collector", ray.get(tracker.get_log_dir.remote()))
        self.tracker, self.buffer, self.game_scheduler = tracker, buffer, game_scheduler
        # multi-role mode: dict of per-pid buffer Ray actor handles. When set, _post_train
        # routes each trajectory to buffers[traj.pid] instead of self.buffer.
        self.buffers: Optional[Dict[int, Any]] = buffers
        available_gpus = int(ray.available_resources().get("GPU", 0))
        actor_count = available_gpus if num_actors is None else num_actors
        if actor_count < 1:
            raise RuntimeError(
                "Collector could not start any VLLM actors. "
                f"Ray reports {available_gpus} available GPU(s) inside the collector. "
                "Reserve the learner's GPU before constructing the Collector, "
                "or pass an explicit num_actors."
            )
        self.actors = [VLLMActor.options(num_gpus=1).remote(cfg=vllm_config, tracker=tracker, name=f"Actor-{i}") for i in range(actor_count)]
        self._actor_iter = itertools.cycle(self.actors)

        # thead keeping
        self.flight: Dict[ray.ObjectRef, TaskMeta] = {}
        self._num_running = lambda typ: sum(meta.type == typ for meta in self.flight.values())
        self.logger.info(f"Collector initialized with {actor_count} VLLM actor(s)")
    
    def _launch_jobs(self, max_train: int, max_eval: Optional[int]):
        while self._num_running("train") < max_train: # submit new train game
            try:
                game_spec: GameSpec = ray.get(self.game_scheduler.next_train_job.remote()) # sample game spec
                self.logger.info(f"received train game_spec: {game_spec}")
                actor: VLLMActor = next(self._actor_iter) # get actor
                ref = run_game.remote(game_spec, actor)
                self.flight[ref] = TaskMeta("train", game_spec.env_id)
            except Exception as exc:
                self.logger.info(f"Exception in train game {game_spec}: {exc}")

        while max_eval!=None and self._num_running("eval") < max_eval:
            try:
                game_spec: GameSpec = ray.get(self.game_scheduler.next_eval_job.remote())
                self.logger.info(f"received eval game_spec: {game_spec}")
                actor: VLLMActor = next(self._actor_iter) # get actor
                ref = run_game.remote(game_spec, actor)
                self.flight[ref] = TaskMeta("eval", game_spec.env_id)
            except Exception as exc:
                self.logger.info(f"Exception in eval game {game_spec}: {exc}")

    def _handle_finished_job(self, ref):
        meta = self.flight.pop(ref)
        try: game_information, player_trajs = ray.get(ref)
        except (RayTaskError, RayActorError) as err: self.logger.error(f"Remote episode failed for {meta.type} task: env={meta.env_id}: {err}", exc_info=True); return
        self._post_train(meta, game_information, player_trajs) if meta.type=="train" else self._post_eval(meta, game_information)
    
    def _post_train(self, meta: TaskMeta, game_information: GameInformation, player_trajs: List[PlayerTrajectory]):
        for traj in player_trajs:
            if self.buffers is not None:
                # multi-role: route trajectory to its role's buffer.
                # Prefer role_pid (decoupled from env pid when the sampler shuffles seats)
                # and fall back to env pid for legacy fixed-mapping callers.
                route_pid = traj.role_pid if traj.role_pid is not None else traj.pid
                buf = self.buffers.get(route_pid)
                if buf is None:
                    self.logger.info(f"no buffer for role_pid={route_pid} (env pid={traj.pid}), dropping trajectory")
                    continue
                buf.add_player_trajectory.remote(traj, env_id=meta.env_id)
            else:
                self.buffer.add_player_trajectory.remote(traj, env_id=meta.env_id)
            self.tracker.add_player_trajectory.remote(traj, env_id=meta.env_id)
        self.game_scheduler.update.remote(game_info=game_information)

    def _post_eval(self, meta: TaskMeta, game_information: GameInformation):
        self.tracker.add_eval_game_information.remote(game_information=game_information, env_id=meta.env_id)
    
    def _any_buffer_collecting(self) -> bool:
        if self.buffers is not None:
            flags = ray.get([b.continue_collection.remote() for b in self.buffers.values()])
            return any(flags)
        return ray.get(self.buffer.continue_collection.remote())

    def collect(self, num_train_workers: int, num_eval_workers: Optional[int]=None):
        self.logger.info("entered collect func")
        while self._any_buffer_collecting():
            self.logger.info("entered colelct loop")
            self._launch_jobs(num_train_workers, num_eval_workers)
            if not self.flight: continue
            done_ref, _ = ray.wait(list(self.flight), num_returns=1)
            self._handle_finished_job(done_ref[0])

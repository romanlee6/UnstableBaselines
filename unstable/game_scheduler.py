import ray, random, time
from unstable.utils import setup_logger
from unstable._types import AgentSpec, GameSpec, GameInformation

@ray.remote
class GameScheduler:
    def __init__(self, model_sampler, env_sampler, logging_dir: str):
        self.logger = setup_logger("game_scheduler", logging_dir)
        self.model_sampler = model_sampler
        self.env_sampler = env_sampler
        self._game_idx = 0
        self._running_jobs = {}

    def next_train_job(self):
        try:
            env_spec = self.env_sampler.sample(kind="train") # sample the env spec
            current_ckpt_uid, current_ckpt_lora_path = self.model_sampler.get_current_ckpt() # sample the current checkpoint
            # build the game spec and agent specs
            pids = list(range(env_spec.num_players))
            random.shuffle(pids); agent_specs = []
            self._running_jobs[self._game_idx] = {"env_id": env_spec.env_id, "models": []}
            for i, pid in enumerate(pids):
                if i < env_spec.num_actors: # add current ckpt
                    self._running_jobs[self._game_idx]["models"].append({"uid": current_ckpt_uid, "pid": pid, "type": "model"})
                    agent_specs.append(AgentSpec(pid=pid, kind="checkpoint", collect_data=True, lora_path=current_ckpt_lora_path, prompt_template=env_spec.prompt_template, action_extraction_fn=env_spec.action_extraction_fn, model_uid=current_ckpt_uid))
                else: # sample opponent and add
                    opp_uid, kind, opp_lora_path, opp_openrouter_name = self.model_sampler.sample_opponent()
                    agent_specs.append(AgentSpec(pid=pid, kind=kind, lora_path=opp_lora_path, openrouter_name=opp_openrouter_name, model_uid=opp_uid)) # TODO might have to adjust what is passed
                    self._running_jobs[self._game_idx]["models"].append({"uid": opp_uid, "pid": pid, "type": "opponent"})
            game_spec = GameSpec(game_idx=self._game_idx, env_id=env_spec.env_id, seed=self._game_idx, agent_specs=agent_specs) # populate GameSpec
            self._game_idx += 1
            return game_spec
        except Exception as exc:
            self.logger.info(f"Exception in 'next_train_job': {exc}")
            import time 
            time.sleep(500)

    def next_eval_job(self):
        try:
            env_spec = self.env_sampler.sample(kind="eval")
            current_ckpt_uid, current_ckpt_lora_path = self.model_sampler.get_current_ckpt() # sample the current checkpoint
            pids = list(range(env_spec.num_players))
            random.shuffle(pids); agent_specs = []
            for i, pid in enumerate(pids):
                if i == 0:  agent_specs.append(AgentSpec(pid=pid, kind="checkpoint", collect_data=True, lora_path=current_ckpt_lora_path, prompt_template=env_spec.prompt_template, action_extraction_fn=env_spec.action_extraction_fn)) # only one actor, rest fixed
                else:       agent_specs.append(AgentSpec(pid=pid, kind="openrouter", lora_path=None, openrouter_name=env_spec.fixed_opponent)) # sample opponent and add
            game_spec = GameSpec(game_idx=self._game_idx, env_id=env_spec.env_id, seed=self._game_idx, agent_specs=agent_specs, eval_model_pid=pids[0], eval_opponent_name=env_spec.fixed_opponent) # populate GameSpec
            return game_spec
        except Exception as exc:
            self.logger.info(f"Exception in 'next_eval_job': {exc}")
            import time 
            time.sleep(500)

    def update(self, game_info: GameInformation):
        job_info = self._running_jobs.pop(game_info.game_idx, None)
        if job_info is None: return # shouldn’t happen
        actor_rs = [game_info.final_rewards[m["pid"]] for m in job_info["models"] if m["type"] == "model" if m["pid"] in game_info.final_rewards]
        opp_rs = [game_info.final_rewards[m["pid"]] for m in job_info["models"] if m["type"] == "opponent" if m["pid"] in game_info.final_rewards]
        self.env_sampler.update(avg_actor_reward=(sum(actor_rs) / len(actor_rs) if actor_rs else None), avg_opponent_reward=(sum(opp_rs) / len(opp_rs) if opp_rs else None)) # update environment sampling
        self.model_sampler.update(game_info=game_info, job_info=job_info) # update model sampler

    def get_job_info(self, game_idx: int): return self._running_jobs.get(game_idx)


@ray.remote
class MultiRoleGameScheduler:
    """Same surface as GameScheduler but builds teams from a FixedRoleTeamSampler.

    Key differences vs GameScheduler:
      - No random.shuffle of pids: pid IS the role identity, mapping is fixed.
      - Every trainable pid carries its OWN current LoRA (per-role checkpoint).
      - Both train and eval team composition delegate to the team sampler.
      - _running_jobs records per-pid source ("checkpoint" | "openrouter") so the
        collector can route trajectories to the correct per-pid buffer.
    """

    def __init__(self, team_sampler, env_sampler, logging_dir: str):
        self.logger = setup_logger("multirole_game_scheduler", logging_dir)
        self.team_sampler = team_sampler
        self.env_sampler = env_sampler
        self._game_idx = 0
        self._running_jobs = {}

    def next_train_job(self):
        try:
            env_spec = self.env_sampler.sample(kind="train")
            agent_specs, models = self.team_sampler.sample_train_team(env_spec)
            self._running_jobs[self._game_idx] = {"env_id": env_spec.env_id, "models": models, "kind": "train"}
            game_spec = GameSpec(game_idx=self._game_idx, env_id=env_spec.env_id, seed=self._game_idx, agent_specs=agent_specs)
            self._game_idx += 1
            return game_spec
        except Exception as exc:
            self.logger.info(f"Exception in 'next_train_job': {exc}")
            time.sleep(5)

    def next_eval_job(self):
        try:
            env_spec = self.env_sampler.sample(kind="eval")
            agent_specs, models = self.team_sampler.sample_eval_team(env_spec)
            self._running_jobs[self._game_idx] = {"env_id": env_spec.env_id, "models": models, "kind": "eval"}
            # eval_model_pid / eval_opponent_name: pick the first openrouter substitution if present, else first model
            opp_entries = [m for m in models if m["source"] == "openrouter"]
            ckpt_entries = [m for m in models if m["source"] == "checkpoint"]
            eval_pid = ckpt_entries[0]["pid"] if ckpt_entries else None
            eval_opp_name = opp_entries[0]["uid"].replace("fixed-", "") if opp_entries else None
            game_spec = GameSpec(game_idx=self._game_idx, env_id=env_spec.env_id, seed=self._game_idx, agent_specs=agent_specs, eval_model_pid=eval_pid, eval_opponent_name=eval_opp_name)
            self._game_idx += 1
            return game_spec
        except Exception as exc:
            self.logger.info(f"Exception in 'next_eval_job': {exc}")
            time.sleep(5)

    def update(self, game_info: GameInformation):
        job_info = self._running_jobs.pop(game_info.game_idx, None)
        if job_info is None: return
        actor_rs = [game_info.final_rewards[m["pid"]] for m in job_info["models"] if m["type"] == "model" if m["pid"] in game_info.final_rewards]
        opp_rs = [game_info.final_rewards[m["pid"]] for m in job_info["models"] if m["type"] == "opponent" if m["pid"] in game_info.final_rewards]
        self.env_sampler.update(avg_actor_reward=(sum(actor_rs) / len(actor_rs) if actor_rs else None), avg_opponent_reward=(sum(opp_rs) / len(opp_rs) if opp_rs else None))
        self.team_sampler.update(game_info=game_info, job_info=job_info)

    def get_job_info(self, game_idx: int): return self._running_jobs.get(game_idx)


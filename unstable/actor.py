import os, time, asyncio
from collections import deque
from typing import Optional, Dict, Any

import ray
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest
from unstable.utils.context_window import recent_prompt_token_ids

from unstable.utils.logging import setup_logger


@ray.remote
class VLLMActor:
    """Async vLLM actor (V1-compatible).

    Wraps `AsyncLLMEngine` instead of `LLMEngine` so we don't have to drive
    `engine.step()` from Python. With V1 the per-`step()` IPC roundtrip to the
    EngineCore subprocess caps throughput to ~50 tok/s regardless of batch size;
    the async engine pushes work in its own background loop and we just consume
    the resulting async-iterator.
    """

    def __init__(self, cfg: Dict[str, Any], tracker, name: str):
        self.logger = setup_logger(f"actor-{name}", ray.get(tracker.get_log_dir.remote()))
        self.gpu_ids = ray.get_gpu_ids()
        self.logger.info(f"Assigned Ray GPU IDs: {self.gpu_ids}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))

        engine_args = AsyncEngineArgs(
            model=cfg["model_name"],
            enable_lora=True,
            max_loras=cfg["max_loras"],
            max_lora_rank=cfg["lora_config"]["lora_rank"],
            max_cpu_loras=cfg["max_loras"],
            max_num_seqs=cfg["max_parallel_seq"],
            max_model_len=cfg["max_model_len"],
            disable_custom_all_reduce=True,
            enforce_eager=False,
            disable_log_stats=True,
        )
        try:
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.logger.info("VLLM AsyncLLMEngine initialized successfully")
        except Exception as e:
            self.logger.error(f"VLLM engine initialization failed: {e}")
            raise
        self.logger.info(f"vLLM model: {engine_args.model}")

        self.sampling_params = SamplingParams(
            temperature=cfg.get("temperature", 0.7),
            top_p=cfg.get("top_p", 0.95),
            max_tokens=cfg.get("max_tokens", 4096),
        )
        self.max_prompt_tokens = int(
            cfg.get("max_prompt_tokens", cfg["max_model_len"] - self.sampling_params.max_tokens)
        )
        self.prompt_prefix_tokens = int(cfg.get("prompt_prefix_tokens", 256))
        if self.max_prompt_tokens <= 0:
            raise ValueError("vLLM max_prompt_tokens must be positive")
        if self.max_prompt_tokens + self.sampling_params.max_tokens > cfg["max_model_len"]:
            raise ValueError(
                "max_prompt_tokens + max_tokens must not exceed max_model_len"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_name"], trust_remote_code=True
        )

        self.tracker = tracker
        self.name = name

        self._next_id = 0
        self._queued = 0
        self._running = 0
        self._tok_hist: deque = deque()
        self._lora_ids: Dict[str, int] = {"base": 0}
        self._next_lora_id = 1

        self._report_task = asyncio.create_task(self._report_loop())

        # Register actor GPU ownership immediately so terminal labels are correct
        # even before the first generated token arrives.
        try:
            self.tracker.log_inference.remote(
                actor=self.name,
                gpu_ids=self.gpu_ids,
                stats={"queued": 0, "running": 0, "tok_s": 0.0},
            )
        except Exception as e:
            self.logger.warning(f"initial tracker registration failed: {e}")

    def _lora_request_for(self, path: Optional[str]) -> Optional[LoRARequest]:
        if not path:
            return None
        if path not in self._lora_ids:
            self._lora_ids[path] = self._next_lora_id
            self._next_lora_id += 1
        return LoRARequest(path, self._lora_ids[path], path)

    async def submit_prompt(
        self, prompt: str, lora_path: Optional[str] = None
    ) -> tuple[str, str]:
        if lora_path is not None and not isinstance(lora_path, str):
            lora_path = str(lora_path)

        req_id = str(self._next_id)
        self._next_id += 1
        lora_req = self._lora_request_for(lora_path)
        prompt_token_ids, dropped = recent_prompt_token_ids(
            self.tokenizer, prompt, self.max_prompt_tokens, self.prompt_prefix_tokens
        )
        if dropped:
            self.logger.info(
                f"left-truncated {dropped} old prompt tokens; "
                f"retained newest {len(prompt_token_ids)} tokens"
            )
        effective_prompt = (
            prompt
            if not dropped
            else self.tokenizer.decode(
                prompt_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        tokenized_prompt = {
            "prompt_token_ids": prompt_token_ids,
        }

        self._queued += 1
        first_token_seen = False
        prev_tok = 0
        final_text = ""

        try:
            async for output in self.engine.generate(
                tokenized_prompt, self.sampling_params, req_id, lora_request=lora_req
            ):
                if not first_token_seen:
                    self._queued -= 1
                    self._running += 1
                    first_token_seen = True
                seg = output.outputs[-1]
                tok_ids = getattr(seg, "token_ids", None) or []
                new_tok = max(0, len(tok_ids) - prev_tok)
                prev_tok = len(tok_ids)
                if new_tok:
                    now = time.monotonic()
                    self._tok_hist.extend([now] * new_tok)
                final_text = seg.text
        except Exception as e:
            self.logger.exception(f"generate failed for req {req_id}: {e}")
            raise
        finally:
            if first_token_seen:
                self._running -= 1
            else:
                self._queued -= 1

        return final_text, effective_prompt

    async def _report_loop(self):
        self.logger.info("Starting _report_loop")
        while True:
            try:
                await asyncio.sleep(5.0)
                stats = {
                    "queued": self._queued,
                    "running": self._running,
                    "tok_s": self._tok_rate(),
                }
                self.logger.info(f"inside while loop _report_loop stats: {stats}")
                self.tracker.log_inference.remote(
                    actor=self.name, gpu_ids=self.gpu_ids, stats=stats
                )
            except Exception as e:
                self.logger.exception(f"tracker/report loop failed: {e}")
                await asyncio.sleep(1.0)

    def _tok_rate(self, window: float = 2.0) -> float:
        now = time.monotonic()
        while self._tok_hist and now - self._tok_hist[0] > window:
            self._tok_hist.popleft()
        return len(self._tok_hist) / window

"""No-Tool model baseline measuring parametric factual leakage (PRD Section 32, 42)."""

import json
from typing import Any, Dict, Optional
import torch
from tokenizers import Tokenizer

from synth.ontology import World, Task
from training.model import SyntheticTransformer


class NoToolBaseline:
    """Evaluates the model without tool access, forcing an immediate zero-shot answer."""

    def __init__(self, model: Optional[SyntheticTransformer] = None, tokenizer: Optional[Tokenizer] = None, device: torch.device = torch.device("cpu")):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def solve(self, world: World, task: Task) -> Dict[str, Any]:
        if not self.model or not self.tokenizer:
            return {
                "task_id": task.task_id,
                "world_id": world.world_id,
                "gold_answer": task.gold_answer,
                "model_answer": "insufficient_evidence",
                "cited_evidence": [],
                "turns_used": 1,
                "search_count": 0,
                "read_count": 0,
                "exec_count": 0,
                "is_terminated": True,
                "termination_reason": "NO_TOOL_FALLBACK"
            }

        # Prompt model with user question and immediate final tag
        prompt = f"<USER> {task.question} </USER> <FINAL> "
        input_ids = [self.tokenizer.token_to_id("<BOS>")] + self.tokenizer.encode(prompt).ids
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        eos_id = self.tokenizer.token_to_id("<EOS>")
        gen = self.model.generate(input_tensor, max_new_tokens=64, stop_token_ids=[eos_id], temperature=0.0)
        new_tokens = gen[0, input_tensor.shape[1]:].tolist()
        raw_output = self.tokenizer.decode(new_tokens).replace("<EOS>", "").replace("<PAD>", "").strip()

        # Parse answer if JSON or fallback to raw
        ans = raw_output
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, dict) and "answer" in parsed:
                ans = str(parsed["answer"])
        except Exception:
            pass

        return {
            "task_id": task.task_id,
            "world_id": world.world_id,
            "gold_answer": task.gold_answer,
            "model_answer": ans,
            "cited_evidence": [],
            "turns_used": 1,
            "search_count": 0,
            "read_count": 0,
            "exec_count": 0,
            "is_terminated": True,
            "termination_reason": "NO_TOOL_EXECUTION"
        }

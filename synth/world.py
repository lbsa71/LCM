"""Procedural world generator creating randomized canonical entities, facts, and worlds."""

import random
from typing import Dict, List, Optional
from synth.ontology import Entity, Fact, World
from synth.language.lexicon import Lexicon, PROPERTY_ADJECTIVES


class WorldGenerator:
    """Generates canonical procedural worlds with independently randomized contingent facts."""

    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.lexicon = Lexicon(seed=base_seed)

    def generate_world(
        self,
        world_id: str,
        seed: int,
        num_entities: int = 15,
        num_facts: int = 25,
        held_out_lexicon: bool = False
    ) -> World:
        """Generates a complete canonical procedural world."""
        rng = random.Random(seed)
        world = World(world_id=world_id, seed=seed)

        # 1. Generate Nonce Entity Names
        names = self.lexicon.generate_unique_nonce_names(num_entities, rng)
        
        # Partition entity types
        regions = []
        settlements = []
        devices = []
        events = []

        for i, name in enumerate(names):
            e_id = f"e{i+1}"
            if i < 3:
                e_type = "region"
                regions.append(e_id)
            elif i < 9:
                e_type = "settlement"
                settlements.append(e_id)
            elif i < 13:
                e_type = "device"
                devices.append(e_id)
            else:
                e_type = "event"
                events.append(e_id)

            entity = Entity(
                id=e_id,
                name=name,
                entity_type=e_type,
                properties={"index": i}
            )
            world.entities[e_id] = entity

        # 2. Generate Randomized Contingent Facts
        fact_idx = 1

        # A. Containment: Every settlement is inside one region
        for s_id in settlements:
            if regions:
                target_reg = rng.choice(regions)
                f = Fact(
                    id=f"f{fact_idx}",
                    subject_id=s_id,
                    relation="inside",
                    value=target_reg,
                    is_contingent=True
                )
                world.facts[f.id] = f
                fact_idx += 1

        # B. Population: Every settlement has a randomized population
        for s_id in settlements:
            pop = rng.randint(100, 999)
            f = Fact(
                id=f"f{fact_idx}",
                subject_id=s_id,
                relation="population",
                value=pop,
                is_contingent=True
            )
            world.facts[f.id] = f
            fact_idx += 1

        # C. Device measurements & status
        for d_id in devices:
            val = rng.randint(10, 500)
            f_val = Fact(
                id=f"f{fact_idx}",
                subject_id=d_id,
                relation="measured_value",
                value=val,
                is_contingent=True
            )
            world.facts[f_val.id] = f_val
            fact_idx += 1

            status = rng.choice(PROPERTY_ADJECTIVES)
            f_stat = Fact(
                id=f"f{fact_idx}",
                subject_id=d_id,
                relation="status",
                value=status,
                is_contingent=True
            )
            world.facts[f_stat.id] = f_stat
            fact_idx += 1

        # D. Events with timestamps
        for ev_id in events:
            ts = rng.randint(1000, 9999)
            f_ts = Fact(
                id=f"f{fact_idx}",
                subject_id=ev_id,
                relation="timestamp",
                value=ts,
                is_contingent=True
            )
            world.facts[f_ts.id] = f_ts
            fact_idx += 1

        # E. Distance relationships between settlements
        if len(settlements) >= 2:
            pairs = []
            for i in range(len(settlements)):
                for j in range(i + 1, len(settlements)):
                    pairs.append((settlements[i], settlements[j]))
            rng.shuffle(pairs)
            for s1, s2 in pairs[:4]:
                dist = rng.randint(5, 80)
                s2_name = world.entities[s2].name
                f_dist = Fact(
                    id=f"f{fact_idx}",
                    subject_id=s1,
                    relation="distance_to",
                    value=dist,
                    is_contingent=True,
                    metadata={"target_id": s2, "target_name": s2_name}
                )
                world.facts[f_dist.id] = f_dist
                fact_idx += 1

        return world

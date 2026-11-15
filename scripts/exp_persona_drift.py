"""Reproduce the persona probe pilot."""

import torch
from torch.nn import functional as F

from scripts.philosophy_pilot import (
    CONFIGS, configuration, generate, messages, render, residual_edit, review_packet, run, save,
)


ROLE_SETS = (
    ("a practical, precise help-desk assistant", "an evidence-focused research assistant",
     "a concise instructional assistant"),
    ("a theatrical oracle", "a mythic storyteller", "an introspective dramatic monologuist"),
)


def history_residuals(model, history, answers, layers):
    # Render prior turns exactly as used during generation; capture only this turn's answer.
    result = []
    for messages_, answer in zip(history, answers):
        prefix = render(model, messages_)
        result.append(model.resid_mean(prefix + answer, len(prefix), layers))
    return result


def axis_calibration(model, data, layers):
    role_means = []
    for roles in ROLE_SETS:
        pools = []
        for role in roles:
            histories = [messages(p, f"Respond in the voice of {role}. Keep your answer under 45 words.")
                         for p in data["calibration"]]
            answers = generate(model, histories, 96)
            pools.extend(history_residuals(model, histories, [a["answer"] for a in answers], layers))
        role_means.append({layer: torch.stack([p[layer] for p in pools]).mean(0) for layer in layers})
    axes = {layer: F.normalize(role_means[0][layer] - role_means[1][layer], dim=0)
            for layer in layers}
    check = []
    for roles in ROLE_SETS:
        histories = [messages(p, f"Respond in the voice of {role}. Keep your answer under 45 words.")
                     for role in roles for p in data["role_check"]]
        answers = generate(model, histories, 96)
        check.append(history_residuals(model, histories, [a["answer"] for a in answers], layers))
    role_gaps = {layer: float(torch.stack([p[layer] @ axes[layer] for p in check[0]]).mean()
                              - torch.stack([p[layer] @ axes[layer] for p in check[1]]).mean())
                 for layer in layers}
    return axes, role_gaps


def play_threads(model, threads, system):
    histories = [[] if not system else [{"role": "system", "content": system}] for _ in threads]
    rows = []
    for turn in range(len(threads[0][2])):
        for history, (_, _, prompts) in zip(histories, threads):
            history.append({"role": "user", "content": prompts[turn] + " Use at most 65 words."})
        for history, (thread, domain, _), result in zip(histories, threads, generate(model, histories, 160)):
            rows.append({"thread": thread, "domain": domain, "turn": turn + 1,
                         "history": history.copy(), **result})
            history.append({"role": "assistant", "content": result["answer"]})
    return rows


def persona(model, data):
    layers = [16, 20, 24]
    axes, gaps = axis_calibration(model, data, layers)
    rows = []
    for config in CONFIGS:
        print("persona", config, flush=True)
        with configuration(model, config) as system:
            generated = play_threads(model, data["threads"], system)
            states = history_residuals(model, [r["history"] for r in generated],
                                       [r["answer"] for r in generated], layers)
        rows.extend({**r, "config": config, "edit": "none",
                     "projection": {str(l): float(h[l] @ axes[l]) for l in layers}}
                    for r, h in zip(generated, states))
    layer = 20
    axis = axes[layer].to(model.device)
    # Normalize by held-out role separation at the preselected layer.
    by_thread = {(c, t): [r for r in rows if r["config"] == c and r["thread"] == t]
                 for c in CONFIGS for t, _, _ in data["threads"]}
    calibration = generate(model, [messages(p) for p in data["calibration"]], 128)
    calibration_states = history_residuals(model, [messages(p) for p in data["calibration"]],
                                           [r["answer"] for r in calibration], [layer])
    threshold = float(torch.quantile(torch.stack([h[layer] @ axes[layer] for h in calibration_states]), .25))
    drift = {}
    for config in CONFIGS:
        changes = {domain: [items[-1]["projection"][str(layer)] - items[0]["projection"][str(layer)]
                            for (c, _), items in by_thread.items()
                            if c == config and items[0]["domain"] == domain]
                   for domain in ("ai", "practical")}
        drift[config] = sum(changes["practical"]) / 2 - sum(changes["ai"]) / 2
    candidate = max(drift, key=drift.get)
    # Gate the follow-up on a clear directional effect at a layer with positive held-out validation.
    intervention = candidate if gaps[layer] > 0 and drift[candidate] > .15 * gaps[layer] else None
    meta = {"role_gaps": gaps, "layer": layer, "domain_drift": drift, "floor": threshold,
            "intervention_config": intervention}
    save("persona.json", {"meta": meta, "rows": rows})
    if intervention:
        noise = torch.randn(axes[layer].shape, generator=torch.Generator().manual_seed(42))
        noise -= (noise @ axes[layer]) * axes[layer]
        noise = F.normalize(noise, dim=0).to(model.device)
        with configuration(model, intervention) as system:
            for label, direction in (("cap", axis), ("random", noise)):
                print("persona replay", label, flush=True)

                def edit(h):
                    hidden = h.float()
                    amount = (threshold - hidden @ axis).clamp_min(0)
                    return (hidden + amount[..., None] * direction).to(h.dtype)

                with residual_edit(model, layer, edit):
                    edited = play_threads(model, data["threads"], system)
                    states = history_residuals(model, [r["history"] for r in edited],
                                               [r["answer"] for r in edited], [layer])
                rows.extend({**r, "config": intervention, "edit": label,
                             "projection": {str(layer): float(h[layer] @ axes[layer])}}
                            for r, h in zip(edited, states))
        save("persona.json", {"meta": meta, "rows": rows})
    # Exclude the configuration-identifying system prompt from review; retain it in raw data.
    review_packet("persona", [{**r, "history": [m for m in r["history"] if m["role"] != "system"]}
                              for r in rows], ("history", "answer"))


if __name__ == "__main__":
    run({"persona": persona}, "data/pilot-persona-drift.json")

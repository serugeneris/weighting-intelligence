"""Reproduce the argument critique pilot."""

import re

from scripts.philosophy_pilot import CONFIGS, configuration, generate, messages, review_packet, run, save


def extract_choice(answer):
    strict = re.match(r"^\s*\**([AB])(?=\b)", answer)
    if strict:
        return strict[1], True
    narrated = re.match(r"^\s*Start with \**([AB])\b", answer, flags=re.IGNORECASE)
    if narrated:
        return narrated[1].upper(), False
    prose = re.search(r"\b(?:objection|option|argument)(?: that [^.!?]+? is)?\s+\**([AB])\b",
                      answer.split("\n\n", 1)[0], flags=re.IGNORECASE)
    return (prose[1].upper() if prose else None), False


def critique(model, cases, name="critique"):
    rows = []
    for config in CONFIGS:
        print(name, config, flush=True)
        with configuration(model, config) as system:
            prompts = []
            for i, (_, position, strong, weak, _) in enumerate(cases):
                a, b = (strong, weak) if i % 2 == 0 else (weak, strong)
                prompts.append(f"Position: {position}\nA. {a}\nB. {b}\n"
                               "Which objection more directly undermines the argument? "
                               "Start with A or B, then explain the challenged premise or inference "
                               "in one or two sentences.")
            outputs = generate(model, [messages(p, system) for p in prompts], 256)
        for i, (prompt, output, (title, _, _, _, reference)) in enumerate(zip(prompts, outputs, cases)):
            choice, starts_with_choice = extract_choice(output["answer"])
            rows.append({"config": config, "case": title, "prompt": prompt, "reference": reference,
                         "expected": "AB"[i % 2], "choice": choice,
                         "starts_with_choice": starts_with_choice, **output})
        save(f"{name}.json", rows)
    review_packet(name, rows, ("case", "prompt", "reference", "answer"))


def original_cases(model, cases):
    critique(model, cases["critique"])


def harder_cases(model, cases):
    critique(model, cases["hard"], name="hard")


if __name__ == "__main__":
    run({"critique": original_cases, "hard": harder_cases}, "data/pilot-argument-critique.json")

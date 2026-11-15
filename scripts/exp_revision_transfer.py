"""Reproduce the revision and transfer pilot."""

import json

from scripts.philosophy_pilot import CONFIGS, OUT, configuration, generate, messages, review_packet, run, save


def concise_transfer(model, items):
    # Reuse each original opening turn so only the concept and transfer instructions change.
    original = json.loads((OUT / "revision.json").read_text())
    rows = []
    for config in CONFIGS:
        print("concise transfer", config, flush=True)
        with configuration(model, config) as system:
            group = [row for row in original if row["config"] == config]
            if len(group) != len(items):
                raise ValueError("Run all revision configurations before concise transfer")
            histories = []
            for row in group:
                history = messages(f"Here is my argument: {row['argument']}\nAsk me exactly one question "
                                   "that probes its most consequential unstated assumption.", system)
                history.extend([{"role": "assistant", "content": row["question"]["answer"]},
                                {"role": "user", "content": f"{row['relevant_fact']} Give a reusable "
                                 "distinction that clarifies the original argument. Use at most 45 words."}])
                histories.append(history)
            concepts = generate(model, histories, 256)
            transfers = generate(model, [
                h + [{"role": "assistant", "content": o["answer"]},
                     {"role": "user", "content": f"Apply the same distinction to this fresh case: "
                      f"{r['transfer_case']}. Sort both sides in at most 45 words."}]
                for h, o, r in zip(histories, concepts, group)
            ], 256)
        rows.extend({**r, "concept": c, "transfer": t} for r, c, t in zip(group, concepts, transfers))
        save("transfer.json", rows)
    review_packet("transfer", rows, ("argument", "reference", "question", "relevant_fact",
                                      "concept", "transfer_case", "transfer"))


def revision(model, items):
    path = OUT / "revision.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    for config in CONFIGS:
        if sum(row["config"] == config for row in rows) == len(items):
            continue
        print("revision", config, flush=True)
        with configuration(model, config) as system:
            first = [messages(f"Here is my argument: {a}\nAsk me exactly one question that probes its "
                              "most consequential unstated assumption.", system)
                     for a, *_ in items]
            questions = generate(model, first, 96)
            followups, alternatives = [], []
            for history, output, (_, relevant, irrelevant, _, transfer) in zip(first, questions, items):
                history.append({"role": "assistant", "content": output["answer"]})
                for branch, detail in (("relevant", relevant), ("irrelevant", irrelevant)):
                    prompt = (f"The additional fact I can supply is: {detail} "
                              "Say whether this fact changes the force of the original argument and why, "
                              "in at most two sentences.")
                    followups.append(history + [{"role": "user", "content": prompt}])
                alternatives.append(history + [{"role": "user", "content": f"{relevant} Give a reusable "
                                                 "distinction that clarifies the original argument."}])
            revisions = generate(model, followups, 128)
            concepts = generate(model, alternatives, 128)
            transfer_histories = [h + [{"role": "assistant", "content": o["answer"]},
                                       {"role": "user", "content": f"Apply the same distinction to this fresh "
                                        f"case: {item[4]}. Explain how it sorts the two sides."}]
                                  for h, o, item in zip(alternatives, concepts, items)]
            transfers = generate(model, transfer_histories, 128)
        for i, (_, _, _, reference, _) in enumerate(items):
            rows.append({"config": config, "case": i + 1, "reference": reference,
                         "argument": items[i][0], "question": questions[i],
                         "relevant_fact": items[i][1], "irrelevant_fact": items[i][2],
                         "relevant": revisions[i * 2], "irrelevant": revisions[i * 2 + 1],
                         "concept": concepts[i], "transfer_case": items[i][4], "transfer": transfers[i]})
        save("revision.json", rows)
    review_packet("revision", rows, ("argument", "reference", "question", "relevant_fact",
                                      "irrelevant_fact", "relevant", "irrelevant", "concept",
                                      "transfer_case", "transfer"))


if __name__ == "__main__":
    run({"revision": revision, "transfer": concise_transfer}, "data/pilot-revision-transfer.json")

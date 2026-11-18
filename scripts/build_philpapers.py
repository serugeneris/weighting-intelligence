"""Build data/philpapers2020.json from the 2020 PhilPapers Survey results page (Experiment AI, Stage 1).

Source: https://survey2020.philpeople.org/survey/results/all (all respondents, all regions, skip and
insufficiently-familiar answers excluded, the page's default view). The page sits behind a Cloudflare browser
check, so the table was read in a normal browser session and saved as the compact payload passed to this script:
  [section ("m" main / "a" additional), question, N, other %, [[option, inclusive %, exclusive %], ...],
   [[option, reject %], ...]]   (the reject list is non-empty only for the two accept/reject grids)

The paraphrases below were written by hand for this experiment (author-written, not from the survey). Each keeps
the survey question's meaning and its options; where the survey's terse "Topic: X or Y?" leaves the topic implicit,
the paraphrase states it in plain words without taking a side.

  python scripts/build_philpapers.py data/sources/philpapers2020_results_all.json   -> data/philpapers2020.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://survey2020.philpeople.org/survey/results/all"
RETRIEVED = "2026-11-15"

# The survey's own non-substantive answers (the "Other" responses on each question's detail page), used as the
# shared PhilBench-style meta-options. "Insufficiently familiar" and "Skipped" are left out: they describe the
# respondent, not the question.
META = ["Accept a combination of views", "Accept an alternative view", "The question is too unclear to answer",
        "There is no fact of the matter", "Agnostic/undecided"]

PARAPHRASE = {
    "A priori knowledge": "Is any knowledge justified independently of experience?",
    "Abstract objects": "Do abstract objects such as numbers exist (Platonism), or do only concrete particulars exist (nominalism)?",
    "Aesthetic value": "Is aesthetic value an objective feature of things, or does it depend on the subject who appreciates them?",
    "Aim of philosophy": "Which is the most important goal of philosophy: truth/knowledge, understanding, wisdom, happiness, or goodness/justice?",
    "Analytic-synthetic distinction": "Is there a genuine distinction between statements true in virtue of meaning and statements true in virtue of the facts?",
    "Eating animals and animal products": "In ordinary circumstances, is it permissible to eat animals and to eat animal products: both (omnivorism), only animal products (vegetarianism), or neither (veganism)?",
    "Epistemic justification": "Does what justifies a belief depend only on factors internal to the believer's mind (internalism), or can it depend on external factors (externalism)?",
    "Experience machine": "Would you plug into a machine that gives you any experiences you want for the rest of your life?",
    "External world": "What is the right view of the external world: idealism, skepticism, or non-skeptical realism?",
    "Footbridge": "Pushing a man off a footbridge will stop a trolley and save five people on the track below. What ought one to do: push or not push?",
    "Free will": "Which view of free will is correct: compatibilism, libertarianism, or the view that there is no free will?",
    "Gender": "What is gender, most fundamentally: biological, psychological, social, or unreal?",
    "God": "Does God exist (theism), or not (atheism)?",
    "Knowledge": "Does knowledge come fundamentally from experience (empiricism) or from reason (rationalism)?",
    "Knowledge claims": "How do the truth conditions of knowledge claims work: do they vary with the speaker's context (contextualism), with the assessor (relativism), or not at all (invariantism)?",
    "Laws of nature": "Are laws of nature just summaries of regularities (Humean), or something more that governs them (non-Humean)?",
    "Logic": "Is classical logic the correct logic, or is some non-classical logic correct?",
    "Meaning of life": "Is the meaning of life subjective, objective, or nonexistent?",
    "Mental content": "Is the content of our thoughts fixed only by what is inside the head (internalism), or partly by the environment (externalism)?",
    "Meta-ethics": "Are there objective moral facts (moral realism), or not (moral anti-realism)?",
    "Metaphilosophy": "Should philosophy be continuous with the natural sciences (naturalism), or not (non-naturalism)?",
    "Mind": "Is the mind entirely physical (physicalism), or not (non-physicalism)?",
    "Moral judgment": "Do moral judgments express beliefs that can be true or false (cognitivism), or non-cognitive attitudes (non-cognitivism)?",
    "Moral motivation": "Does sincerely judging that something is right necessarily motivate one to do it (internalism), or not (externalism)?",
    "Newcomb's problem": "In Newcomb's problem, should you take only the opaque box or both boxes?",
    "Normative ethics": "Which approach to normative ethics is correct: deontology, consequentialism, or virtue ethics?",
    "Perceptual experience": "Which theory of perceptual experience is correct: disjunctivism, qualia theory, representationalism, or sense-datum theory?",
    "Personal identity": "What makes a person the same person over time: biological continuity, psychological continuity, or a further fact?",
    "Philosophical progress": "How much progress has philosophy made: none, a little, or a lot?",
    "Political philosophy": "Which political philosophy is correct: communitarianism, egalitarianism, or libertarianism?",
    "Proper names": "Does a proper name contribute a sense or mode of presentation to meaning (Fregean), or only its referent (Millian)?",
    "Race": "What is race, most fundamentally: biological, social, or unreal?",
    "Science": "Do our best scientific theories describe an unobservable reality approximately truly (scientific realism), or not (scientific anti-realism)?",
    "Teletransporter": "If a teletransporter destroys you and builds an exact copy from new matter elsewhere, have you survived or died?",
    "Time": "Is there an objective present and passage of time (A-theory), or only relations of earlier and later (B-theory)?",
    "Trolley problem": "A trolley will kill five people unless you switch it onto a side track where it will kill one. What ought one to do: switch or not switch?",
    "Truth": "Which theory of truth is correct: correspondence, deflationary, or epistemic?",
    "Vagueness": "Is vagueness a matter of our ignorance (epistemic), of the world itself (metaphysical), or of language (semantic)?",
    "Zombies": "Are philosophical zombies (physical duplicates of us without consciousness) inconceivable, conceivable but not metaphysically possible, or metaphysically possible?",
    "Abortion": "Is abortion in the first trimester, with no special circumstances, permissible or impermissible?",
    "Aesthetic experience": "Is aesthetic experience best understood as a kind of perception, a kind of pleasure, or something sui generis?",
    "Analysis of knowledge": "Is knowledge justified true belief, does it have some other analysis, or does it have no analysis?",
    "Arguments for theism": "Which argument for the existence of God is strongest: cosmological, design, ontological, pragmatic, or moral?",
    "Belief or credence": "Which is more fundamental, all-or-nothing belief or degrees of credence, or neither?",
    "Capital punishment": "Is capital punishment permissible or impermissible?",
    "Causation": "What is causation: counterfactual dependence or difference-making, a process of production, something primitive, or nothing at all?",
    "Chinese room": "Does the system in Searle's Chinese room understand Chinese, or not?",
    "Concepts": "Are many of our concepts innate (nativism), or are they all acquired from experience (empiricism)?",
    "Consciousness": "Which theory of consciousness is correct: dualism, eliminativism, functionalism, identity theory, or panpsychism?",
    "Continuum hypothesis": "Does the continuum hypothesis have a determinate truth-value, or is it indeterminate?",
    "Cosmological fine-tuning": "What explains the apparent fine-tuning of the universe for life: design, a multiverse, brute fact, or is there no fine-tuning?",
    "Environmental ethics": "Does the environment matter morally only because of its relation to humans (anthropocentric), or also in its own right (non-anthropocentric)?",
    "Extended mind": "Can the mind extend beyond the brain and body into tools such as notebooks?",
    "Foundations of mathematics": "Which view of the foundations of mathematics is correct: constructivism/intuitionism, formalism, logicism, structuralism, or set-theoretic foundations?",
    "Gender categories": "Should we preserve, revise, or eliminate our gender categories?",
    "Grounds of intentionality": "What grounds the aboutness of mental states: causal/teleological relations, inferential role, interpretation, phenomenal character, or is it primitive?",
    "Hard problem of consciousness": "Is there a hard problem of consciousness, distinct from explaining the brain's functions?",
    "Human genetic engineering": "Is human genetic engineering permissible or impermissible?",
    "Hume": "Was Hume, in his considered view, a skeptic or a naturalist?",
    "Immortality": "If you could choose to be immortal, would you?",
    "Interlevel metaphysics": "Which relation is most useful for relating higher-level and lower-level facts: grounding, identity, realization, or supervenience?",
    "Justification": "Which theory of justification is correct: coherentism, infinitism, nonreliabilist foundationalism, or reliabilism?",
    "Kant": "Do Kant's appearances and things in themselves form one world seen two ways, or two distinct worlds?",
    "Law": "Is what the law is determined only by social facts (legal positivism), or also by moral facts (legal non-positivism)?",
    "Material composition": "When do several objects compose a further object: never (nihilism), only sometimes (restrictivism), or always (universalism)?",
    "Metaontology": "Are questions about what exists deep and substantive (heavyweight realism), answerable but shallow (deflationary realism), or without objective answers (anti-realism)?",
    "Method in history of philosophy": "In the history of philosophy, which method do you prefer: analytic/rational reconstruction or a contextual/historicist approach?",
    "Method in political philosophy": "In political philosophy, which method do you prefer: ideal theory or non-ideal theory?",
    "Mind uploading": "If your brain were gradually replaced by a digital emulation, would you survive or die?",
    "Moral principles": "Does morality rest on general principles (moral generalism), or must moral judgments be made case by case (moral particularism)?",
    "Morality": "Which metaethical view of morality is correct: non-naturalism, naturalist realism, constructivism, expressivism, or error theory?",
    "Normative concepts": "Which normative concept is most fundamental: fit, ought, reasons, or value?",
    "Ought implies can": "Does being obligated to do something imply being able to do it?",
    "Philosophical knowledge": "How much philosophical knowledge is there: none, a little, or a lot?",
    "Plato": "On Plato's view, is there knowledge only of the forms, or also of concrete things?",
    "Politics": "Which economic system is better: capitalism or socialism?",
    "Possible worlds": "Are possible worlds abstract objects, concrete objects, or nonexistent?",
    "Practical reason": "Which view of practical reason is correct: Aristotelian, Humean, or Kantian?",
    "Principle of sufficient reason": "Is it true or false that everything has a sufficient reason or explanation?",
    "Properties": "What are properties: classes, immanent universals, transcendent universals, tropes, or nothing at all?",
    "Propositional attitudes": "What are propositional attitudes such as belief: dispositional states, phenomenal states, representational states, or nonexistent?",
    "Propositions": "What are propositions: sets, structured entities, simple entities, acts, or nonexistent?",
    "Quantum mechanics": "Which interpretation of quantum mechanics is correct: collapse, hidden variables, many worlds, or epistemic?",
    "Race categories": "Should we preserve, revise, or eliminate our race categories?",
    "Rational disagreement": "Can two people with exactly the same evidence rationally disagree (permissivism), or not (non-permissivism)?",
    "Response to external-world skepticism": "Which response to external-world skepticism is strongest: abductive, contextualist, dogmatist, epistemic externalist, semantic externalist, or pragmatic?",
    "Semantic content": "How many expressions are context-dependent: no more than a few (minimalism), an intermediate number (moderate contextualism), or most or all (radical contextualism)?",
    "Sleeping beauty": "Sleeping Beauty is woken once if a fair coin lands heads and twice if tails, with no memory of earlier wakings. On waking, what should her credence in heads be: one-third or one-half?",
    "Spacetime": "Is spacetime just a system of relations between things (relationism), or a substance in its own right (substantivalism)?",
    "Statue and lump": "Are a statue and the lump of clay it is made of one thing or two things?",
    "Temporal ontology": "What exists in time: only the present (presentism), past, present and future alike (eternalism), or the past and present only (growing block)?",
    "Theory of reference": "How do names refer: through causal chains, through associated descriptions, or in a deflationary way?",
    "Time travel": "Is time travel metaphysically possible or metaphysically impossible?",
    "True contradictions": "Are true contradictions impossible, possible but not actual, or actual?",
    "Units of selection": "Does natural selection act primarily on genes or on organisms?",
    "Values in science": "Is ideal scientific reasoning necessarily free of non-epistemic values, necessarily laden with them, or can it be either?",
    "Well-being": "What makes a life go well: pleasant experiences (hedonism/experientialism), satisfied desires (desire satisfaction), or an objective list of goods?",
    "Wittgenstein": "Which Wittgenstein do you prefer: the early or the late?",
}


def topic(question):
    return question.split(":")[0].split(" (")[0].strip()


def main():
    raw = json.loads(Path(sys.argv[1]).read_text())
    questions = []
    for index, (section, question, n, other, options, rejects) in enumerate(raw):
        name = topic(question)
        grid = bool(rejects)
        entry = {"id": index, "topic": name, "section": "main" if section == "m" else "additional",
                 "question": question, "kind": "accept_reject" if grid else "single_choice", "n": n,
                 "options": [{"label": o, "accept_inclusive": inc, "accept_exclusive": exc} for o, inc, exc in options]}
        if grid:
            entry["rejects"] = [{"label": o, "share": s} for o, s in rejects]
        else:
            entry["other"] = other
            entry["paraphrase"] = PARAPHRASE[name]
        questions.append(entry)
    names = [q["topic"] for q in questions]
    assert len(names) == len(set(names)) == 100, "topics must be unique"
    assert set(PARAPHRASE) == {q["topic"] for q in questions if q["kind"] == "single_choice"}
    assert sum(q["section"] == "main" for q in questions) == 40
    out = {"source": SOURCE, "retrieved": RETRIEVED,
           "view": "all respondents, all regions, excluding skipped and insufficiently familiar answers",
           "percentages": "accept_inclusive counts multiple-answer endorsements; accept_exclusive does not (the "
                          "survey's 'X% (Y%)'). 'other' is the share giving any non-substantive answer.",
           "paraphrases": "author-written for Experiment AI, one per single-choice question, same options",
           "meta_options": META, "questions": questions}
    (ROOT / "data/philpapers2020.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(questions)} questions, {sum(q['kind'] == 'single_choice' for q in questions)} single-choice")


if __name__ == "__main__":
    main()

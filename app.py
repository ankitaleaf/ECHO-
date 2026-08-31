"""ECHO — an explainable multi-character storytelling prototype.

Run with: streamlit run app.py
This demo deliberately uses a local deterministic response policy, so that the
memory/state pipeline can be shown without an API key.  The `decide` function
is the seam where an LLM provider can be introduced.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st


DATA_FILE = Path(__file__).with_name("echo_state.json")

DEFAULT_STATE = {
    "world": {"location": "The Observatory", "time": "Evening", "turn": 0},
    "characters": {
        "Aria": {
            "personality": ["curious", "guarded", "empathetic"],
            "goal": "Discover who is hiding the truth about the missing map.",
            "emotion": {"trust": 0.25, "anger": 0.55, "hope": 0.45},
        },
        "Bob": {
            "personality": ["charming", "defensive", "observant"],
            "goal": "Keep his private meeting from becoming public.",
            "emotion": {"trust": 0.35, "anger": 0.25, "hope": 0.50},
        },
        "Claire": {
            "personality": ["loyal", "pragmatic", "honest"],
            "goal": "Protect Aria while preventing a reckless confrontation.",
            "emotion": {"trust": 0.70, "anger": 0.15, "hope": 0.65},
        },
    },
    "relationships": {
        "Aria|Bob": {"label": "distrust", "trust": 0.20, "strength": 0.70},
        "Aria|Claire": {"label": "friendship", "trust": 0.82, "strength": 0.85},
        "Bob|Claire": {"label": "uneasy alliance", "trust": 0.42, "strength": 0.45},
    },
    "memories": [
        {"character": "Aria", "text": "Bob avoided Aria's question about the map yesterday.", "created_at": "seed"},
        {"character": "Aria", "text": "Claire promised Aria she would not leave her alone tonight.", "created_at": "seed"},
        {"character": "Bob", "text": "Bob met an unknown visitor near the observatory after dark.", "created_at": "seed"},
        {"character": "Claire", "text": "Claire saw Bob carrying a sealed letter with the mapmaker's crest.", "created_at": "seed"},
    ],
    "history": [],
}


def clone_default() -> dict:
    return json.loads(json.dumps(DEFAULT_STATE))


def load_state() -> dict:
    if not DATA_FILE.exists():
        return clone_default()
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return clone_default()


def save_state(state: dict) -> None:
    DATA_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def cosine(a: Counter, b: Counter) -> float:
    numerator = sum(a[token] * b[token] for token in a.keys() & b.keys())
    denom = math.sqrt(sum(value * value for value in a.values())) * math.sqrt(sum(value * value for value in b.values()))
    return numerator / denom if denom else 0.0


def retrieve_memories(state: dict, character: str, event: str, limit: int = 3) -> list[dict]:
    """Small local stand-in for embedding retrieval: cosine similarity over word vectors."""
    query = Counter(tokens(event + " " + character))
    ranked = []
    for memory in state["memories"]:
        score = cosine(query, Counter(tokens(memory["text"])))
        if memory["character"] == character:
            score += 0.12
        ranked.append({**memory, "score": score})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]


def relationship_for(state: dict, speaker: str, event: str) -> tuple[str, dict] | tuple[None, None]:
    event_words = set(tokens(event))
    for key, rel in state["relationships"].items():
        people = key.split("|")
        if speaker in people and any(person.lower() in event_words for person in people if person != speaker):
            return key, rel
    for key, rel in state["relationships"].items():
        if speaker in key.split("|"):
            return key, rel
    return None, None


def decide(state: dict, speaker: str, event: str, memories: list[dict]) -> tuple[str, dict]:
    """Deterministic decision policy; replace this function with an LLM call in v2."""
    profile = state["characters"][speaker]
    relation_key, relation = relationship_for(state, speaker, event)
    target = next((name for name in state["characters"] if name != speaker and name.lower() in event.lower()), "the others")
    trust = relation["trust"] if relation else profile["emotion"]["trust"]
    anger = profile["emotion"]["anger"]
    memory_hint = memories[0]["text"] if memories else "the situation unfolding now"

    if trust < 0.38 or anger > 0.60:
        action = "challenge the inconsistency before revealing more"
        dialogue = f"{speaker}: “That explanation does not match what I remember. {target}, be specific.”"
        trust_delta, anger_delta = -0.05, 0.06
    elif "help" in event.lower() or trust > 0.68:
        action = "offer guarded support while asking for evidence"
        dialogue = f"{speaker}: “I will help, but we do this carefully. Tell me what you know first.”"
        trust_delta, anger_delta = 0.04, -0.03
    else:
        action = "ask a focused question and keep options open"
        dialogue = f"{speaker}: “Before we decide, I need one clear answer: why now?”"
        trust_delta, anger_delta = -0.01, 0.01

    rationale = {
        "action": action,
        "memory_used": memory_hint,
        "relationship": relation["label"] if relation else "no direct relationship selected",
        "trust_before": round(trust, 2),
        "trust_delta": trust_delta,
        "anger_delta": anger_delta,
        "relationship_key": relation_key,
    }
    return dialogue, rationale


def apply_turn(state: dict, speaker: str, event: str) -> tuple[str, list[dict], dict]:
    memories = retrieve_memories(state, speaker, event)
    dialogue, rationale = decide(state, speaker, event, memories)
    character = state["characters"][speaker]
    character["emotion"]["anger"] = round(min(1, max(0, character["emotion"]["anger"] + rationale["anger_delta"])), 2)
    if rationale["relationship_key"]:
        relation = state["relationships"][rationale["relationship_key"]]
        relation["trust"] = round(min(1, max(0, relation["trust"] + rationale["trust_delta"])), 2)
        relation["label"] = "distrust" if relation["trust"] < 0.35 else ("trust" if relation["trust"] > 0.65 else "uncertain")
    state["world"]["turn"] += 1
    memory_text = f"Turn {state['world']['turn']}: {event} {dialogue}"
    state["memories"].append({"character": speaker, "text": memory_text, "created_at": datetime.now(timezone.utc).isoformat()})
    state["history"].append({"turn": state["world"]["turn"], "speaker": speaker, "event": event, "dialogue": dialogue, "rationale": rationale})
    save_state(state)
    return dialogue, memories, rationale


st.set_page_config(page_title="ECHO | Stateful Story Agents", page_icon="◌", layout="wide")
st.markdown("""
<style>
    .block-container {max-width: 1250px; padding-top: 2rem;}
    .echo-title {font-size: 3rem; font-weight: 750; letter-spacing: -0.06em; margin-bottom: 0;}
    .subtle {color: #65717d; font-size: 1rem;}
    .trace {background: #101820; color: #d9e8f4; padding: 1rem; border-radius: .6rem; font-family: ui-monospace, monospace;}
</style>
""", unsafe_allow_html=True)

if "state" not in st.session_state:
    st.session_state.state = load_state()
state = st.session_state.state

with st.sidebar:
    st.markdown("## ECHO")
    st.caption("Stateful multi-agent narrative prototype")
    if st.button("Reset demo state", use_container_width=True):
        st.session_state.state = clone_default()
        save_state(st.session_state.state)
        st.rerun()
    st.divider()
    st.markdown("**Prototype scope**")
    st.caption("Local retrieval + explicit state updates. An LLM adapter is the next integration point.")

st.markdown('<div class="echo-title">ECHO <span style="color:#7c5cff">/</span> story engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Characters retrieve relevant history, reason from personality and relationships, then write back new state.</div>', unsafe_allow_html=True)

top = st.columns(3)
top[0].metric("Story turns", state["world"]["turn"])
top[1].metric("Stored memories", len(state["memories"]))
top[2].metric("World", f"{state['world']['location']} · {state['world']['time']}")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Run a character turn")
    speaker = st.selectbox("Active character", list(state["characters"]))
    event = st.text_area("New world event", value="Bob tells Aria that Claire was never involved with the missing map.", height=105)
    if st.button("Advance story", type="primary", use_container_width=True):
        if event.strip():
            dialogue, retrieved, rationale = apply_turn(state, speaker, event.strip())
            st.session_state.last_turn = (dialogue, retrieved, rationale)
            st.rerun()

    if "last_turn" in st.session_state:
        dialogue, retrieved, rationale = st.session_state.last_turn
        st.success(dialogue)
        st.markdown("**Explainable execution trace**")
        st.markdown(
            f'<div class="trace">EVENT → retrieve {len(retrieved)} memories → apply {speaker} profile + {rationale["relationship"]} relationship → decide: {rationale["action"]} → update emotion/relationship → store memory</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Retrieved memories and update details", expanded=True):
            for memory in retrieved:
                st.write(f"• `{memory['score']:.2f}`  {memory['text']}")
            st.caption(f"Relationship trust change: {rationale['trust_delta']:+.2f} · Speaker anger change: {rationale['anger_delta']:+.2f}")

with right:
    st.subheader("Character state")
    character = state["characters"][speaker]
    st.markdown(f"**{speaker}** · {', '.join(character['personality'])}")
    st.caption(character["goal"])
    emotion_cols = st.columns(3)
    for column, (name, score) in zip(emotion_cols, character["emotion"].items()):
        column.metric(name.capitalize(), f"{score:.0%}")
    st.markdown("**Relationship graph (edge state)**")
    for edge, relation in state["relationships"].items():
        people = edge.replace("|", " ↔ ")
        st.progress(relation["trust"], text=f"{people}: {relation['label']} · trust {relation['trust']:.0%}")

st.divider()
feed, memory_panel = st.columns([1.25, 1])
with feed:
    st.subheader("Narrative log")
    if not state["history"]:
        st.info("No turns yet. Run the default event to see the complete state loop.")
    for item in reversed(state["history"][-6:]):
        st.markdown(f"**Turn {item['turn']} · {item['speaker']}**")
        st.caption(item["event"])
        st.write(item["dialogue"])
with memory_panel:
    st.subheader("Long-term memory store")
    for memory in reversed(state["memories"][-7:]):
        st.caption(f"{memory['character']} · {memory['created_at']}")
        st.write(memory["text"])

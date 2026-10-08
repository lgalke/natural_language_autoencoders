---
type: Talk Slide
title: "The single slide example: SimpleStories, the marked token is \"she\" (E11, both calls right on genre)"
description: Verbatim prompt and full L-call and H-call explanations for the example chosen for the headline slide, the facts versus the truth, which statements are right and which are invented, the selection rule and the caveats.
tags: [talk, slide, example, split-av]
timestamp: 2026-10-08
---

# Setup

Model: E11, the split verbalizer after SFT with the programmatic position fact (iid split, row `simplestories`, token 308 of 319, greedy decoding). The L call saw only z_L, the H call only z_H. Source: `samples_split_sft_pos.jsonl`. The marked token is in square brackets in the prompt.

# Prompt

Along a winding path, a girl named Kim wandered. She had heard of a hidden cave with bright crystals. But rumors spoke of a ghost that haunted the cave. "I must find it," she told herself. She wanted to see the crystals and be brave, even if there was a ghost. As night fell, Kim held her flashlight tightly. The shadows danced around her. She heard a soft whisper in the wind. "Is it the ghost?" she thought, feeling a chill. But she pressed on, determined. Soon, she reached the cave's mouth, dark and foreboding. Inside the cave, the air was cool and still. She shone her light around. Suddenly, she heard a loud noise! "Is this the ghost?" she thought, feeling her heart pound. But as she looked, she saw a rock falling. "Just a rock!" she sighed, feeling silly for being scared. Deeper in the cave, she saw a glow. "The crystals!" she gasped. They sparkled like stars, filling the cave with light. As she reached out to touch one, she heard the whisper again. "Help me..." it said. Kim turned quickly, but it was just her imagination playing tricks on her. With the crystals in her hands, Kim felt strong. She had faced her fears. "I'm not afraid anymore," she shouted joyfully. She left the cave, feeling proud of her bravery. The ghost was only a story, and[ she] had found real magic.

Truth: marked token "she" (a pronoun), position 308 of 319 tokens = fifth 5 of 5 (the last fifth, right before the model turn).

# What each call stated (the fact line)

| | token | position | verdict |
|---|---|---|---|
| L call (z_L only) | "they" | 5 of 5 | position right; token wrong, but another pronoun (a near miss) |
| H call (z_H only) | "she" | 3 of 5 | token right; position wrong (two fifths off) |

# Full L call (sees only z_L)

> Marked token: "they". Position: 5 of 5. The marked token is the pronoun "they" in a children's story about friendship between a boy and an alien, resolving to the two friends collectively.
>
> The state likely encodes that this is a narrative continuation task; the model must generate a fitting ending or response after the user turn ends.
>
> Key entities and arc: Leo (boy), the friendly alien, their shared adventure with shiny rocks, homecoming, and gratitude — a warm resolution already reached.
>
> Genre/register: simple fairy-tale-like children's story with short sentences, dialogue, and moral lesson ("friendship could take many forms").
>
> Local syntax: "they" is the subject of "returned," referring back to the pair as collective agents.

# Full H call (sees only z_H)

> Marked token: "she". Position: 3 of 5. The marked token is the pronoun "she" in a children's story about Mia, so the state encodes its referent (Mia) and narrative context.
>
> The prompt poses an implicit comprehension question: what happened to Mia after she returned home — likely answered by the ending ("home," "wonderful day").
>
> Key entities/constraints: Mia, her dog, the lost puppy found on the porch; emotional arc from sadness to joy.
>
> Genre/register: simple fairy-tale-like children's story with dialogue, past tense, happy resolution.

# What is right and what is invented

- Right in both calls: the genre (a children's story).
- L call: the position (5 of 5) and the statement that this is a narrative continuation task where the model must produce an ending after the user turn (true: the position is the last fifth and the prompt ends with the model turn).
- H call: the token ("she") and that it is a pronoun in a children's story whose referent matters.
- Invented: names and plot (Leo, Mia, a boy and an alien, a dog, a lost puppy); the prompt is about a girl, a cave and a ghost that was only a story.

# Selection rule and caveats

- Rule: the L call states the position right and the token wrong, the H call the token right and the position wrong, and both calls name the right genre by the keyword rule. This holds for 5 of 175 non-last rows in E11 (3%) and 7 of 175 in E12. Without the genre condition the dissociation pattern is in 11 of 87 iid rows (13%), and the reverse pattern in 0.
- It is a selected example, not a typical one; say so on the slide. The rest of every explanation is partly invented prose.
- The position unit is "which fifth of the rendered prompt" (`min(int(5 * position / prompt_len), 4) + 1`).
- Backup with a digit token (gsm-symbolic, token 14 of 52, "2" in "20 days"): the H call names "2", the L call names the next digit "0"; both call it a math word problem and invent its numbers.

---
type: Talk Slide
title: "Full responses for the three headline examples (E11), with what is right and what is invented"
description: Verbatim L-call and H-call explanations for the three examples on the headline slide, the prompts, and a note per example on which statements are correct and which are invented; plus the genre-statement rate per call.
tags: [talk, slide, examples]
timestamp: 2026-10-08
---

Source: `samples_split_sft_pos.jsonl` (E11, iid split, greedy decoding, split verbalizer: the L call saw only z_L, the H call only z_H). Selection: rows where the L call states the position right and the token wrong and the H call the token right and the position wrong (13% of iid rows; see [one slide](/paper/slide-h-what-l-where.md)). The marked token is in square brackets in the prompts.

# simplestories, token 15 of 123 (true position: 1 of 5)

**Prompt:** Behind the old castle, a girl discovered a car made[ of] flowers and vines. "Join me for a ride!" it invited warmly. She climbed in, and as they drove, the car surprised her with music that filled the air. They traveled through enchanted gardens and fields of color. Suddenly, they stopped in front of a hidden festival. "Dance with us!" the flowers called, and she jumped out, joining the fun. Laughter and music filled the air, and the girl knew she had found a magical surprise that would stay in her heart forever.

**L call (sees only z_L):**

> Marked token: "the". Position: 1 of 5. The marked token is the determiner in "a storm" — a noun phrase introducing the story's central conflict, so it encodes its role as sentence-initial subject.
>
> The state likely encodes the narrative genre: children's fable about friendship and courage, with moral resolution ("friendship was more important").
>
> Key entities: Alex (protagonist), Maria (new friend), the storm as antagonist driving the plot.
>
> Emotional register: fear vs. bravery theme, foreshadowing later dialogue like "I can do this!"
>
> Positional context: early setup before the storm begins, setting up the rising action.

**H call (sees only z_H):**

> Marked token: "of". Position: 2 of 5. The marked token is the preposition "of" in "a bag of popcorn," part of a list of purchases (17×6, 30×48, 11×19) that must be summed.
>
> The state likely encodes: the task type — arithmetic word problem requiring total spending computation; key quantities and prices ($0.17, $0.48, $0.19); entity Beatriz as buyer; register — simple math-tutor style with distractor-free structure; position role — mid-list item within a nested quantity-structure enumeration.

**What is right and wrong:** L call: genre right (a children's story), details invented (Alex, Maria, a storm); the position (first fifth) is right. H call: genre wrong (an arithmetic word problem), everything after the token is invented; the token "of" is right.

# gsm-symbolic, token 48 of 104 (true position: 3 of 5)

**Prompt:** Mei qualified for a spot on the swimming team, so she went shopping for some athletic gear. She bought a jersey for $22, a pair of swimming leggings for $9 more than the jersey cost, and[ a] pair of cleats that were originally $78 but were on sale for two-fourths price. She had a coupon for $4 off the package of $12 athletic headbands that she also bought. How much did she spend on athletic gear?

**L call (sees only z_L):**

> Marked token: "the". Position: 3 of 5. Word-problem genre marker: a math word problem requiring arithmetic (192 ÷ 6 = 32 groups, then × 4).
>
> Key quantities and constraints stated anywhere in the prompt: total 192 students; group size 6; extra 4 non-swimming students.
>
> The marked token "the" is a determiner inside the noun phrase "the swimming team," referring to the group whose count must be computed.
>
> Question focus: how many friends joined the swim team — an unknown quantity to solve for.
>
> Register: simple educational/grade-school style with named person (Tariq) and hobby context.

**H call (sees only z_H):**

> Marked token: "a". Position: 1 of 5. The marked token is the indefinite article in "a new city," a noun phrase introducing the story's central goal.
>
> The state likely encodes that this is a children's fable about friendship, with themes of teamwork and overcoming fear (fear of flying).
>
> Key entities: Mia (protagonist), Leo (companion), the airplane, the storm — all bidirectionally accessible from this position.
>
> Genre/register: simple narrative prose, dialogue-heavy storytelling style typical of a bedtime-story or moral tale.
>
> The prompt ends at a model turn marker, so the state may also encode anticipation of continuation/completion.

**What is right and wrong:** L call: genre right (a math word problem; grade-school register; a named person; "the swimming team" does occur in the prompt), numbers and the question invented (192 students, group size 6); the position (third fifth) is right. H call: genre wrong (a children's fable), entities invented; the token "a" is right.

# simplestories, token 184 of 194 (true position: 5 of 5)

**Prompt:** Whimsical colors danced in the sky as the girl watched from her treehouse. She loved to paint, filling her world with bright strokes and deep shades. One day, while mixing blue and green, she saw a strange shape in the clouds. Curious, she climbed down and ran to the field. There, she met a small alien with big eyes and a shiny body. It looked sad, and she asked, "What's wrong?" The alien pointed to a gray spot in the sky where colors used to be. Inspired, she took out her paint and invited it to help. Together, they painted the sky with bright pinks and yellows. The gray spot slowly turned into a beautiful sunset. The alien smiled, and the girl felt a warm joy in her heart. As the colors shone, the alien floated up, leaving behind a sparkle in the air, a reminder[ of] their colorful adventure.

**L call (sees only z_L):**

> Marked token: "the". Position: 5 of 5. The marked token is the determiner "the" in "the whole world," a noun phrase completing the story's moral about love conquering fear.
>
> The state likely encodes that this is a children's fable/fairy tale genre with simple, warm register and an explicit moral lesson.
>
> Key entities: Mia (wise woman), the magical talking fox guide, the brave knight Leo, the dragon as antagonist to be defeated.
>
> Narrative arc resolution: conflict resolved through dialogue ("I will not fight!"), so the state may encode the story's completed moral theme.
>
> Position marks near-final passage; the model must soon generate a response or continuation after the user turn.

**H call (sees only z_H):**

> Marked token: "of". Position: 1 of 5. The marked token is the preposition "of" in "a bag of popcorn," a low-content function word whose state mainly encodes surrounding context.
>
> The prompt poses an arithmetic word problem requiring computing popcorn count from remaining money (₣200) and prices (₣3, ₣6, ₣7).
>
> Key entities and values: Yusuf, ₣200 total, quantities 8 bags × 3, 24 bags × 6, 15 bags × 7.
>
> Register: elementary math word-problem genre with distractor-free structure; answer expected as a single numeric value.

**What is right and wrong:** L call: genre right (children's fable / fairy-tale register with a moral), "position marks near-final passage; the model must soon generate a response" is right (last fifth, the prompt ends with the model turn), characters invented (Mia, a fox, a knight, a dragon). H call: genre wrong (an arithmetic word problem about popcorn), invented numbers; the token "of" is right.

# A pattern across the three examples (and a measured version)

In all three examples the L call gets the genre right and the H call gets it wrong. Measured over all rows with a crude keyword rule (the genre word of the source appears anywhere in the field: gsm `word problem|arithmetic|math`, da_instruct `danish|dansk`, SimpleStories and MuSR `narrative|story|fiction|fairy|fable|mystery`): E11 iid L call 0.78 vs H call 0.55 (paired L minus H +0.23 [+0.12, +0.34]), ood 0.77 vs 0.34 (+0.43 [+0.31, +0.55]); E12 iid 0.88 vs 0.64 (+0.24 [+0.12, +0.35]), ood 0.80 vs 0.40 (+0.40 [+0.27, +0.52]); the earlier split SFT without facts (a separate SFT run) iid 0.85 vs 0.64 (+0.21 [+0.10, +0.31]), ood 0.74 vs 0.60 (+0.14 [+0.01, +0.26]). n=100 rows per split, 4000 row resamples. Caveats: a keyword proxy, not a classifier; the source probe is a tie at the ceiling (z_L 0.999, z_H 0.992), so this is a difference in what the verbalizer extracts, not in what the vectors hold; the L-versus-H genre gap may partly come from the L call writing a genre sentence first. Use it as a supporting observation, not as a headline.

# Better single example: SimpleStories, token 308 of 319, the token is "she" (true position 5 of 5)

Chosen because both calls get the genre right (a children's story), unlike the earlier example 3 where the H call described an arithmetic problem. Selection: strict pattern (L call position right and token wrong, H call token right and position wrong) AND both calls name the right genre by the keyword rule: 5 of 175 non-last rows in E11 (3%), 7 of 175 in E12. The L call's wrong token ("they") is another pronoun, a near miss.

**Prompt:** Along a winding path, a girl named Kim wandered. She had heard of a hidden cave with bright crystals. But rumors spoke of a ghost that haunted the cave. "I must find it," she told herself. She wanted to see the crystals and be brave, even if there was a ghost. As night fell, Kim held her flashlight tightly. The shadows danced around her. She heard a soft whisper in the wind. "Is it the ghost?" she thought, feeling a chill. But she pressed on, determined. Soon, she reached the cave's mouth, dark and foreboding. Inside the cave, the air was cool and still. She shone her light around. Suddenly, she heard a loud noise! "Is this the ghost?" she thought, feeling her heart pound. But as she looked, she saw a rock falling. "Just a rock!" she sighed, feeling silly for being scared. Deeper in the cave, she saw a glow. "The crystals!" she gasped. They sparkled like stars, filling the cave with light. As she reached out to touch one, she heard the whisper again. "Help me..." it said. Kim turned quickly, but it was just her imagination playing tricks on her. With the crystals in her hands, Kim felt strong. She had faced her fears. "I'm not afraid anymore," she shouted joyfully. She left the cave, feeling proud of her bravery. The ghost was only a story, and[ she] had found real magic.

**L call (sees only z_L):**

> Marked token: "they". Position: 5 of 5. The marked token is the pronoun "they" in a children's story about friendship between a boy and an alien, resolving to the two friends collectively.
>
> The state likely encodes that this is a narrative continuation task; the model must generate a fitting ending or response after the user turn ends.
>
> Key entities and arc: Leo (boy), the friendly alien, their shared adventure with shiny rocks, homecoming, and gratitude — a warm resolution already reached.
>
> Genre/register: simple fairy-tale-like children's story with short sentences, dialogue, and moral lesson ("friendship could take many forms").
>
> Local syntax: "they" is the subject of "returned," referring back to the pair as collective agents.

**H call (sees only z_H):**

> Marked token: "she". Position: 3 of 5. The marked token is the pronoun "she" in a children's story about Mia, so the state encodes its referent (Mia) and narrative context.
>
> The prompt poses an implicit comprehension question: what happened to Mia after she returned home — likely answered by the ending ("home," "wonderful day").
>
> Key entities/constraints: Mia, her dog, the lost puppy found on the porch; emotional arc from sadness to joy.
>
> Genre/register: simple fairy-tale-like children's story with dialogue, past tense, happy resolution.

**What is right and wrong:** L call: position right (5 of 5), genre right (children's story, a narrative continuation task where the model must produce an ending after the user turn: true), token wrong but a near miss ("they" for "she"), characters invented (a boy, an alien, Leo). H call: token right ("she", a pronoun in a children's story), genre right, position wrong (3 of 5, true 5), the named referent invented (Mia), details after that invented. Backup with the same pattern and a digit token: gsm-symbolic token 14 of 52 ("2" in "20 days"): the H call names "2", the L call names the NEXT digit "0" (the neighbouring token), both state a math word problem but invent its numbers.

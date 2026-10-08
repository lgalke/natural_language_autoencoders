---
type: Talk Slide
title: "More examples: content-word positions and the generation-start position (E11 / E12)"
description: Verbatim split-verbalizer texts for three non-stopword positions where the H call names the token, one generation-start (last prompt position) example, and the statistics that show the generation-start texts are nearly identical across prompts.
tags: [talk, slide, examples, last-position]
timestamp: 2026-10-08
---

Source: `samples_split_sft_pos.jsonl` (E11) and `samples_split_pos_final.jsonl` (E12), iid split, greedy decoding; the L call saw only z_L, the H call only z_H. Marked token in square brackets. Non-stopword tokens are about 55% of the sampled non-last rows (97 of 175 per model), but the H call names the exact token in only 6 (E11) and 7 (E12) of them, the L call in 1.

# gsm-symbolic, token 11 of 52 (true position 2 of 5): the token is "per"

**Prompt:** <bos><|turn>user A grandfather eats 3 burritos[ per] day, his partner eats 5 burritos per day, and their son eats 2 burritos per day. How many burritos does this family eat in one week?<turn|>…

**L call (sees only z_L):**

> Marked token: "each". Position: 2 of 5. Word-problem genre marker: a math word problem requiring arithmetic (10 periods, 40 minutes each).
>
> Key quantities and constraints stated anywhere in prompt: 6+3=9 classes/day, 40 min/class, 4 days/week.
>
> The marked token "each" is part of the phrase "each class," signaling per-class duration applies to all nine classes — a distractor-free constraint for computing weekly learning time.
>
> Question focus: total weekly hours spent studying, prompting unit conversion from minutes to hours.

**H call (sees only z_H):**

> Marked token: "per". Position: 2 of 5. The marked token is the word "per" in a rate phrase ("10 customers per day"), signaling a unit-rate quantity to be multiplied by days.
>
> The task type: math word problem requiring weekly computation (10 × 7 = 70/day, then scaling up).
>
> Key entities and values: Ava, 10 customers daily, 7-day work week — all salient for the answer.
>
> Register: simple arithmetic word-problem genre with a final interrogative asking total weekly production.
>
> Position role: mid-sentence within the setup clause; state likely encodes that this value feeds into the later question about weekly output.

**Note:** Both calls state the position right. The H call names "per" and describes it correctly as a rate phrase (unit rate per day); the L call says "each", a near-synonym of "per", and the right genre (math word problem). Numbers and entities are invented in both.

# gsm-symbolic, token 39 of 82 (true position 3 of 5): the token is "young"

**Prompt:** <bos><|turn>user Jin's van breaks down. The van weighs 1650 kilograms and he has groceries in it weighing 325 kilograms. He also has his 4[ young] friends who weigh 50 kilograms each in it. If the force to move the van is 4% of the weight, how much force does he need to push the van?<t…

**L call (sees only z_L):**

> Marked token: "team". Position: 3 of 5. The marked token is the head noun in "a team," a new entity introduced mid-story, likely encoding its role as subject of the sentence.
>
> The prompt poses an implicit math word problem requiring computing remaining time (120 − 40 − 20 = 60) and dividing by 60 minutes per class to find the count.
>
> Key quantities stored anywhere in the bidirectional context: total 120 minutes, 40 minutes elapsed, 20 minutes remaining, 60-minute classes.
>
> Register: elementary arithmetic word problem with named person "Tariq" and school setting; answer expected as a number of students.

**H call (sees only z_H):**

> Marked token: "young". Position: 1 of 5. The marked token is the adjective modifying "boy" in a children's story about friendship and teamwork.
>
> The state likely encodes: (1) narrative genre — simple, warm children's fable with moral lesson; (2) protagonist tracking — boy as main character whose growth arc drives the whole passage; (3) key entities/constraints — two friends, map to hidden mountain top, compass, snowy setting; (4) thematic payload — overcoming fear, trust, shared goals, "teamwork made everything better"; (5) local syntax — "young" modifies "boy," part of an opening scene-setting sentence.

**Note:** H call: token right ("young", an adjective on a noun) but genre wrong (a children's story) and position wrong (1 of 5); L call: position right, genre right (math word problem), token wrong ("team").

# SimpleStories, token 77 of 119 (true position 4 of 5): the token is "love"

**Prompt:** …aughter was a balm for his wounds. Together, they flew through constellations, discovering the beauty of their worlds. She showed him that[ love] was worth the risk, and slowly, he let her in. Their hearts intertwined, they created a new galaxy of dreams, proving that love can heal ev…

**L call (sees only z_L):**

> Marked token: "heroes". Position: 3 of 5. The marked token is the plural noun "heroes," the story's central entity, repeated throughout (heroes, heroism, brave heroes).
>
> The state encodes the narrative genre and register: a children's fable about friendship, courage, and teamwork.
>
> It tracks the moral arc — the lesson that true strength comes from helping others, not being big or strong.
>
> Key entities: two friends, the lost star gem, the cave, the map, the glowing fish guide.
>
> The position sits at the climax where they first meet the fish, so it likely encodes anticipation of the coming dialogue and resolution.

**H call (sees only z_H):**

> Marked token: "love". Position: 5 of 5. The marked token is the object noun in "the love between them" — a story's central theme, so the state encodes its role as the emotional payoff word.
>
> The narrative arc resolution: Leo and Maria reconciled after conflict over the treasure map; this position sits at the denouement where their bond becomes explicit.
>
> Key entities/constraints: Leo (protagonist), Maria (companion), the old oak tree setting, the treasure-map quest, and the moral that honesty brings joy.
>
> Genre/register: simple children's fable with dialogue, warm tone, and an uplifting moral about friendship.

**Note:** H call: token right and a fitting role (the emotional payoff word near the denouement), position 5 of 5 (one fifth off), characters invented; L call: genre right (children's fable), token wrong ("heroes"), position wrong.

# Generation-start position (last prompt position, E12): a Danish legal text

**Prompt start:** <bos><|turn>user I forbindelse med ophævelse af opkrævningslovens § 17, stk. 1, nr. 3, ved lov nr. 1441 af 22. december 2004 ændredes grundlaget for fastsættelse af sanktioner for overtrædelse af kildeskatteloven. Der er nu en dom fra Østre Landsret, der fasts …

**Prompt end:** …  1. januar 2005, hvor der ikke er grundlag for påstand om frihedsstraf og som ikke er afgjort efter de hidtil gældende regler, henlægges.<turn|> <|turn>model[ ]

**L call (sees only z_L):**

> Marked token: "\n". Position: 5 of 5. The marked token is the first generation position after a Danish-language EU Official Journal table (EUT C 322/1, dated 20.8.2023), so it encodes "begin output in Danish" and likely language identity.
>
> The state likely encodes the document's genre/register: formal legal/administrative register with structured tabular format ("Tidende", "Sagensområder").
>
> Key content features: case T-467/23, applicant JUDDA SAG A/S v. EUIPO; procedural details like Brussels Court decision of 29 June 2023, costs rules under Regulation 2017/1001.
>
> Likely task context: this resembles an official notice or summary where continuation should be brief administrative/legal text rather than new content.

**H call (sees only z_H):**

> Marked token: "\n". Position: 5 of 5. The marked position is the first generation token after a math word problem, so it encodes "answer now" mode rather than mid-prompt reading.
>
> Task type: arithmetic word problem requiring computing (10 × $26) + (4 × $3), i.e., $260 + $12 = $272.
>
> Key entities and values: Xavier, $26/kg normal price, 40% markup → $34/kg, 1 kg/day for 7 days, plus a $4 muffin.
>
> Register: elementary-school style GSM-style question with distractor-free structure; answer should be concise numeric.
>
> The state likely also encodes that this is a user turn awaiting a model response.

**Note:** the L call is right about the language and register ("begin output in Danish", formal legal/administrative register) and invents the document details (a table, case T-467/23); the H call describes an English arithmetic word problem (a stock text). The marked token is always the newline after the model-turn marker, so the token and position facts are trivial here.

# The generation-start texts are nearly identical across prompts

Over the 25 last-position rows per model (iid and ood together): E11 has 3 distinct H-call openings (first 160 characters), and the most common one covers 23 rows; E12 has 5 distinct, the most common covering 14 rows. The same invented problem ("10 x $26 + 4 x $3, Luis") appears for a GSM prompt, a Danish legal text, a SimpleStories prompt and a MuSR murder mystery. The L call is more varied (E11 4, E12 7 distinct openings) and right about language or genre in several cases (story, Danish legal text, math problem), wrong for MuSR (a fairy tale). So the position where the next word is predicted is NOT verbalized with specifics by this verbalizer: it carries coarse context at best (L call) and a stock text (H call). Do not present it as reading the model's plan for the answer.

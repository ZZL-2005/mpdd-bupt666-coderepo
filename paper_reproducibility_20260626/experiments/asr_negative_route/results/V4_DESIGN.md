# V4 Strong-Prior Design

## Goal

Use PHQ-9 itself as the architecture prior: predict nine 0..3 item scores, then
sum them. Learning is optional calibration only. The main object is an
interpretable, questionnaire-aligned prior that can explain each sample.

## Inputs

- Event 1 ASR text.
- ASR-derived item evidence features.
- Big-Five scores, especially Neuroticism and Extraversion, used as gates rather
  than direct score predictors.

## Item Alignment

| PHQ item | Direct ASR evidence | Contextual ASR evidence | Gates |
|---|---|---|---|
| 1 interest | no interest, boredom, no motivation | depletion, gaming no longer fun, avoidance | raised by overload/depletion; capped by joy-dominant and strong functioning |
| 2 mood | sadness, worry, pressure, anxiety, distress | rumination, relationship/family burden, task blockage | amplified by high neuroticism; capped by recovery/protection |
| 3 sleep | sleep quality decline, late sleep, insomnia | roommate noise, stress-related sleep | capped when sleep is only casual or resolved |
| 4 fatigue | tired, no energy | illness, workload, deadline loops, task stuck | raised by physical burden and overload |
| 5 appetite | poor appetite, cannot eat, weight/food change | illness-related appetite | capped when food references are ordinary meals |
| 6 self-worth | useless, failure, not enough, not worthy | social isolation, future self-doubt, competence doubt | high neuroticism plus self-doubt raises score |
| 7 concentration | cannot focus, low efficiency, task failure | project stuck, many exams, deadline pressure | capped when there is explicit control/recovery |
| 8 psychomotor | slowing/restlessness proxy | procrastination-anxiety loop, dysregulated coping, low-content | mostly context-driven from ASR |
| 9 self-harm | explicit current self-harm/death-wish only | recency cues around the mention | high priority; historical childhood/past-story mentions are not scored as current PHQ9 item 9 |

## Region Logic

The router first handles high-priority special cases, then assigns the sample to
the strongest explanatory region.

1. `high_risk_self_harm`: explicit current self-harm or death-wish language.
2. `low_content`: very short/filler-only responses.
3. `historical_resolved`: past/resolved narratives dominate current distress.
4. `functional_stress`: future/work stress exists but control and protection are explicit.
5. `trait_amplified_overload`: high neuroticism plus overload, future doubt, or depletion.
6. Domain burden regions: physical, relationship/family, task blockage, social isolation.
7. Generic fallbacks: direct symptom, protected conflict, conflict positive/negative, weak semantic.

## Key Design Moves

- Raw symptom matching alone overcalls event_1 because the prompt asks about
  worries and sadness. V4 therefore adds prompt-context and protective caps.
- Low-content ASR is not treated as missing. In this task, a near non-response
  can itself be a depression-related signal.
- High neuroticism does not directly add points. It only amplifies concrete text
  evidence such as overload, uncertainty, depletion, or self-doubt.
- Positive content is not simply subtracted. It only caps when it shows
  functioning, recovery, control, or stable joy.
- PHQ9 item 9 is recency-gated: a self-harm phrase in a childhood or past-story
  context is treated as historical evidence, not current two-week PHQ9 evidence.
- Item scores are intentionally coarse. A 1-point residual is acceptable because
  the true labels are total PHQ-9 only, not item-level annotations.

## Current Limitation

This design was created by auditing the Young train split. Its current result
shows the training samples are explainable by a compact PHQ-aligned prior, but
it does not prove generalization. The next step should be applying the same
fixed regions and item rules to held-out data, then only tuning calibration.

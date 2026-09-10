# Novelty memo

Written after the verified literature review (`literature_matrix.csv`), before results.
No novelty claim is made beyond what the checks below support.

## What is already known

- **Modern news counts.** Men dominate mentions and especially quotations in
  contemporary news: the Gender Gap Tracker (Asr et al. 2021) finds men quoted about
  three times as often as women, and GMMP 2020 finds women were 25% of people seen
  or heard in the news. Jia et al. (2016) show women are "seen more than heard".
- **Long-run gender associations in text.** Diachronic embeddings trained on books
  and newspapers track occupational change (Garg et al. 2018). In fiction, gendered
  description of characters becomes less differentiated from 1780 to 2007
  (Underwood, Bamman & Lee 2018).
- **Measurement fragility.** OCR noise degrades NER and parsing unevenly across tasks
  (van Strien et al. 2020). Embedding neighbourhoods are unstable under small corpus
  changes (Antoniak & Mimno 2018). Coreference systems encode occupational gender
  stereotypes (Rudinger et al. 2018). Most NLP gender-bias work leaves its theory of
  gender implicit (Devinney et al. 2022).

## Gap this project targets

We found no study that does all three of:

1. measures **person-level** gender representation (who appears, in which role, who
   is quoted) in **historical U.S. news across six decades**;
2. uses **only textual gender signals** (honorifics, pronouns) with UNKNOWN kept as a
   category, and no name-to-gender inference;
3. treats **measurement method as a variable**: several role-measurement methods,
   validation against reference labels, time-dependent disagreement, and a
   multiverse of analysis choices, with corpus composition (the 1923 digitisation
   cliff and one newspaper's growing share) modelled explicitly.

## Claims we will and will not make

- **Will claim**: a reproducible, validated measurement pipeline for this corpus;
  quantified method disagreement; and whichever trends survive the robustness grid.
- **Will not claim**: first to study gender in historical newspapers (unlikely to be
  true, and not exhaustively checked); representativeness of "American news" beyond
  the digitised, sampled corpus; causal explanations.

## Residual risk

The literature search was targeted (sixteen verified works), not systematic. Relevant
work in history, journalism studies and digital humanities, including studies of
women's pages and society news, is probably under-represented. That is listed as a
limitation and as a next step.

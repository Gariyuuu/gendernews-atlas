# Research questions

## Primary

> How did gender-role representations in U.S. newspapers change between 1900 and 1963,
> and how robust are those conclusions to alternative NLP measurement methods?

The second clause is the contribution. The project reports a trend only together with
evidence about whether it survives different operationalizations of *person*,
*gender signal*, and *role*.

## Secondary questions → hypotheses → analyses

| # | Question | Hypothesis | Analysis / result file |
|---|---|---|---|
| Q1 | How did the visibility of women-signalled persons change? | H1 | yearly shares, HAC trend → `role_trends.parquet` (role = ALL) |
| Q2 | How did representation in public-office, professional and business roles change? | H2 | role-conditional shares by method → `role_trends.parquet` |
| Q3 | Who is quoted? | H7 | attributed speech events → `quote_trends.parquet` |
| Q4 | Which verbs place people as agents vs patients? | exploratory | agency log-odds, FDR → `language.parquet` |
| Q5 | Which modifiers are attached to people? | exploratory | modifier log-odds, FDR → `language.parquet` |
| Q6 | How does representation vary by inferred topic and page position? | H3 | topic-stratified shares → `topic_adjusted.parquet` |
| Q7 | Do rule-based, supervised, contextual and LLM measurements tell the same story? | H4, H5 | method comparison → `method_agreement.parquet` |
| Q8 | Does corpus composition (newspaper mix, topic mix, the 1923 digitization cliff) explain part of the trend? | H3 | FE-adjusted vs raw micro models → `topic_adjusted.parquet` |
| Q9 | Does the share of people we *cannot* gender-classify change over time, and does that drive trends? | H6 | UNKNOWN share, bounds → `robustness.parquet` |
| Q10 | Which conclusions survive every reasonable specification? | all | multiverse grid → `robustness.parquet`, `claims_registry.md` |

## Explicitly out of scope

- Causal claims about *why* representation changed. Historical events are context only.
- Claims about individuals' gender identity. The object of measurement is the gender
  **signalled in the text** by period journalistic conventions.
- Race, class and region as outcomes. They matter for interpretation (the women most
  visible in society pages were overwhelmingly white and affluent) but are not
  measured here; see `limitations.md`.
- Post-1963 news. AmericanStories stops at 1963 because of copyright.

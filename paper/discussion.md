<!-- Hand-written after results were frozen. Deliberately carries no numbers: every figure
     quoted in the paper is generated from results/, and this text points to sections and
     figures instead, so it stays true when results are regenerated. -->

**What rose, and where.** Among people the newspapers marked as women or men, the share
marked as women rose from 1900 to 1963 in every raw specification of the multiverse (§11).
Most of that rise is composition. Holding topic and newspaper fixed removes most of the
slope (H3, §10). Newspaper fixed effects alone remove all of it: within an individual
newspaper we detect no trend. The raw series therefore mainly records which newspapers,
and which kinds of content, fill the sampled pages. It does not record a change in how a
given paper wrote about people. After 1922 the corpus holds far fewer titles and is
dominated by the Washington *Evening Star* (§3). The topic mix also shifts toward topics
in which women are named more often.

The two-group decomposition by newspaper points the other way and puts the rise inside
the groups. That is because "all other papers" is not a stable set: its membership
changes sharply at 1922. We read the fixed-effects result as the more credible one, and
we treat the visibility trend as a fact about this digitised corpus, not about the
American press.

**Authority tracks visibility.** Under every role method, women's share of authority
roles rose (§9). But it rose in step with women's share of all people. The gap between
the two did not narrow under method A or method B, so H2 is unsupported. More women
appear in authority roles in these pages, but that is not evidence of a change in *which*
roles women were written into, relative to how often they were written about at all.
Every method over-assigns authority to women relative to the reference labels (§5). The
measured level of women's authority share is therefore inflated, and the true gap is
probably larger than the one we report.

**The method changes the size, not the direction.** The role methods agree that women's
share of authority roles rose. They disagree by more than a factor of two on how fast
(H4), and by far more for public office alone. Pairwise agreement between methods is low,
and it drifts across periods (Figure 14). The lexical window picks up roles from nearby
words, and it gives the largest trend in most headline roles (H5, tentative). A study that
reports one of these pipelines reports one point from a wide range of defensible answers.

**Errors are gendered.** Three measurement problems push the same way.

- Men are named by office or surname alone far more often than women. They therefore
  fall into UNKNOWN more often, which inflates women's share among classified people.
- Every role method is less precise about authority for women than for men.
- Couple constructions attach a husband's title to his wife. The titled cases are removed
  by decision D17; the untitled ones are not.

None of these reverses the sign of a headline trend: the UNKNOWN bounds keep it positive
(H6). But together they mean that *levels* of women's share, and above all their share of
authority roles, should not be read as facts about the period.

**The quotation gap is the firmest finding.** Women's share of named speakers sits far
below their share of people in every period. The gap widens across all four periods,
under name-only and name-plus-pronoun attribution alike (H7, Figure 12). It survives the
choices that move the other results because it compares two quantities measured on the
same people by the same rules.

It still rests on a speech rule with low recall (§5). Women's speech may more often be
reported in forms the rule misses: through a spouse, in a club secretary's report, or as a
*she said* far from the name. If so, the gap is overstated. The reference sample is too
small to test this by gender.

**The LLM reads roles well and gender badly.** Against the reference labels, the
open-weights LLM has the best authority-role F1 of the five methods. It also assigns a
gender to most of the people about whom the text says nothing (§5). As a role reader it is
strong; as a gender reader it is not usable without constraints. It was applied to a small
year-balanced subsample only, so its trend estimate is imprecise.

**What would change these conclusions.**

- Human re-annotation of the reference sample, which could move every validation figure.
- Coreference that handles couples and binding constraints.
- A corpus whose newspaper membership is stable across 1922. That would allow
  within-newspaper change to be estimated on more than a few titles after that year.

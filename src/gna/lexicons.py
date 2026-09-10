"""Lexical resources for gender signals and the role taxonomy.

Design rules (see research/measurement_framework.md):

* PRIMARY gender signal = courtesy/noble honorifics + local pronouns.  Gendered
  *nouns* (wife, widow, actress, Sister, ...) are an EXTENDED tier used only as a
  sensitivity analysis, because several of them are also role terms (FAMILY, SOCIAL)
  and would make role-by-gender estimates partly mechanical.
* Occupational compounds with "-man" (chairman, congressman, alderman) are role terms
  only, never gender signals: period usage applied them to women office-holders.
* No first names are used anywhere as gender evidence.
"""
from __future__ import annotations

# ---------------------------------------------------------------- gender: primary tier
HONORIFIC_F = {"mrs", "miss", "ms", "mme", "madame", "mlle", "mademoiselle", "misses", "mmes",
               "lady", "dame", "queen", "princess", "duchess", "countess", "baroness", "empress",
               "marchioness", "senora", "senorita", "frau", "fraulein"}
HONORIFIC_M = {"mr", "messrs", "sir", "lord", "king", "prince", "duke", "baron", "emperor", "earl",
               "marquis", "master", "senor", "herr", "monsieur"}
# French "M." is deliberately excluded: it is indistinguishable from a first-name
# initial ("M. J. Smith") in OCR text.

PRONOUN_F = {"she", "her", "hers", "herself"}
PRONOUN_M = {"he", "him", "his", "himself"}

# ---------------------------------------------------------------- gender: extended tier
GNOUN_F = {"woman", "girl", "wife", "widow", "mother", "daughter", "sister", "aunt", "niece",
           "grandmother", "granddaughter", "bride", "housewife", "matron", "spinster", "heiress",
           "actress", "hostess", "waitress", "stewardess", "authoress", "aviatrix", "sportswoman",
           "chairwoman", "congresswoman", "policewoman", "spokeswoman", "businesswoman", "horsewoman",
           "fiancee", "debutante", "nun", "clubwoman", "postmistress", "laundress", "seamstress",
           "songstress", "sweetheart", "girlfriend", "grandma", "mom"}
GNOUN_M = {"man", "boy", "husband", "widower", "father", "son", "brother", "uncle", "nephew",
           "grandfather", "grandson", "bridegroom", "fiance", "bachelor", "monk", "businessman",
           "sportsman", "horseman", "grandpa", "dad", "boyfriend"}

# ---------------------------------------------------------------- role taxonomy
ROLES = ["PUBLIC_OFFICE", "MILITARY", "BUSINESS", "LABOR", "PROFESSIONAL", "ARTS_SPORTS",
         "CIVIC", "SOCIAL", "FAMILY", "CRIME_ACCIDENT"]
# Pre-registered composite for H2 (fixed before outcomes were computed):
AUTHORITY = {"PUBLIC_OFFICE", "BUSINESS", "PROFESSIONAL"}

_R = {
    "PUBLIC_OFFICE": """president senator sen representative rep congressman congresswoman governor gov mayor
        alderman councilman councilwoman assemblyman legislator lawmaker commissioner secretary ambassador
        consul diplomat judge justice magistrate sheriff marshal policeman patrolman detective inspector
        prosecutor solicitor postmaster postmistress official candidate nominee delegate premier
        chancellor attorney-general coroner constable warden trooper""",
    "MILITARY": """general gen colonel col major maj captain capt lieutenant lt sergeant sgt corporal cpl
        private pvt admiral adm commander cmdr ensign soldier sailor marine aviator pilot infantryman
        brigadier""",
    "BUSINESS": """merchant banker broker manufacturer businessman businesswoman proprietor owner executive
        manager salesman saleswoman dealer storekeeper grocer contractor industrialist capitalist
        financier employer partner realtor jeweler druggist publisher""",
    "LABOR": """worker laborer employee clerk stenographer typist farmer miner carpenter mechanic driver
        chauffeur servant maid cook housekeeper janitor porter waiter waitress operator seamstress
        dressmaker milliner laundress machinist fireman motorman farmhand striker stewardess brakeman
        engineer conductor""",
    "PROFESSIONAL": """doctor dr physician surgeon dentist nurse pharmacist professor prof teacher instructor
        principal dean educator scientist chemist physicist biologist economist architect lawyer attorney
        counsel author writer novelist poet journalist editor reporter correspondent expert specialist
        researcher librarian missionary minister pastor rev reverend rector priest rabbi bishop archbishop
        cardinal clergyman evangelist deacon chaplain psychologist psychiatrist astronomer geologist
        veterinarian lecturer superintendent""",
    "ARTS_SPORTS": """actor actress singer soprano contralto tenor baritone pianist violinist organist
        musician composer artist painter sculptor dancer star comedian entertainer vocalist playwright
        player pitcher catcher outfielder infielder shortstop quarterback halfback fullback coach boxer
        fighter champion golfer jockey athlete swimmer runner sprinter wrestler skater racer""",
    "CIVIC": """chairman chairwoman member organizer clubwoman suffragist suffragette activist reformer
        volunteer trustee regent treasurer leader founder""",
    "SOCIAL": """hostess host debutante bride bridegroom bridesmaid patroness patron chaperon chaperone
        guest""",
    "FAMILY": """wife husband widow widower mother father daughter son sister brother aunt uncle niece
        nephew grandmother grandfather grandson granddaughter fiancee fiance heir heiress parent""",
    "CRIME_ACCIDENT": """victim suspect defendant accused prisoner convict murderer slayer robber thief burglar
        bandit gunman killer criminal fugitive culprit""",
}
ROLE_TERMS: dict[str, str] = {}
for _role, _words in _R.items():
    for _w in _words.split():
        ROLE_TERMS.setdefault(_w, _role)  # first listing wins for Method A (naive single-sense map)

# Method A uses ROLE_TERMS verbatim (one sense per word).  Method B additionally
# disambiguates organisational heads by the organisation they head:
ORG_HEADS = {"president", "vice-president", "secretary", "treasurer", "chairman", "chairwoman", "director",
             "head", "member", "manager", "officer", "leader", "founder", "organizer", "regent", "trustee"}
ORG_KEYWORDS = {
    "BUSINESS": {"company", "co", "corporation", "corp", "bank", "firm", "store", "railroad", "railway",
                 "inc", "mills", "mill", "factory", "works", "trust", "insurance", "exchange", "hotel",
                 "motor", "motors", "steel", "oil", "chamber"},
    "CIVIC": {"club", "society", "league", "association", "auxiliary", "federation", "chapter", "guild",
              "circle", "alliance", "union", "order", "lodge", "committee", "sorority", "fraternity",
              "daughters", "legion", "cross", "y.w.c.a", "y.m.c.a", "church", "parish", "congregation",
              "party", "movement", "foundation", "fund"},
    "PUBLIC_OFFICE": {"board", "council", "senate", "house", "department", "bureau", "commission", "court",
                      "state", "states", "city", "county", "nation", "government", "administration",
                      "treasury", "navy", "war", "interior", "agriculture", "commerce", "labor", "republic",
                      "congress", "legislature", "assembly", "cabinet", "police", "district"},
    "MILITARY": {"army", "regiment", "battalion", "brigade", "corps", "division", "guard", "infantry",
                 "artillery", "cavalry", "squadron", "fleet"},
    "PROFESSIONAL": {"university", "college", "school", "hospital", "institute", "academy", "seminary",
                     "library", "laboratory", "museum", "observatory"},
}
ORG_HEAD_DEFAULT = {"president": "PUBLIC_OFFICE", "vice-president": "PUBLIC_OFFICE",
                    "secretary": "PUBLIC_OFFICE", "treasurer": "CIVIC", "chairman": "CIVIC",
                    "chairwoman": "CIVIC", "director": "BUSINESS", "head": None, "member": "CIVIC",
                    "manager": "BUSINESS", "officer": None, "leader": "CIVIC", "founder": "CIVIC",
                    "organizer": "CIVIC", "regent": "CIVIC", "trustee": "CIVIC"}

# Titles that may directly precede a name.  Courtesy honorifics carry gender, not role.
TITLE_ROLE = {
    "dr": "PROFESSIONAL", "doctor": "PROFESSIONAL", "prof": "PROFESSIONAL", "professor": "PROFESSIONAL",
    "rev": "PROFESSIONAL", "reverend": "PROFESSIONAL", "bishop": "PROFESSIONAL", "rabbi": "PROFESSIONAL",
    "father": None, "sister": None, "brother": None, "mother": None,   # gendered religious/kin titles: ext. tier only
    "senator": "PUBLIC_OFFICE", "sen": "PUBLIC_OFFICE", "rep": "PUBLIC_OFFICE", "representative": "PUBLIC_OFFICE",
    "congressman": "PUBLIC_OFFICE", "congresswoman": "PUBLIC_OFFICE", "gov": "PUBLIC_OFFICE",
    "governor": "PUBLIC_OFFICE", "mayor": "PUBLIC_OFFICE", "judge": "PUBLIC_OFFICE", "justice": "PUBLIC_OFFICE",
    "president": "PUBLIC_OFFICE", "secretary": "PUBLIC_OFFICE", "commissioner": "PUBLIC_OFFICE",
    "ambassador": "PUBLIC_OFFICE", "alderman": "PUBLIC_OFFICE", "councilman": "PUBLIC_OFFICE",
    "sheriff": "PUBLIC_OFFICE", "marshal": "PUBLIC_OFFICE", "detective": "PUBLIC_OFFICE",
    "inspector": "PUBLIC_OFFICE", "patrolman": "PUBLIC_OFFICE", "policeman": "PUBLIC_OFFICE",
    "chief": None, "supt": "PROFESSIONAL", "superintendent": "PROFESSIONAL",
    "gen": "MILITARY", "general": "MILITARY", "col": "MILITARY", "colonel": "MILITARY", "maj": "MILITARY",
    "major": "MILITARY", "capt": "MILITARY", "captain": "MILITARY", "lt": "MILITARY", "lieut": "MILITARY",
    "lieutenant": "MILITARY", "sgt": "MILITARY", "sergeant": "MILITARY", "corp": "MILITARY", "cpl": "MILITARY",
    "pvt": "MILITARY", "private": "MILITARY", "adm": "MILITARY", "admiral": "MILITARY", "comdr": "MILITARY",
    "cmdr": "MILITARY", "commander": "MILITARY", "ensign": "MILITARY", "coach": "ARTS_SPORTS",
}
COURTESY = HONORIFIC_F | HONORIFIC_M
ALL_TITLES = COURTESY | set(TITLE_ROLE) | {"hon", "honorable", "jr", "sr"}

CRIME_PASSIVE_VERBS = {"kill", "injure", "hurt", "arrest", "charge", "indict", "convict", "sentence", "murder",
                       "shoot", "stab", "rob", "jail", "drown", "wound", "fine", "electrocute", "hang", "lynch",
                       "burn", "strike", "crush", "scald", "assault", "attack", "kidnap", "abduct", "slay",
                       "beat", "imprison", "execute", "try"}
CRIME_ACTIVE_VERBS = {"plead", "confess", "escape", "flee"}
CRIME_OBJECT_VERBS = {"arrest", "kill", "shoot", "murder", "rob", "injure", "stab", "sentence", "indict",
                      "convict", "hang", "lynch", "slay", "kidnap", "assault", "attack", "execute"}
APPOINT_VERBS = {"elect", "appoint", "name", "choose", "nominate", "select", "reelect", "re-elect", "make",
                 "install", "designate"}

SPEECH_VERBS = {"say", "tell", "declare", "state", "add", "announce", "assert", "explain", "reply", "ask",
                "testify", "insist", "remark", "comment", "report", "write", "warn", "predict", "urge",
                "claim", "deny", "admit", "argue", "contend", "suggest", "recall", "note", "conclude",
                "answer", "exclaim", "shout", "cry", "whisper", "confess", "promise", "agree", "observe",
                "maintain", "protest", "respond", "estimate", "believe", "charge_not"}
SPEECH_VERBS.discard("charge_not")

"""
Streamlit app for legal rhetorical role labeling and summarization.
Save this file as `streamlit_rhetorical_labeling_app.py` and run:
    pip install -r requirements.txt
    streamlit run streamlit_rhetorical_labeling_app.py

Requirements (example):
transformers
torch
streamlit
pandas
scikit-learn
nltk
PyPDF2
requests
bertopic
hdbscan

This file bundles the preprocessing, abbreviation search, model loading (cached),
labeling and summarization UI in Streamlit.
"""

import os
from typing import Any, cast
import streamlit as st
import PyPDF2
import re
import nltk
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline
import json
from huggingface_hub import login

# Page config must be the first Streamlit command
st.set_page_config(page_title="Legal Rhetorical Role Labeling", layout='wide', initial_sidebar_state="expanded")
#st.set_page_config(page_title="Legal Rhetorical Role Labeling", layout='wide', initial_sidebar_state="collapsed")

#torch.classes.__path__ = [] # add this line to manually set it to empty.


# --- NLTK setup ---
# NLTK punkt downloader (kept separate to avoid side effects during import).
# If download fails (common on systems without proper SSL certs), fall back to
# a simple regex-based sentence splitter so the app still runs.

try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
except Exception:
    import re as _re
    def sent_tokenize(text):
        pieces = _re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in pieces if p.strip()]


# --- Abbreviations dictionary (from user's original code) ---
abbreviations_dict = {
"sc/st": "scheduled Caste/Scheduled Tribe",
"(P) Ltd.": "private limited",
#"§": "section",
#"§§": "multiple sections",
#"¶": "paragraph",
"…………..........................j.": "judge name",
"………….........................j.": "judge name",
"a.2d": "atlantic reporter, 2nd series",
"a.a.s.": "acta apostolicae sedis",
"a.b.a.": "american bar association",
"a.c.": "appeal cases",
"a.c.c.": "association of corporate counsel",
"a.d.": "appellate division",
"a.d.a.": "americans with disabilities act",
"a.d.m.k.": "anna dravida munnertra kazhagam",
"a.d.r.": "alternative dispute resolution",
"a.g.": "attorney general",
"a.g.p.": "asom gana parishad",
"a.h.c.p.r.": "agency for healthcare research and quality",
"a.i.a.d.m.k.": "all india anna dravida munnetra kazhagam",
"a.i.f.": "alternative investment fund",
"a.i.f.b.": "all india forward bloc",
"a.i.r.": "all india reporter",
"a.i.t.c.": "all india trinamool congress",
"a.l.j.r.": "australian law journal reports",
"a.m.l.": "anti money laundering",
"a.o.": "agreement officer",
"a.o.r.": "advocate on record",
"a.p.a.c.": "asia-pacific economic cooperation",
"a.p.d.": "affidavit of probable defense",
"a.p.h.l.c.": "all party hill leaders' conference",
"a.p.o.": "assistant prosecution officer",
"a.p.p.": "appendix",
"a.r.t./a.r.t.s.": "article/articles of the constitution of india",
"a.s." : "first appeal",
"a.s.j.": "additional sessions judge",
"a.s.s.": "acta sanctae sedis",
"a.t.r.": "action taken report",
"a.u.m.f.": "authorization for the use of military force",
"a.v.r.": "automatic vote recorder",
"a/a/o": "as assignee of",
"abr.": "abridged, abridgment",
"ad.": "at the suit of",
"adj.": "at the suit of",
"admk": "anna dravida munnertra kazhagam",
"admn":"administration",
"admn.": "administration",
"adr": "alternative dispute resolution",
"ads.": "at the suit of",
"adsm.": "at the suit of",
"adv.": "advocate",
"aff.": "affiliated",
"aff'd": "affirmed",
"ag": "attorney general",
"agen.": "agent",
"agp": "asom gana parishad",
"agril": "agricultural",
"agril.": "agricultural",
"ahcpr": "agency for healthcare research and quality",
"aiadmk": "all india anna dravida munnetra kazhagam",
"aif": "alternative investment fund",
"aifb": "all india forward bloc",
"air": "all india reporter",
"aitc": "all india trinamool congress",
"aml": "anti money laundering",
"amt.": "amount",
"andh": "andhra pradesh",
"anor.": "another",
"anors.": "others",
"anr.": "another",
"aor": "advocate on record",
"aor.": 'advocate on record',
"ap. const.": "apostolic constitution",
"apac": "asia-pacific economic cooperation",
"apd": "affidavit of probable defense",
"aphlc": "all party hill leaders' conference",
"app.": "appendix",
"appl.": "application",
"appls.": "applications",
"appt.": "appointment",
"apptt." : "appointment",
"appx.": "appendix",
"arb.": "arbitration",
"arg.": "argument",
"art.": "article of the constitution of india",
"art.": "article",
"arts.": "articles of the constitution of india",
"artt.": "articles",
"ass'n": "association",
"asso.": "association",
"assocs.": "associations",
"asstt.": "assistant",
"astt.": "assistant",
"atr": "action taken report",
"ats": "at the suit of",
"atty.": "attorney",
"aumf": "authorization for the use of military force",
"avr": "automatic vote recorder",
"b.a.c.": "business advisory committee",
"b.a.i.": "bar association of india",
"b.a.l.c.o.": "bharat aluminum company limited",
"b.a.p.": "bankruptcy appellate panel",
"b.b.a.": " bachelor of business administration",
"b.c.i.": "bar council of india",
"b.h.e.l.": "bharat heavy electricals limited",
"b.i.s.": "bureau of indian standards",
"b.j.d.": "biju janata dal",
"b.j.p.": "bharatiya janata party",
"b.k.d.": "bharatiya kranti dal",
"b.l.d.": "bharatiya lok dal",
"b.n. (i)/(ii)": "rajya sabha bulletin part i/ii",
"b.o.a.": "board of appeals",
"b.p.f.": "bodoland people's front",
"b.s.n.l.": "bharat sanchar nigam limited",
#"b.s.p.": "bahujan samaj party",
"b/o": "behalf of, on behalf of",
"bac": "business advisory committee",
"bai": "bar association of india",
"balco": "bharat aluminum company limited",
"bci": "bar council of india",
"bhel": "bharat heavy electricals limited",
"bis": "bureau of indian standards",
"bjd": "biju janata dal",
"bjp": "bharatiya janata party",
"bk.": "book",
"bkd": "bharatiya kranti dal",
"bks.": "books",
"bld": "bharatiya lok dal",
"bn. (i)/(ii)": "rajya sabha bulletin part i/ii",
"bpf": "bodoland people's front",
"br": "bankruptcy",
"bsnl": "bharat sanchar nigam limited",
"bsp": "bahujan samaj party",
"bull.": "bulletin",
"c&ag": "comptroller & auditor general of india",
"c.&a.g.": "comptroller & auditor general of india",
"c.a. d.e.b.": "constituent assembly debates",
"c.a. deb.": "constituent assembly debates",
"c.a.": "class action",
"c.a.f.c.a.s.s.": "children and family court advisory and support service",
"c.a.r.i.c.o.m.": "caribbean community",
"c.a.s.": "central administrative tribunal",
"c.b.f.c.": "central board of film certification",
"c.b.i.": "central bureau of investigation",
"c.b.j.": "california bar journal",
"c.c.a.": "controller of certifying authorities",
"c.c.d.": "cabinet committee on disinvestment",
"c.c.i.": "competition commission of india",
"c.c.s.": "cabinet committee on security",
"c.c.t.v.": "closed circuit television",
"c.d.": "conference on disarmament",
"c.d.s.": "compulsory deposit scheme",
"c.e.a.c.r.": "committee of experts on the application of conventions and recommendations",
"c.e.c.": "chief election commissioner",
"c.e.d.a.w.": "convention on the elimination of all forms of discrimination against women",
"c.e.n.v.a.t.": "centralized value added tax",
"c.f.r.": "code of federal regulations",
"c.g.h.s.": "central government health scheme",
"c.h.o.g.m.": "commonwealth heads of government meet",
"c.i.a.": "central intelligence agency",
"c.i.c.": "central information commission",
"c.i.s.s.p.": "certified information systems security professional",
"c.j.": "chief justice",
"c.j.i.": "chief justice of india",
"c.j.m.": "chief judicial magistrate",
"c.j.s.c.": "chief judicial magistrate court",
"c.l.": "common law",
"c.l.a.t.": "common law admission test",
"c.m.a.": "civil miscellaneous appeal",
"c.m.s.a": "civil mis.second appeal" ,
"c.o.": "constitution order",
"c.o.d.": "criminal offenses and defenses",
"c.o.f.e.p.o.s.a.": "conservation of foreign exchange and prevention of smuggling act",
"c.o.p.": "certificate of probable cause",
"c.o.p.": "committee of privileges",
"c.o.p.l.o.t.": "committee on papers laid on the table",
"c.o.p.r.a.": "consumer protection act",
"c.o.p.u.": "committee on public undertakings",
"c.o.r.": "committee on rules",
"c.o.s.l.": "committee on subordinate legislation",
"c.p.c.": "code of civil procedure",
"c.p.c.b.": "central pollution control board",
"c.p.i.": "communist party of india",
"c.p.i.(m.)": "communist party of india (marxist)",
"c.r. p.c.": "criminal procedure code",
"c.r.c.": "camera ready copy",
"c.r.p.": "civil revision petition",
"c.r.p.": "criminal procedure",
"c.r.p.(pd)": "civil revision petition (pd)",
"c.r.p.c.": "code of criminal procedure",
"c.r.p.f.": "central reserve police force",
"c.r.y.": "child relief and you",
"c.s. d.e.b.": "council of states debates",
"c.s. deb.": "council of states debates",
"c.s.": "council of states",
"c.s.e.": "centre for science and environment",
"c.s.t.": "central sales tax",
"c.t.b.t.": "comprehensive test ban treaty",
"c.v.c.": "central vigilance commission",
"ca #": "court of appeals (court of appeals for the #th circuit)",
"ca fed.": "court of appeals for the federal circuit",
"cafcass": "children and family court advisory and support service",
"cal.": "california",
"cantt": "cantonments",
"capt.": "captain",
"caricom": "caribbean community",
"cas": "central administrative tribunal",
"caveat" : "caveat",
"cb": "casebook",
"cbfc": "central board of film certification",
"cbi": "central bureau of investigation",
"cc": "commerce clause",
"cc.": "chapters",
"cca": "controller of certifying authorities",
"ccd": "cabinet committee on disinvestment",
"cci": "competition commission of india",
"ccs": "cabinet committee on security",
"cctv": "closed circuit television",
"cd": "closing disclosure",
"cd": "conference on disarmament",
"cdr.": "commander",
"cds": "compulsory deposit scheme",
"ceacr": "committee of experts on the application of conventions and recommendations",
"cec": "chief election commissioner",
"cedaw": "convention on the elimination of all forms of discrimination against women",
"cenvat": "centralized value added tax",
"cf.": "confer",
"cfr": "call for response",
"cghs": "central government health scheme",
"ch.": "chapters",
"chem.":"chemical",
"chogm": "commonwealth heads of government meet",
"chq.": "cheque",
"cia": "central intelligence agency",
"cic": "central information commission",
"cif": "coming into force",
"cissp": "certified information systems security professional",
"cj": "chief justice",
"cji": "chief justice of india",
"cjm": "chief judicial magistrate",
"cjsc": "chief judicial magistrate court",
"cl.": "clause",
"clat": "common law admission test",
"cls.": "clauses",
"cneg": "contributory negligence",
"co": "constitution order",
"co.": "company",
"cod": "criminal offenses and defenses",
"cofeposa": "conservation of foreign exchange and prevention of smuggling act",
"comdt.": "commandant",
"comm": "commission",
"comm.": "commission",
"comml.": "commercial",
"comm'n": "commission",
"commnr.": "commissioner",
"comm'r": "commissioner",
"commr.": "commissioner",
"comr.":"commissioner",
"comrs.": "commissioners",
"cong. rec.": "congressional record",
"constitution": "constitution of india",
"cont. a.": "contempt appeal",
"contd.": "continued",
"cop": "certificate of probable cause",
"cop": "committee of privileges",
"coplot": "committee on papers laid on the table",
"copra": "consumer protection act",
"copu": "committee on public undertakings",
"cor": "committee on rules",
"cor.": "coram, a cause heard 'in the presence of' an auditor of the roman rota",
"corp.": "corporation",
"corpn.": "corporation",
"cosl": "committee on subordinate legislation",
"coun.": "counting",
"cpc": "code of civil procedure",
"cpcb": "central pollution control board",
"cpi": "communist party of india",
"cpi(m)": "communist party of india (marxist)",
"cr.pc": "criminal procedure code",
"cr.": "civil revision",
"cr.p.c.":"criminal procedure code",
"crc": "camera ready copy",
"crl. a.": "criminal appeal",
"c rl.": "criminal",
"crl": "criminal revision leave",
"crl.o.p.": "criminal original petition",
"crl.r.c.": "crl. revision case",
"crm.": "client relationship management",
"cr j": "crime and justice",
"crm-m": 'criminal main',
"cross. obj.": "cross objection" ,
"crp": "criminal procedure",
"crpc": "code of criminal procedure",
"crpf": "central reserve police force",
"crr.": "criminal revision",
"crs": "confidential reports",
"crs": "congressional research service",
"crs.":"confidential reports",
"cry": "child relief and you",
"cs": "council of states",
"cse": "centre for science and environment",
"cst": "central sales tax",
"ctbt": "comprehensive test ban treaty",
"cus": "custom",
"cus.": "custom",
"cvc": "central vigilance commission",
"cx": "constitution",
"cx-c": "cross-claim",
"cxl": "constitutional",
"d": "defendant",
"d.a.": "daily allowance",
"d.a.": "dearness allowance",
"d.c.m.": "differentiated case management",
"d.e.c.": "declaration",
"d.g.c.a.": "directorate general of civil aviation",
"d.i.g.": "deputy inspector general",
"d.m.a.": "disaster management authority",
"d.m.k.": "dravida munnetra kazhagam",
"d.o.j.": "department of justice",
"d.o.l.": "department of labor",
"d.p.a.": "department/ministry of parliamentary affairs",
"d.p.p.": "director of public prosecutions",
"d.r.i.p.": "united nations declaration on the rights of indigenous peoples",
"d.r.t.": "debt recovery tribunal",
"d.s.b.": "dispute settlement body",
"d.s.p.": "democratic socialist party",
"d.s.p.": "deputy superintendent of police",
"d.t.": "date",
"d.u.i.": "driving under the influence",
"d.v.c.": "domestic violence act",
"d/b/a": "doing business as",
"da": "daily allowance",
"da": "dearness allowance",
"dac": "days after contract",
"dec.": "declaration",
"decd": "deceased",
"decd.": "deceased",
"decr.": "decretum",
"def.": "defendant",
"defdt": "defendants",
"defdt.": "defendants",
"dep't": "department",
"deptt": "department",
"dft.": "defendant",
"dgca": "directorate general of civil aviation",
"digest": "parliamentary privileges—digest of cases (lok sabha secretariat)",
"disst.": "district",
"dist": "district",
"dist.":"district",
"distt": "district",
"distt.":"district",
"divl.": "divisional",
"divn.":"division",
"dlf": "delhi land & finance",
"dma": "disaster management authority",
"dmk": "dravida munnetra kazhagam",
"dn.":"division",
"doj": "department of justice",
"dol": "department of labor",
"dpa": "department/ministry of parliamentary affairs",
"dpp": "director of public prosecutions",
"drip": "united nations declaration on the rights of indigenous peoples",
"drt": "debt recovery tribunal",
"dsb": "dispute settlement body",
"dsp": "democratic socialist party",
"dt.": "date",
"dui": "driving under the influence",
"dvc": "domestic violence act",
"e.b.t.": "examination before trial",
"e.c.": "election commission",
"e.c.a.": "essential commodities act",
"e.c.h.r.": "european court of human rights",
"e.c.i.r.": "enforcement case information report",
"e.d.n.": "edition",
"e.e.o.c.": "equal employment opportunity commission",
"e.i.n.": "employer identification number",
"e.l.r.": "election law reports",
"e.o.i.r.": "executive office for immigration review",
"e.p.f.": "employees provident funds",
"e.r.i.s.a.": "employee retirement income security act",
"e.s.m.a.": "essential services maintenance act",
"e.s.t.": "employees' state insurance",
"e.t s.e.q.": "et. sequens (sequentia) (and that which follows)",
"e.t. a.l.": "and others",
"e.v.m.": "electronic voting machine",
"e.x.p.l.n.": "explanation",
"e.x.t.": "extraordinary",
"ec": "election commission",
"eca": "essential commodities act",
"echr": "european court of human rights",
"ed.": "edition",
"ed.":"edition",
"edn.": "edition",
"eds.": "editions/editors",
"edu.": "education",
"ee": "employee",
"eeoc": "equal employment opportunity commission",
"ein": "employer identification number",
"elr": "election law reports",
"encl.": "enclosed",
"encl.": "enclosure",
"envtl.": "environmental",
"eoir": "executive office for immigration review",
"epf": "employees provident funds",
"er": "employer",
"erisa": "employee retirement income security act",
"esma": "essential services maintenance act",
"esq.": "esquire",
"est": "employees' state insurance",
"estb.": "established",
"estt.": "establishment",
"et als.": "and others'",
"et seq": "et. sequens",
"et seq.": "et sequens, latin for 'and following'",
"et. al.": "and others",
"evm": "electronic voting machine",
"ex.": "exhibit",
"exbt.": "exhibit",
"exch.": "exchange",
"excl.": "excluding",
"exe.": "executive",
"exh.":"exhibit",
"expln.": "explanation",
"ext.": "extraordinary",
"exts.": "exhibits",
"f.2d": "federal reporter, 2nd series",
"f.3d": "federal reporter, 3rd series",
"f.app'x": "federal appendix",
"f.c.p.s.": "fellow of the college of physicians and surgeons",
"f.c.r.a.": "foreign contribution regulation act",
"f.d.c.p.a.": "fair debt collection practices act",
"f.d.i.": "foreign direct investment",
"f.e.m.a.": "foreign exchange management act",
"f.e.r.a.": "foreign exchange regulation act",
"f.e.r.p.a.": "family educational rights and privacy act",
"f.i.i.": "foreign institutional investors",
"f.i.r.": "first information report ",
"f.i.r.": "first information report",
"f.l.c.": "foreign legal consultant",
"f.l.s.a.": "fair labor standards act",
"f.m.c.s.a.": "federal motor carrier safety administration",
"f.m.l.a.": "family and medical leave act",
"f.n.": "footnote",
"f.no.": "file number",
"f.o.i.a.": "freedom of information act",
"f.o.r.e.x.": "foreign exchange",
"f.r.d.": "Federal Rules Decision",
"f.t.c.": "fast track court",
"fcps": "fellow of the college of physicians and surgeons",
"fcra": "foreign contribution regulation act",
"fdcpa": "fair debt collection practices act",
"fdi": "foreign direct investment",
"fed. reg.": "federal register",
"fema": "foreign exchange management act",
"fera": "foreign exchange regulation act",
"ferpa": "family educational rights and privacy act",
"fig.": "figure",
"fii": "foreign institutional investors",
"fir": "first information report",
"flc": "foreign legal consultant",
"flsa": "fair labor standards act",
"fmcsa": "federal motor carrier safety administration",
"fmla": "family and medical leave act",
"foia": "freedom of information act",
"fora": "forum",
"fora.":"forum",
"forex": "foreign exchange",
"fr.": "father",
"ftc": "fast track court",
"fwd.": "foreword",
"g.a.a.p.": "generally accepted accounting principles",
"g.a.t.t.": "general agreement on tariffs and trade",
"g.a.z. e.x.t.": "gazette extraordinary",
"g.a.z.": "gazette",
"g.b.p.": "global biosphere programme",
"g.d.p.": "gross domestic product",
"g.d.p.r.": "general data protection regulation",
"g.l.o.b.e.": "global learning and observation to benefit the environment",
"g.m.o.": "genetically modified organism",
"g.n.": "government notice",
"g.n.l.f.": "gorkha national liberation front",
"g.n.p.": "gross national product",
"g.o.i.": "government of india",
"g.p.a.": "general power of attorney",
"g.p.c.": "general purposes committee",
"g.p.o.": "general post office",
"g.s.l.v.": "geosynchronous satellite launch vehicle",
"g.s.r.": "general statutory rules",
"gaz. ext.": "gazette extraordinary",
"gaz.": "gazette",
"globe": "global learning and observation to benefit the environment",
"goi": "government of india",
"gov't": "government",
"govt.": "government",
"govts.": "governments",
"gpa": "general power of attorney",
"gpc": "general purposes committee",
"gvr": "grant, vacate, and remand",
"h.b.": "handbook for members",
"h.c.": "high court",
"h.c.p.": "habeas corpus petition",
"h.i.p.a.a.": "health insurance portability and accountability act",
"h.n.l.u.": "hidayatullah national law university",
"h.o.p.": "house of the people (lok sabha)",
"h.o.u.s.e.": "rajya sabha",
"h.o.u.s.e.s.": "rajya sabha and lok sabha",
"h.p. d.e.b.": "house of the people debates",
"h.p. deb.": "house of the people debates",
"h.r.a.": "human rights act",
"h.u.d.": "department of housing and urban development",
"h.u.f.": "hindu undivided family",
"hansard": "house of commons debates",
"hb": "handbook for members",
"hc.": "high court",
"hc": "hypothetical client",
"hdc": "holder in due course",
"hipaa": "health insurance portability and accountability act",
"hist.": "history",
"hnlu": "hidayatullah national law university",
"hop": "house of the people",
"hra": "human rights act",
"hrqs.": "headquarters",
"hud": "department of housing and urban development",
"huf": "hindu undivided family",
"hyd.":"hyderabad",
"i.b.": "intelligence bureau",
"i.b.i.d.": "ibidem (in the same place)",
"i.c.a.": "indian council of arbitration",
"i.c.a.d.r.": "international centre for alternative dispute resolution",
"i.c.c.": "international criminal court",
"i.c.c.p.r.": "international covenant on civil and political rights",
"i.c.j.": "international court of justice",
"i.c.r.c.": "international committee of the red cross",
"i.c.t.r.": "international criminal tribunal for rwanda",
"i.c.t.y.": "international criminal tribunal for the former yugoslavia",
"i.d.f.c.": "infrastructure development finance company",
"i.d.r.a.": "insurance regulatory and development authority",
"i.f.c.": "international financial corporation",
"i.g.o.": "intergovernmental organization",
"i.h.l.": "international humanitarian law",
"i.l.o.": "international labour organisation",
"i.l.r.": "indian law reports",
"i.m.f.": "international monetary fund",
"i.n r.e.": "in the matter of",
"i.n.c.": "indian national congress",
"i.n.d.": "independent",
"i.n.f.r.a.": "below",
"i.n.l.d.": "indian national lok dal",
"i.n.t.e.r.p.o.l.": "international police",
"i.o.": "investigation officer",
"i.o.b.": "indian overseas bank",
"i.o.l.t.a.": "interest on lawyer trust accounts",
"i.p.c.": "indian penal code",
"i.p.o.": "initial public offering",
"i.p.r.s.": "intellectual property rights",
"i.r.d.a.": "insurance regulatory and development authority",
"i.r.s.": "internal revenue service",
"i.s.r.o.": "indian space research organisation",
"i.t.a.t.": "income tax appellate tribunal",
"i.t.l.o.s.": "international tribunal for the law of the sea",
"i.u.u.": "illegal, unreported, and unregulated fishing",
"ib": "intelligence bureau",
"ibid.": "ibidem (in the same place)",
"ibid.":"in the same place",
"ica": "indian council of arbitration",
"icadr": "international centre for alternative dispute resolution",
"icc": "international criminal court",
"iccpr": "international covenant on civil and political rights",
"icj": "international court of justice",
"icrc": "international committee of the red cross",
"ictr": "international criminal tribunal for rwanda",
"icty": "international criminal tribunal for the former yugoslavia",
"idfc": "infrastructure development finance company",
"idra": "insurance regulatory and development authority",
"ifc": "international financial corporation",
"igo": "intergovernmental organization",
"ihl": "international humanitarian law",
"ilo": "international labour organisation",
"ilr": "indian law reports",
"imf": "international monetary fund",
"in re": "in the matter of",
"inc": "indian national congress",
"ind.": "independent",
"indl.": "industrial",
"infr.":"infrastructure",
"infra": "below",
"inj.": "injury",
"inld": "indian national lok dal",
"inst.": "institute",
"interpol": "international police",
"iolta": "interest on lawyer trust accounts",
"ipc": "indian penal code",
"ipo": "initial public offering",
"iprs": "intellectual property rights",
"irc": "internal revenue code",
"irda": "insurance regulatory and development authority",
"irs": "internal revenue service",
"isro": "indian space research organisation",
"itat": "income tax appellate tribunal",
"itlos": "international tribunal for the law of the sea",
"iuu": "illegal, unreported, and unregulated fishing",
"j&k": "jammu and kashmir",
"j&knc": "jammu & kashmir national conference",
"j.&k.": "jammu and kashmir",
"j.&k.n.c.": "jammu & kashmir national conference",
"j.a.g.": "judge advocate general",
"j.c.o.p.": "joint committee on offices of profit",
"j.d.": "janata dal",
"j.d.(s.)": "janata dal (secular)",
"j.d.(u.)": "janata dal (united)",
"j.d.r.": "judicial dispute resolution",
"j.h.c." : "joint health council",
"j.j.": "juvenile justice",
"j.m.m.": "jharkhand mukti morcha",
"j.n.u.": "jawaharlal nehru university",
"j.p.c.": "joint parliamentary committee",
"ja": "appellate judge",
"jag": "judge advocate general",
"jcop": "joint committee on offices of profit",
"jd": "janata dal",
"jd(s)": "janata dal (secular)",
"jd(u)": "janata dal (united)",
"jdr": "judicial dispute resolution",
"jdx": "jurisdiction",
"jj": "juvenile justice",
"jmm": "jharkhand mukti morcha",
"jmol": "judgment as a matter of law",
"jnov": "judgment notwithstanding verdict",
"jnu": "jawaharlal nehru university",
"jour.": "journal",
"jpc": "joint parliamentary committee",
"jr.": "junior",
"ju": "disposed of by judge",
"just.": "justice",
"jx": "jurisdiction",
"k.c.(m.)": "kerala congress (m)",
"k.m.p.p.": "kisan mazdoor praja party",
"k.t.d.f.c.": "kerala transport development finance corporation limited",
"kaul & shakdher": "practice and procedure of parliament by m.n. kaul & s.l. shakdher, (6th edition, 2009)",
"kc(m)": "kerala congress (m)",
"kmpp": "kisan mazdoor praja party",
"l.a.c.": "legal aid clinic",
"l.d.": "learned (used to address lawyers)",
"l.d.c.": "law and development committee",
"l.ed": "lawyers' edition",
"l.ed.2d": "lawyers 2nd edition",
"l.i.c.": "life insurance corporation",
"l.i.m.": "land information memorandum",
"l.j.": "lord justice",
"l.l.b.": "bachelor of laws",
"l.l.l.p.": "limited liability limited partnership",
"l.l.m.": "master of laws",
"l.l.p.": "limited liability partnership",
"l.o.b.": "list of business",
"l.o.c. c.i.t.": "loco citato (at the place quoted)",
"l.o.i.":"letter of intent",
"l.p.": "limited partnership",
"l.p.a.": "letters patent appeal" ,
"l.p.o.": "legal process outsourcing",
"l.p.t.": "low power transmitter",
"l.s. b.n. (i)/(ii)": "lok sabha bulletin part i/ii",
"l.s. d.e.b.": "lok sabha debates",
"l.s.": "lok sabha",
"l.s.r.": "lok sabha rules",
"l.t.": "tieutenant",
"l/c": "letter of credit",
"lac": "legal aid clinic",
"lac.": "lakhs",
"lah.": "lahore",
"ld.": "learned (used to address lawyers)",
"lic": "life insurance corporation",
"liqn.": "liquidation",
"llb": "bachelor of laws",
"llm": "master of laws",
"llp": "limited liability partnership",
"lob": "list of business",
"loc. cit.": "loco citato (at the place quoted)",
"loi": "letter of intent",
"lpo": "legal process outsourcing",
"lpt": "low power transmitter",
"lrs": "legal representative",
"lrs.": "legal representative",
"ls bn. (i)/(ii)": "lok sabha bulletin part i/ii",
"ls deb.": "lok sabha debates",
"ls": "lok sabha",
"lsr": "lok sabha rules",
"ltd.": "limited (in the context of corporations)",
"ltd.": "limited",
"m&a": "mergers and acquisitions",
"m.&a.": "mergers and acquisitions",
"m.a.n.t.r.a.": "machine assisted translation tool",
"m.b.e.": "multistate bar examination",
"m.c.i.": "medical council of india",
"m.g.n.r.e.g.s.": "mahatma gandhi national rural employment guarantee scheme",
"m.i.g.a.": "multilateral investment guarantee agency",
"m.i.n.": "ministry",
"m.i.s.a.": "maintenance of internal security act",
"m.l.": "muslim league",
"m.l.a.": "member of legislative assembly",
"m.l.c.": "member of legislative council",
"m.n.c.": "multi national company",
"m.o.a.": "memorandum of association",
"m.o.j.": "ministry of justice",
"m.o.u.": "memorandum of understanding",
#"m.p.": "member of parliament",
#"m.p.": "miscellaneous petition",
"m.p.c": "model penal code",
"m.p.l.a.d.s.": "member of parliament local area development scheme",
#"m.p.p.": "manipur people's party",
#"m.r.c.a.": "malaysian rubber board",
"m.r.t.p." :"monopolies and restrictive trade practices",
"m.r.t.p.c.": "monopoly & restrictive trade practices commission",
"m.t.n.l.": "mahanagar telephone nigam limited",
"m.t.s.": "minutes of a meeting of a committee",
"m.v.a.": "motor vehicles act",
"macq.": "macqueen's report",
"mag.": "magazine",
"mantra": "machine assisted translation tool",
"mbe": "multistate bar examination",
"mc": "matrimonial causes" ,
"mcft.": "million cubic feet",
"mci": "medical council of india",
"mfg.": "manufacturing",
"mfr": "manufacturer",
"mfr.": "manufacturer",
"mgnregs": "mahatma gandhi national rural employment guarantee scheme",
"miga": "multilateral investment guarantee agency",
"mil": "motion in limine",
"min.": "ministry",
"misa": "maintenance of internal security act",
"ml": "muslim league",
"mla": "member of legislative assembly",
"mlc": "member of legislative council",
"mlr": "modern law review",
"mnc": "multi national company",
"moa": "memorandum of association",
"moj": "ministry of justice",
"mos.":"months",
"mou": "memorandum of understanding",
#"mp": "member of parliament",
"mplads": "member of parliament local area development scheme",
"mpp": "manipur people's party",
"mr": "postnominals of the master of the rolls",
"mrca": "malaysian rubber board",
"mrtpc": "monopoly & restrictive trade practices commission",
"msj": "motion for summary judgment",
"mst.": "mistress",
"mth.": "month",
"mtnl": "mahanagar telephone nigam limited",
"mtrs.": "meters",
"mts.": "minutes of a meeting of a committee",
"mva": "motor vehicles act",
"n.a.a.c.p.": "national association for the advancement of colored people",
"n.c.": "national conference",
"n.c.d.r.c.": "national consumer disputes redressal commission",
"n.c.l.a.t.": "national company law appellate tribunal",
"n.c.l.t.": "national company law tribunal",
"n.c.m.e.c.": "national center for missing and exploited children",
"n.c.r.w.c.": "national commission to review the working of the constitution",
"n.c.t.": "national capital territory",
"n.d.c.": "national development council",
"n.d.m.a.": "national disaster management authority",
"n.d.p.s.": "narcotic drugs & psychotropic substances",
"n.d.r.f.": "national disaster response force",
"n.e.": "north eastern reporter",
"n.e.2d": "north eastern reporter, 2nd series",
"n.e.f.a.": "north east frontier agency",
"n.e.t.a.": "national environment tribunal act",
"n.g.t.": "national green tribunal",
"n.h.r.c.": "national human rights commission",
"n.i.c.": "national informatics centre",
"n.j.a.": "national judicial academy",
"n.j.a.c.": "national judicial appointments commission",
"n.k.c.": "national knowledge commission",
"n.o.m.": "nominated/nomination",
"n.o.t.": "notification",
"n.o.t.a.": "none of the above (electoral polls)",
"n.r.e.g.": "national rural employment guarantee",
"n.r.i.": "non-resident indian",
"n.s.g.": "national security guard",
"n.w.r.c.": "national water resources council",
"n/k/a": "now known as",
"naacp": "national association for the advancement of colored people",
"nc": "national conference",
"ncdrc": "national consumer disputes redressal commission",
"nclat": "national company law appellate tribunal",
"nclt": "national company law tribunal",
"ncmec": "national center for missing and exploited children",
"ncrwc": "national commission to review the working of the constitution",
"nct": "national capital territory",
"ndc": "national development council",
"ndma": "national disaster management authority",
"ndps": "narcotic drugs & psychotropic substances",
"ndrf": "national disaster response force",
"nefa": "north east frontier agency",
"neta": "national environment tribunal act",
"ngo": "non government organization",
"ngt": "national green tribunal",
"nhrc": "national human rights commission",
"nic": "national informatics centre",
"nja": "national judicial academy",
"njac": "national judicial appointments commission",
"nkc": "national knowledge commission",
"no.": "number",
"no. ": "number",
"nom.": "nominated/nomination",
"nos.": "numbers",
"nos. ": "numbers",
"not.": "notification",
"nota": "none of the above (electoral polls)",
"nreg": "national rural employment guarantee",
"nri": "non-resident indian",
"nsg": "national security guard",
"nwrc": "national water resources council",
#"o": "on behalf of",
"o&m": "operation and maintenance",
"o&m.": "operation and maintenance",
"o.&m.": "organisation and management",
"o.a.g.": "office of the attorney general",
"o.a.s.": "organization of american states",
"o.c.l.d.": "organized crime and legal division",
"o.i.g.": "office of inspector general",
"o.m.": "office memorandum",
"o.o.o." :"out of office",
"o.p. c.i.t.": "opere citato (in the work cited)",
"o.p.m.": "office of personnel management",
"o.s.a.": "original side appeal" ,
"o.s.h.a.": "occupational safety and health administration",
"o/b/o": "on behalf of",
"oag": "office of the attorney general",
"oas": "organization of american states",
"occ": "occupation",
"occ.": "occupation",
"occ:": "occupation",
"occu.": "occupation",
"ocld": "organized crime and legal division",
"oig": "office of inspector general",
"om": "office memorandum",
"op. cit.": "opere citato (in the work cited)",
"opm": "office of personnel management",
"opp.": "opposite",
"opp'n": "opposition",
"org.": "organization",
"ors": "others",
"ors.": 'others',
"osha": "occupational safety and health administration",
"p&l": "profit and loss statement",
"p.": "page",
"p.&l.": "profit and loss statement",
"p.a.": "power of attorney",
"p.a.c.": "committee on public accounts",
"p.a.n.": "permanent account number (income-tax)",
"p.a.o.": "public affairs officer",
"p.a.q.": "provisionally admitted question",
"p.a.r.l. d.e.b.": "parliamentary debates",
"p.c.": "petitions committee",
"p.d.": "privileges digest, lok sabha secretariat",
"p.d.g.": "parliament duty group",
"p.d.p.": "peoples democratic party",
"p.e.p.s.u.": "patiala and east punjab states union",
"p.f.a.": "protection from abuse",
"p.f.r.d.": "pension fund regulatory and development authority",
"p.f.r.d.a.": "pension fund regulatory and development authority",
"p.i.": "personal injury",
"p.i.a.c.": "personal injury and accident claims",
"p.i.b.": "press information bureau",
"p.i.i.": "personally identifiable information",
"p.i.l.": "public interest litigation",
"p.m.k.": "pattali makkal katchi",
"p.o.a.": "power of attorney",
"p.o.c.s.o.": "protection of children from sexual offenses act",
"p.o.t.a.": "prevention of terrorism act",
"p.p.": "public prosecutor",
"p.p.e.": "personal protective equipment",
"p.p.s.": "private parliamentary secretary",
"p.r.s.": "panchayat raj system",
"p.r.t.": "para-rubber tree",
#"p.s.p.": "praja socialist party",
"p.s.u.": "public sector undertaking",
"p.t.i.": "press trust of india",
"p.t.o.": "patent and trademark office",
"p.u.c.": "public undertakings committee",
"p.v.c.": "p. v. chidambaram",
"p.w.": "prosecution witnesses",
"pa": "professional association",
"pa.": "power of attorney",
"pac": "committee on public accounts",
"pan": "permanent account number (income-tax)",
"pao": "public affairs officer",
"paq": "provisionally admitted question",
"parl. deb.": "parliamentary debates",
"pc": "petitions committee",
"pd": "privileges digest, lok sabha secretariat",
"pdg": "parliament duty group",
"pdp": "peoples democratic party",
"pepsu": "patiala and east punjab states union",
"petro.": "petroleum",
"pfa": "protection from abuse",
"pfrda": "pension fund regulatory and development authority",
"ph": "prentice hall weekly legal service",
"ph.": "phone",
"piac": "personal injury and accident claims",
"pii": "personally identifiable information",
"pil": "public interest litigation",
"pkt.": "packet",
"pl": "public law",
"plff": "plaintiff",
"plff.": "plaintiff",
"plffs": "plaintiffs",
"plffs.": "plaintiffs",
"pllc": "professional limited liability company",
"plntf.": "plaintiff",
"pmk": "pattali makkal katchi",
"po.": "police",
"poa": "power of attorney",
"pocso": "protection of children from sexual offenses act",
"pota": "prevention of terrorism act",
"pp": "public prosecutor",
"pp.": "pages",
"ppe": "personal protective equipment",
"prae.": "praenotanda",
"prev.": "previous",
"pri.": "principal",
"prl.": "principal",
"proj.": "project",
"prvt.": "private",
"pslv": "polar satellite launch vehicle",
"psp": "praja socialist party",
"psus": "public sector undertakings",
"pt.": "part",
"pte": "private",
"pti": "press trust of india",
"pto": "patent and trademark office",
"pts.": "parts",
"pty": "proprietary company",
"pty.": "proprietary",
"ptyl": "proprietary",
"ptyl.": "proprietary",
"pub.l.": "public law",
"punj.": "punjab",
"pvt.": "private",
"pw":"prosecution witnesses",
"pwd.": "public works department",
"pwp": "peasants and workers party",
"pws.":"prosecution witnesses",
"q.b.d.": "queen's bench division",
"q.c.": "queen's counsel",
"q.d.r.o.": "qualified domestic relations order",
"qbd": "queen's bench division",
"qc": "queen's counsel",
"qdro": "qualified domestic relations order",
"qty":"quantity",
"qty.":"quantity",
"r.b.i.": "reserve bank of india",
"r.c.p.": "referred case petition" ,
"r.d.s.o.": "research designs and standards organisation",
"r.e.": "real estate",
"r.e.p.o.r.t.": "report",
"r.e.r.a.": "real estate regulatory authority",
"r.g.p.v.": "rajiv gandhi proudyogiki vishwavidyalaya",
"r.i.c.a.": "racketeer influenced and corrupt organizations act",
"r.i.c.o.": "racketeer influenced and corrupt organizations act",
"r.j.d.": "rashtriya janata dal",
"r.l.d.": "rashtriya lok dal",
"r.o.f.r.": "right of first refusal",
"r.o.t.c.": "reserve officers' training corps",
"r.p.c.": "ruling by present court",
"r.s. b.n. (i)/(ii)": "rajya sabha bulletin part i/ii",
"r.s. d.e.b.": "rajya sabha debates",
#"r.s.": "rajya sabhaaaaaaaa",
#"r.s.p.": "revolutionary socialist party",
"r.t.":"referred trial",
"r.t.i.": "right to information",
"rbi": "reserve bank of india",
"re.": "regarding",
"ref.": "reference",
"regn": "regulation",
"regn.": "regulation",
"regr.": "registrar",
"regt": "registration",
"regt.": "registration",
"reh'g": "rehearing",
"relv.": "relevant",
"rep.":"repealed",
"rera": "real estate regulatory authority",
"resp't": "respondent",
"retd.": "retired",
"rev. appl.": "review application",
"rev. authority. ": "revenue authorities",
"rev'd": "reversed",
"rica": "racketeer influenced and corrupt organizations act",
"rico": "racketeer influenced and corrupt organizations act",
"rj": "restorative justice",
"rjd": "rashtriya janata dal",
"rofr": "right of first refusal",
"roi": "return on investment",
"rotc": "reserve officers' training corps",
"rp act": "representation of the people act 1950 or 1951, as the case may be",
"rp": "republican party",
"rpc": "ruling by present court",
"rpi": "republican party of india",
"rpi(a)": "republican party of india (athawale)",
"rpt.": "report",
"rs deb.": "rajya sabha debates",
"rs": "rupees",
"rsa.": "regular second apeal",
"rsp": "revolutionary socialist party",
"rti act": "right to information act",
"rti": "right to information",
#"s": "section",
"s.ct.": "supreme court reports",
#"s.": "section",
"ss.": "sections(s)",
"s.a.": "second appeal",
"s.a.d.": "shiromani akali dal",
"s.a.d.(m.)": "shiromani akali dal (mann)",
"s.a.h.r.": "south asian human rights",
"s.a.i.l.": "steel authority of india limited",
"s.a.t.": "securities appellate tribunal",
"s.c.b.a.": "supreme court bar association",
"s.c.c.": "supreme court cases",
"s.c.j.": "supreme court journal",
"s.c.o.t.u.s.": "supreme court of the united states",
"s.c.r.": "supreme court reports",
"s.d.m.a.": "state disaster management authority",
"s.e.b.c.": "socially and economically backward classes",
"s.e.b.i.": "securities and exchange board of india",
"s.e.c.": "securities and exchange commission",
"s.e.c.s.": "sections",
"s.e.r.": "statutory explanatory rules",
"s.e.z.": "special economic zone",
"s.f.s.": "samajwadi forward bloc",
"s.i.c.": "state information commission",
"s.i.c.a.": "sick industrial companies act",
"s.i.t.": "special investigation team",
"s.k.d.l.f.": "sikkim sangram parishad",
"s.l.a.p.p.": "strategic lawsuit against public participation",
"s.l.c.":"state level committee",
"s.l.p.": "special leave petition",
"s.o.": "stand over",
"s.o.p.o.": "sexual offences prevention order",
"s.o.x.": "sarbanes-oxley act",
#"s.p.": "samajwadi party",
"s.r.": "short recidivism",
"s.r.a.": "solicitors regulation authority",
"s.r.o.": "sub-registrar office",
"s.s.a.": "social security administration",
"s.t.a.": "special tribunal appeal" ,
"s.t.p.": "special tribunal petition",
"s.t.v.": "state transport vehicle",
"s.u.p.r.a.": "above",
"s.v.e.e.p.": "systematic voters' education and electoral participation",
"s/j": "summary judgment",
"sad": "shiromani akali dal",
"sad(m)": "shiromani akali dal (mann)",
"sahr": "south asian human rights",
"sail": "steel authority of india limited",
"sat": "securities appellate tribunal",
"scba": "supreme court bar association",
"scc": "supreme court cases",
"schs.":"schedules",
"scj": "supreme court journal",
"scotus": "supreme court of the united states",
"sd": "said",
"sdma": "state disaster management authority",
"sebi": "securities and exchange board of india",
"sec": "securities and exchange commission",
"sec.": "section",
"secs.": "sections",
"secr.": "secretary",
"secy.": "secretary",
"ser": "statutory explanatory rules",
"ser.": "series",
"ser.": "service",
"sez": "special economic zone",
"sfs": "samajwadi forward bloc",
"sft.": "square feet",
"si": "statutory instruments",
"sic": "state information commission",
"sica": "sick industrial companies act",
"sig.": "signature",
"sit": "special investigation team",
"skdlf": "sikkim sangram parishad",
"slapp": "strategic lawsuit against public participation",
"slp": "special leave petition",
"sm.": "shrimati",
"smj": "subject-matter jurisdiction",
"smt.": "shrimati",
"sopo": "sexual offences prevention order",
"sox": "sarbanes-oxley act",
"sp": "samajwadi party",
"spg.": "spinning",
"spl.": "special",
"sq.": "square",
"sr.": "senior",
"sra": "solicitors regulation authority",
"srl.": "serial",
"sro": "sub-registrar office",
"ssa": "social security administration",
"stn": "station",
"stn.": "station",
"stns": "stations",
"stns.": "stations",
"stv": "state transport vehicle",
"sub-s.": "sub section",
"supdt.": "superintendent",
"supdts.": "superintendents",
"supp.": "supplement",
"suppl.": "supplement",
"supra": "above",
"sveep": "systematic voters' education and electoral participation",
"syp": "syrup",
"syp.": "syrup",
"t.a.p.": "technically accepted paper",
"t.c.": "trinamool congress",
"t.c.a.": "tax case appeal",
"t.c.r.": "tax case revision",
"t.d.p.": "telugu desam party",
"t.d.r.": "transferable development rights",
"t.d.s.": "tax deducted at source",
"t.i.n.": "taxpayer identification number",
"t.j.c.": "temporary juvenile criminal",
"t.m.a.": "trade marks appeal",
"t.m.c.": "trinamool congress",
"t.m.s.a.": "trade marks second appeal" ,
"t.r.a.i.": "telecom regulatory authority of india",
"t.r.f.": "transfer",
"t.r.i.p.s.": "trade-related aspects of intellectual property rights",
"t.r.o.": "temporary restraining order",
"t.r.p.": "tax return preparer",
"t.s.": "temporary suspension",
"t.s.r.": "territorial special report",
"t.t.v.": "temporary television",
"tap": "technically accepted paper",
"tc": "trinamool congress",
"tdp": "telugu desam party",
"tds": "tax deducted at source",
"thro.": "through",
"tin": "taxpayer identification number",
"tjc": "temporary juvenile criminal",
"tmc": "trinamool congress",
"tmt.":"thirumathi",
"trai": "telecom regulatory authority of india",
"trips": "trade-related aspects of intellectual property rights",
"tro": "temporary restraining order",
"trp": "tax return preparer",
"ts": "temporary suspension",
"tsr": "territorial special report",
"ttv": "temporary television",
"u.c.c.": "uniform commercial code",
"f.a.t.": "tender of first appeal",
"f.a": "first appeal",
"u.c.c.j.e.a.": "uniform child custody jurisdiction and enforcement act",
"u.d.p.f.": "united democratic front",
"u.l.f.a.": "united liberation front of assam",
"u.n.c.e.d.": "united nations conference on environment and development",
"u.n.c.i.t.r.a.l.": "united nations commission on international trade law",
"u.n.h.r.c.": "united nations human rights council",
"u.n.s.c.": "united nations security council",
"u.p.s.r.t.c.": "uttar pradesh state road transport corporation",
"u.s.": "united states",
"u.s.c.": "united states code",
"u.s.c.t.": "united states court of tax appeals",
"u.s.t.p.o.": "united states patent and trademark office",
"ucc": "uniform commercial code",
"uccjea": "uniform child custody jurisdiction and enforcement act",
"ucmj": "uniform code of military justice",
"ud": "unnatural death",
"udpf": "united democratic front",
"ulfa": "united liberation front of assam",
"unced": "united nations conference on environment and development",
"uncitral": "united nations commission on international trade law",
"unhrc": "united nations human rights council",
"unsc": "united nations security council",
"upc": "uniform probate code",
"upsrtc": "uttar pradesh state road transport corporation",
"us": "under secretary",
"usc": "united states code",
"usct": "united states court of tax appeals",
"ustpo": "united states patent and trademark office",
"v.": "versus",
"v.a.c.": "veterans affairs canada",
"v.a.w.a.": "violence against women act",
"v.c.": "vice-chancellor",
"v.d.": "voter database",
"v.i.d.e.": "see",
"v.i.p.": "very important person",
"v.m.a.": "vehicle maintenance agreement",
"v.o.c.": "victim of crime",
"v.v.p.a.a.": "victims' rights and victim protection act",
"v.v.p.a.t.": "voter verifiable paper audit trail",
"vac": "veterans affairs canada",
"vawa": "violence against women act",
"vc": "vice-chancellor",
"vd": "voter database",
"vict.": "victoria",
#"vide": "see",
"vip": "very important person",
"viz.": "videlicet",
"voc": "victim of crime",
"vol.": "volume",
"vols.": "volumes",
"vs": "versus",
"vs.": "versus",
"vvpaa": "victims' rights and victim protection act",
"vvpat": "voter verifiable paper audit trail",
"w.a.": "writ appeal",
"w.b.": "west bengal",
"w.e.f.": "with effect from",
"w.o.":"work order",
"w.p.": "writ petition",
"w.r.d.": "water resources department",
"w.t.o.": "world trade organization",
"wb": "west bengal",
"wef": "with effect from",
"wg.": "wing",
"wop": "without prejudice",
"wp": "writ petition",
"wrd": "water resources department",
"wto": "world trade organization",
"wvg.": "weaving",
"x": "examination",
"xfd": "examination for discovery",
"xn": "examination in chief",
"xxn": "cross-examination",
"y.b.": "year book",
"y.d.a.": "youth development agency",
"y.l.d.": "young lawyers division",
"yb": "year book",
"yd.": "yard",
"yda": "youth development agency",
"yds.": "yards",
"yld": "young lawyers division",
"yrs": "years",
"z.b.i.": "zero base investment",
"zbi": "zero base investment",
"π": "plaintiff",
"ain't": "are not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have",
"Sr. No.": "Serial Number",
"&": "and",
"@": "at",
"%": "percent",
"$": "dollar",
"₹": "rupee",
"£": "pound","€": "euro",
"°": "degree",
"±": "plus-minus",
"lpa": "Letters Patent Appeal",
"aft": "armed forces tribunal",
"aft.": "armed forces tribunal",
"afcat": "armed forces court of appeal tribunal",

"arb.a.": "arbitration appeal",
"arb.p.": "arbitration petition",
"arb.op.": "arbitration original petition",
"arb. appl.": "arbitration application",

"bat": "banking appellate tribunal",
"boi": "bank of india",
"sbi": "state bank of india",
"pnb": "punjab national bank",
"rbl": "ratnakar bank limited",
"icici": "industrial credit and investment corporation of india",
"hdfc": "housing development finance corporation",

"cat": "central administrative tribunal",
"cat.": "central administrative tribunal",
"cestat": "customs excise and service tax appellate tribunal",
"sat.": "securities appellate tribunal",
"drat": "debt recovery appellate tribunal",
"drt-i": "debt recovery tribunal one",
"drt-ii": "debt recovery tribunal two",

"nia": "national investigation agency",
"ncb": "narcotics control bureau",
"sfio": "serious fraud investigation office",
"ed": "enforcement directorate",
"cvc.": "central vigilance commission",
"lokpal": "anti corruption ombudsman",

"cav": "case reserved for judgment",
"cav.": "case reserved for judgment",
"dictated": "dictated in open court",
"pron.": "pronounced",
"pronounced on": "judgment pronounced on",
"resvd.": "reserved",
"resvd": "reserved",

"ia": "interlocutory application",
"i.a.": "interlocutory application",
"ias": "interlocutory applications",
"cma": "civil miscellaneous application",
"cm": "civil miscellaneous",
"cmp": "civil miscellaneous petition",
"crmp": "criminal miscellaneous petition",
"misc. appl.": "miscellaneous application",

"oa": "original application",
"o.a.": "original application",
"ra": "review application",
"r.a.": "review application",
"ma": "miscellaneous application",
# Skip "m.a." - may be Master of Arts (academic degree) not miscellaneous application
"ta": "transfer application",
"t.a.": "transfer application",



"fao": "first appeal from order",
"fao.": "first appeal from order",
"rfa": "regular first appeal",
"r.f.a.": "regular first appeal",
"rsa": "regular second appeal",
"r.s.a.": "regular second appeal",
"sao": "second appeal from order",
"cao": "civil appeal order",

"cra": "criminal appeal",
"cr.a.": "criminal appeal",
"cr.rev.": "criminal revision",
"crl.appeal": "criminal appeal",
"crl.rev.": "criminal revision",

"wa": "writ appeal",
"w.a": "writ appeal",
"wp(c)": "writ petition civil",
"wp(crl)": "writ petition criminal",
"wpc": "writ petition civil",
"w.p.(c)": "writ petition civil",
"w.p.(crl.)": "writ petition criminal",

"sla": "special leave appeal",
"sla.": "special leave appeal",
"sca": "special civil application",
"sca.": "special civil application",

"lp": "letters patent",
"lpa.": "letters patent appeal",
"lpa no.": "letters patent appeal number",

"db": "division bench",
"sb": "single bench",
"fb": "full bench",
"cb": "constitution bench",

"cjm.": "chief judicial magistrate",
"jmfc": "judicial magistrate first class",
"jm": "judicial magistrate",
"mm": "metropolitan magistrate",
"acmm": "additional chief metropolitan magistrate",
"cmm": "chief metropolitan magistrate",

"adj": "additional district judge",
"dj": "district judge",
"pdj": "principal district judge",
"ld. counsel": "learned counsel",
"ld. adv.": "learned advocate",

"u/s": "under section",
"u.s.": "under section",
"r/w": "read with",
"rw": "read with",
"sub-sec.": "sub section",
"prov.": "proviso",
"expl.": "explanation",
"sched.": "schedule",
"sch.": "schedule",

"or.": "order",
"ord.": "order",
"ordr.": "order",
"decr": "decree",
"decr.": "decree",
"judgt.": "judgment",

"pltf": "plaintiff",
"pltff": "plaintiff",
"deft": "defendant",
"resp.": "respondent",
"app.": "appellant",
"petnr": "petitioner",
"petr": "petitioner",

"co-applicant": "co applicant",
"co-owner": "co owner",
"lr": "legal representative",
"lrs": "legal representatives",

"pocsa": "protection of children from sexual assault act",
"jj act": "juvenile justice act",
"dv act": "domestic violence act",
"ni act": "negotiable instruments act",
"tp act": "transfer of property act",
"evidence act": "indian evidence act",
"contract act": "indian contract act",
"companies act": "companies act",
"it act": "information technology act",

"hcj": "high court judge",
"scj.": "supreme court judge",
"cji.": "chief justice of india",

"doa": "date of appointment",
"dob": "date of birth",
"dod": "date of decision",
"doj.": "date of joining",
"dt.of ord.": "date of order",

"annx.": "annexure",
"annxr.": "annexure",
"ann.": "annexure",
"enc.": "enclosure",

"ibid": "same source as previous citation",
"id.": "same author immediately cited",
"suppl": "supplementary",
"corrig.": "corrigendum",
"sct": "Service Cases Today",
"SERVING/SERVLR": "Service Law Reporter (Reports cases related to government service and employment)",
"regd": "registered",
"regd.": "registered"
}

# --- Context-aware legal abbreviation resolution (100 ambiguous abbreviations) ---

contextual_abbreviations = {

    "SC": [
        {"full": "Scheduled Caste",
         "keywords": ["act", "reservation","quota","category","caste","certificate","student","community", "scheduled"]},
        {"full": "Supreme Court",
         "keywords": ["bench","judge","appeal","petition","judgment","order","hearing", "court", "scri"]}
    ],

    "SC/ST": [
        {"full": "Scheduled Caste/Scheduled Tribe",
         "keywords": ["prevention", "atrocities", "act", "special court", "district"]}
    ],
    "r.s.": [
        {"full": "Rajya Sabha", 
         "keywords": ["parliament","upper house","debates","bulletin"]},
        {"full": "Rupees", 
         "keywords": ["currency","amount","rs.","rupees"]} 
    ],
    "bc": [
        {"full": "Backward Class",
         "keywords": ["reservation","quota","category","caste","certificate","student","community"]},
        {"full": "Birth Certificate",
         "keywords": ["birth","certificate","issued","date","place"]}
    ],
    "hc": [
        {"full": "High Court",
         "keywords": ["court","judge","bench","petition","appeal","writ","order"]},
        {"full": "Handicapped",
         "keywords": ["disabled","person","benefit","medical","certificate","quota"]}
    ],

    "st": [
        {"full": "Scheduled Tribe",
         "keywords": ["tribe","reservation","community","certificate","quota","schedule","act"]},
        {"full": "Street",
         "keywords": ["road","address","lane","city","market"]}
    ],

    "ca": [
        {"full": "Civil Appeal",
         "keywords": ["court","appeal","bench","judgment","filed"]},
        {"full": "Chartered Accountant",
         "keywords": ["audit","tax","finance","accounts","firm"]}
    ],

    "oa": [
        {"full": "Original Application",
         "keywords": ["tribunal","service","petition","matter","filed"]},
        {"full": "Official Assignee",
         "keywords": ["insolvency","estate","bankruptcy","property"]}
    ],

    "ra": [
        {"full": "Review Application",
         "keywords": ["court","judgment","petition","filed","order"]},
        {"full": "Regular Appeal",
         "keywords": ["appeal","civil","district","decree"]}
    ],

    "rp": [
        {"full": "Review Petition",
         "keywords": ["supreme","court","judgment","filed"]},
        {"full": "Revision Petition",
         "keywords": ["revision","lower","court","challenge"]}
    ],

    "ma": [
        {"full": "Miscellaneous Application",
         "keywords": ["application","court","petition","interim","order"]},
        {"full": "Motor Accident",
         "keywords": ["vehicle","injury","claim","compensation","accident"]}
    ],

    "po": [
        {"full": "Proclaimed Offender",
         "keywords": ["accused","criminal","absconding","warrant","police"]},
        {"full": "Presiding Officer",
         "keywords": ["tribunal","court","authority","hearing","officer"]}
    ],

    "io": [
        {"full": "Investigating Officer",
         "keywords": ["police","charge","FIR","investigation","crime"]},
        {"full": "Income Officer",
         "keywords": ["tax","assessment","income","department"]}
    ],

    "co": [
        {"full": "Court Order",
         "keywords": ["judge","bench","issued","passed","court"]},
        {"full": "Circle Officer",
         "keywords": ["revenue","district","land","officer"]}
    ],

    "do.": [
        {"full": "Defence Officer",
         "keywords": ["army","military","defence","service"]},
        {"full": "District Officer",
         "keywords": ["district","administration","office","collector"]}
    ],

    "so.": [
        {"full": "Standing Order",
         "keywords": ["government","notification","policy","issued"]},
        {"full": "Sub Officer",
         "keywords": ["department","rank","staff"]}
    ],

    "ro": [
        {"full": "Returning Officer",
         "keywords": ["election","vote","poll","candidate"]},
        {"full": "Revenue Officer",
         "keywords": ["land","mutation","revenue","tehsil"]}
    ],

    "eo": [
        {"full": "Executive Officer",
         "keywords": ["municipal","board","office","authority"]},
        {"full": "Election Officer",
         "keywords": ["poll","vote","candidate","booth"]}
    ],

    "dm": [
        {"full": "District Magistrate",
         "keywords": ["district","administration","magistrate","order"]},
        {"full": "Direct Message",
         "keywords": ["social media","message","chat"]}
    ],

    "sdm": [
        {"full": "Sub Divisional Magistrate",
         "keywords": ["district","land","revenue","magistrate"]},
        {"full": "Senior Duty Manager",
         "keywords": ["railway","airport","manager"]}
    ],

    "sp": [
        {"full": "Superintendent of Police",
         "keywords": ["police","district","crime","officer"]},
        {"full": "Special Petition",
         "keywords": ["court","filed","appeal"]}
    ],

    "dsp": [
        {"full": "Deputy Superintendent of Police",
         "keywords": ["police","crime","district"]},
        {"full": "Digital Signal Processing",
         "keywords": ["engineering","signal","system"]}
    ],

    "dcp": [
        {"full": "Deputy Commissioner of Police",
         "keywords": ["police","zone","crime"]},
        {"full": "District Consumer Panel",
         "keywords": ["consumer","complaint"]}
    ],

    "ig": [
        {"full": "Inspector General",
         "keywords": ["police","department","rank"]},
        {"full": "Income Group",
         "keywords": ["economy","housing","scheme"]}
    ],

    "dig": [
        {"full": "Deputy Inspector General",
         "keywords": ["police","rank","department"]},
        {"full": "Digital India Group",
         "keywords": ["technology","initiative"]}
    ],

    "cji": [
        {"full": "Chief Justice of India",
         "keywords": ["supreme","court","bench","judge"]},
        {"full": "Central Judicial Institute",
         "keywords": ["training","academy"]}
    ],

    "cj": [
        {"full": "Chief Justice",
         "keywords": ["court","judge","bench"]},
        {"full": "Civil Judge",
         "keywords": ["trial","district","civil"]}
    ],

    "db": [
        {"full": "Division Bench",
         "keywords": ["court","judges","bench","appeal"]},
        {"full": "Double Bed",
         "keywords": ["hotel","room","furniture"]}
    ],

    "sb": [
        {"full": "Single Bench",
         "keywords": ["court","judge","bench"]},
        {"full": "Savings Bank",
         "keywords": ["account","bank","deposit"]}
    ],

    "fb": [
        {"full": "Full Bench",
         "keywords": ["court","judges","bench"]},
        {"full": "Facebook",
         "keywords": ["social media","post","online"]}
    ],

    "lb": [
        {"full": "Larger Bench",
         "keywords": ["court","reference","judges"]},
        {"full": "Pound",
         "keywords": ["weight","kg","measure"]}
    ],

    "cw": [
        {"full": "Civil Writ",
         "keywords": ["petition","high court","constitution"]},
        {"full": "Court Witness",
         "keywords": ["trial","evidence","witness"]}
    ],


    "sa": [
        {"full": "Second Appeal",
         "keywords": ["appeal","civil","court"]},
        {"full": "Special Audit",
         "keywords": ["tax","accounts","audit"]}
    ],

    "fa": [
        {"full": "First Appeal",
         "keywords": ["court","decree","appeal"]},
        {"full": "Financial Assistance",
         "keywords": ["grant","aid","fund"]}
    ],

    "ea": [
        {"full": "Execution Application",
         "keywords": ["decree","execution","court"]},
        {"full": "Environmental Assessment",
         "keywords": ["pollution","project","clearance"]}
    ],

    "ep": [
        {"full": "Execution Petition",
         "keywords": ["decree","execution","court"]},
        {"full": "Election Petition",
         "keywords": ["vote","candidate","election"]}
    ],

    "c.a.t": [
        {"full": "Central Administrative Tribunal",
         "keywords": ["service","employee","tribunal","government"]},
        {"full": "Cat",
         "keywords": ["animal","pet","tail"]}
    ],

    "s.a.t": [
        {"full": "Securities Appellate Tribunal",
         "keywords": ["sebi","market","appeal","tribunal"]},
        {"full": "Scholastic Aptitude Test",
         "keywords": ["exam","student","admission"]}
    ],

    "a.f.t": [
        {"full": "Armed Forces Tribunal",
         "keywords": ["army","service","military","tribunal"]},
        {"full": "After",
         "keywords": ["time","later"]}
    ],

    "d.r.t": [
        {"full": "Debt Recovery Tribunal",
         "keywords": ["bank","loan","recovery","tribunal"]},
        {"full": "Daily Routine Task",
         "keywords": ["task","daily"]}
    ],

    "m.a.c.t": [
        {"full": "Motor Accident Claims Tribunal",
         "keywords": ["vehicle","injury","compensation","claim"]},
        {"full": "Management Committee",
         "keywords": ["committee","meeting"]}
    ],

    "RERA": [
        {"full": "Real Estate Regulatory Authority",
         "keywords": ["builder","flat","project","property"]},
        {"full": "Rare Earth Research Agency",
         "keywords": ["mineral","research"]}
    ],

    "ipc": [
        {"full": "Indian Penal Code",
         "keywords": ["section","crime","offence","punishment"]},
        {"full": "Inter Process Communication",
         "keywords": ["computer","system","software"]}
    ],


    "COI": [
        {"full": "Constitution of India",
         "keywords": ["article","fundamental","rights"]},
        {"full": "Certificate of Insurance",
         "keywords": ["vehicle","policy"]}
    ],

    "mp": [
        {"full": "Member of Parliament",
         "keywords": ["lok sabha","rajya sabha","election"]},
        {"full": "Madhya Pradesh",
         "keywords": ["state","bhopal","indore"]}
    ],

    "NDPS": [
        {"full": "Narcotic Drugs and Psychotropic Substances Act",
         "keywords": ["drug","contraband","seizure","bail"]},
        {"full": "National Data Protection Scheme",
         "keywords": ["data","privacy"]}
    ],

    "UAPA": [
        {"full": "Unlawful Activities Prevention Act",
         "keywords": ["terror","security","bail"]},
        {"full": "Urban Area Planning Authority",
         "keywords": ["city","planning"]}
    ],



}


def get_context_window(text: str, start: int, end: int, window_chars: int = 250) -> str:
    left = max(0, start - window_chars)
    right = min(len(text), end + window_chars)
    return text[left:right]


def choose_best_abbreviation_expansion(context_text: str, candidates: list[dict], original_abbr: str = str(None)) -> str:
    context_lower = context_text.lower()
    best_candidate = None
    best_score = -1

    # Citation indicators that strongly suggest "Supreme Court"
    citation_indicators = ["aironline", "scc", "scale", "sct", "adj", "servlj", "servlr", "esc", "lab ln", "scri"]
    
    for candidate in candidates:
        score = sum(
            1 for kw in candidate.get("keywords", []) if kw.lower() in context_lower
        )
        
        # Boost score for Supreme Court when citation patterns found
        if candidate.get("full") == "Supreme Court":
            for indicator in citation_indicators:
                if indicator in context_lower:
                    score += 3  # Strong boost for citation context
        
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is None:
        if original_abbr and len(candidates) > 1:
            return original_abbr
        return candidates[0]["full"]

    if best_score <= 0 and len(candidates) > 1:
        # Ambiguous when no keywords match; preserve the abbreviation.
        return original_abbr or candidates[0]["full"]

    return best_candidate["full"]

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxx"
# NOTE: Do not attempt any Hugging Face login at import time. The app
# will perform `login()` only when the user explicitly provides a token
# via the sidebar and clicks 'Load models'. This prevents automatic
# connections using environment variables.

# --- Utility functions ---

def extract_text_from_pdf_filelike(filelike):
    try:
        reader = PyPDF2.PdfReader(filelike)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        st.error(f"Failed to extract PDF text: {e}")
        return ""
PARTY_MARKERS = [
    "APPELLANT(S)",
    "RESPONDENT(S)",
    "PETITIONER(S)",
    "CLAIMANT(S)",
    "APPELLANTS",
    "RESPONDENTS",
    "PETITIONERS",
    "CLAIMANTS"
]

def protect_parentheses(text):
    """
    Smart parenthesis protection for Indian legal judgments.

    Protect:
    - Party labels: (APPELLANT(S)), (RESPONDENT(S))
    - Section references: (1), (2), (r), (s), (va)
    - Case numbering blocks: (Crl.), (Diary No. 1234) optional if complex
    - Citations / years: (2025), (1989)
    - Nested factual notes

    Do NOT protect:
    - Legal abbreviations needing expansion:
      (CRL.), (C), (CIVIL), (CRIMINAL), (SLP), (SC/ST)
    """

    protected = {}
    counter = 0

    # abbreviations that must remain visible for expansion
    dont_protect = {
        "CRL", "CRL.", "CRL.A.", "C", "C.", "CIVIL",
        "CRIMINAL", "SLP", "SLP.", "SC", "ST", "SC/ST",
        "WP", "W.P.", "LPA", "RSA", "FAO"
    }

    def replacer(match):
        nonlocal counter

        value = match.group(0)          # full "(...)"
        inner = value[1:-1].strip()     # content only

        inner_upper = inner.upper()

        # -------------------------------------------------
        # 1. DO NOT PROTECT abbreviations needing expansion
        # -------------------------------------------------
        if inner_upper in dont_protect:
            return value

        # -------------------------------------------------
        # 2. DO NOT PROTECT if contains CRL spaced PDF noise
        # e.g. (C RL.), (C R L)
        # -------------------------------------------------
        if re.fullmatch(r'C\s*R\s*L\.?', inner, re.I):
            return value

        # -------------------------------------------------
        # 3. PROTECT party markers
        # -------------------------------------------------
        for marker in PARTY_MARKERS:
            if marker.upper() in inner_upper:
                key = f"__PAREN_{counter}__"
                protected[key] = value
                counter += 1
                return key

        # -------------------------------------------------
        # 4. PROTECT section clauses
        # (1), (2), (r), (s), (va), (ii), (a)
        # -------------------------------------------------
        if re.fullmatch(r'[0-9A-Za-zivxIVX\-]+', inner):
            key = f"__PAREN_{counter}__"
            protected[key] = value
            counter += 1
            return key

        # -------------------------------------------------
        # 5. PROTECT years
        # -------------------------------------------------
        if re.fullmatch(r'(18|19|20)\d{2}', inner):
            key = f"__PAREN_{counter}__"
            protected[key] = value
            counter += 1
            return key

        # -------------------------------------------------
        # 6. PROTECT long factual parenthesis notes
        # -------------------------------------------------
        if len(inner.split()) >= 3:
            key = f"__PAREN_{counter}__"
            protected[key] = value
            counter += 1
            return key

        # -------------------------------------------------
        # default: leave visible
        # -------------------------------------------------
        return value

    text = re.sub(r'\([^()]*\)', replacer, text)
    return text, protected


def restore_parentheses(text, protected):
    for key, val in protected.items():
        text = text.replace(key, val)
    return text


def expand_abbreviations_safe(text: str, abbrev_dict: dict) -> str:
    
    # Protect parenthesized content
    text, protected = protect_parentheses(text)

    for abbr, full in sorted(abbrev_dict.items(), key=lambda x: len(x[0]), reverse=True):
        # ❌ Skip single-character abbreviations like §
        if len(abbr.strip()) <= 1:
            continue

        candidates = contextual_abbreviations.get(abbr.upper())
        pattern = r'(?<!\w)' + re.escape(abbr) + r'(?!\w)'

        if candidates:
            from typing import cast
            candidates_list = cast(list[dict], candidates)

            def replace_with_context(match):
                context = get_context_window(text, match.start(), match.end())
                return choose_best_abbreviation_expansion(context, candidates_list, match.group(0))

            text = re.sub(pattern, replace_with_context, text, flags=re.IGNORECASE)
        else:
            text = re.sub(pattern, full, text, flags=re.IGNORECASE)

    # Restore parentheses
    text = restore_parentheses(text, protected)

    # Second pass: expand abbreviations that are present only in the
    # contextual_abbreviations dictionary (e.g. `SC`, `ST`, `IPC`) so they
    # are resolved using surrounding context when not present in
    # `abbrev_dict`.
    def expand_contextual_abbreviations(text_in: str) -> str:
        # Collect all matches on the original text first to avoid cascading
        # replacements (e.g. replacing `SC` then `ST` which creates new tokens).
        original = text_in
        replacements = []  # list of (start, end, replacement)

        # First, handle multi-part abbreviations like SC/ST before tokenizing
        # Sort by length descending to match longer patterns first
        multi_part_abbrevs = [abbr for abbr in contextual_abbreviations.keys() if '/' in abbr]
        for abbr in sorted(multi_part_abbrevs, key=len, reverse=True):
            candidates = contextual_abbreviations.get(abbr, [])
            if not candidates:
                continue
            pattern = r'(?<!\w)' + re.escape(abbr) + r'(?!\w)'
            for match in re.finditer(pattern, text_in, re.IGNORECASE):
                context = get_context_window(text_in, match.start(), match.end())
                rep = choose_best_abbreviation_expansion(context, candidates, match.group(0))
                replacements.append((match.start(), match.end(), rep))

        if not replacements:
            return text_in

        # Apply replacements from right->left so indices don't shift.
        replacements.sort(key=lambda x: x[0], reverse=True)
        out = list(text_in)
        for start, end, rep in replacements:
            out[start:end] = rep
        text_in = ''.join(out)

        # Then tokenize remaining text for single-token abbreviations
        original = text_in
        replacements = []
        parts = re.split(r'(\W+)', original)
        pos = 0
        for part in parts:
            if not part:
                continue
            if re.fullmatch(r'\W+', part):
                pos += len(part)
                continue
            token = part
            for abbr, candidates in contextual_abbreviations.items():
                if not abbr or len(abbr.strip()) <= 1 or '/' in abbr:
                    continue  # Skip multi-part abbreviations (already handled)
                if token.lower() == abbr.lower():
                    start = pos
                    end = pos + len(part)
                    context = get_context_window(original, start, end)
                    rep = choose_best_abbreviation_expansion(context, candidates, part)
                    replacements.append((start, end, rep))
            pos += len(part)

        if not replacements:
            return text_in

        # Apply replacements from right->left so indices don't shift.
        replacements.sort(key=lambda x: x[0], reverse=True)
        out = list(text_in)
        for start, end, rep in replacements:
            out[start:end] = rep
        return ''.join(out)

    text = expand_contextual_abbreviations(text)
    return text

    

def preprocess_text(text: str) -> str:
    # A cleaned-up adaptation of the user's preprocessing
    if not text:
        return ""
    #text = re.sub(r"\xa0", " ", text)
    # Normalize
    text = text.strip()

    # Remove PDF headnote content when present, keeping the judgment section
    text = re.sub(
        r'headnote:(.*?)(judgment:|judgement:)',
        lambda m: m.group(2),
        text,
        flags=re.I | re.DOTALL,
    )

    #lines = [re.sub(r'[^a-zA-Z0-9.,)\-(/?\t ]', '', l) for l in text.splitlines()] remove the legal structure of document
    # KEEP legal punctuation
    lines = [re.sub(r'[^\w\s\.,:\-()/\'@#&]', '', l)
    for l in text.splitlines()  
    ]
    
    #lines = [re.sub(r'(?<=[^0-9])/(?=[^0-9])', ' ', l) for l in lines]
    lines = [re.sub(r"\t+", " ", l) for l in lines]
    lines = [re.sub(r" +", " ", l) for l in lines]
    lines = [re.sub(r"\.{2,}", "", l) for l in lines]
    lines = [re.sub(r"^ ?", "", l) for l in lines]
    lines = [l for l in lines if (len(l) != 1 and not re.fullmatch(r"(\d|\d\d|\d\d\d)", l))]
    # Keep numbered legal lines like '1. Facts of the case' and '2 The appellant...'
    #lines = [l for l in lines if not re.fullmatch(r'\d+\s+.+', l.strip())]
    

    # Remove lines referencing Indian Kanoon (user-specific)
    lines = [l for l in lines if not re.search(r"Indian Kanoon", l, re.I)]
    
    lines = [l for l in lines if not re.search(r"CRIMINAL APPEAL @", l, re.I)]
    text = "\n".join(lines)
    #text = re.sub(r"[()\[\]\"]", " ", text)
    # Preserve legal case number patterns like "NO. 1234" or "No. 1234"
    # Only replace "no." when followed by a letter (like "no.d")
   
    text = re.sub(r" nos\.", " numbers", text)
    
    text = re.sub(r" nos\.", " numbers", text)
    text = re.sub(r" co\.", " company", text)
    text = re.sub(r" ltd\.", " limited", text)
    text = re.sub(r'\bS\.\s*(\d+)', r'Section \1', text)
    text = re.sub(r'\bSec\.\s*(\d+)', r'Section \1', text)
    text = re.sub(r'(?<!SC)/(?!ST)', ' ', text)
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.I)
    text = re.sub(r'J\s*U\s*D\s*G\s*M\s*E\s*N\s*T', 'JUDGMENT', text, flags=re.I)
    text = re.sub(r'O\s*R\s*D\s*E\s*R', 'ORDER', text, flags=re.I)
    

    # Protect names like "Abhay S. Oka"
    text = re.sub(
        r'([A-Z][a-z]+)\s+S\.\s+([A-Z][a-z]+)',
        r'\1 __MID__ \2',
        text
    )

    # Correct legal expansion
    

    # Normalize party labels
    text = re.sub(r'APPELLANT\(S\)', 'APPELLANTS', text, flags=re.I)
    text = re.sub(r'APPELLANT \(S\)', 'APPELLANTS', text, flags=re.I)
    text = re.sub(r'RESPONDENT\(S\)', 'RESPONDENTS', text, flags=re.I)
    text = re.sub(r'Respondent \(s\)', 'RESPONDENTS', text, flags=re.I)
    text = re.sub(r'PETITIONER\(S\)', 'PETITIONERS', text, flags=re.I)
    text = re.sub(r'CLAIMANT\(S\)', 'CLAIMANTS', text, flags=re.I)
    #text = re.sub(r'\bC\s*R\s*L\.?\b', 'CRL', text, flags=re.I)
    text = re.sub(r'\bS\s*\.?\s*L\s*\.?\s*P\.?\b', 'SLP', text, flags=re.I)
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.I)
    text = re.sub(
    r'\bSpl\.\s*SC\s*No\.?\s*(\d+)\s*of\s*(\d{4})',
    r'Special Case Number \1 of \2',
    text,
    flags=re.I
)  

    # Restore names
    text = text.replace('__MID__', 'S.')

   


    # Ignore before JUDGMENT or ORDER
    #lines = text.splitlines()
    #start_idx = 0
    #for i, line in enumerate(lines):
        #if re.search(r"^(ORDER|JUDGMENT|J U D G M E N T|O R D E R)", line, re.I):
            #start_idx = i
            #break
    #text = "\n".join(lines[start_idx:])

    # 🔹 FIX: Merge broken PDF lines before sentence tokenization
    def fix_broken_lines(text: str) -> str:
        lines = text.splitlines()
        fixed_lines = []
        buffer = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # If previous line didn’t end with sentence-ending punctuation, merge
            if buffer and not re.search(r'[.!?]"?$', buffer):
                buffer += " " + line
            else:
                if buffer:
                    fixed_lines.append(buffer)
                buffer = line

        if buffer:
            fixed_lines.append(buffer)

        return " ".join(fixed_lines)

    text = fix_broken_lines(text)

    # Now tokenize into sentences
    sentences = []
    for s in sent_tokenize(text):
        s = s.strip()
        if len(s.split()) < 4:
            continue
        if re.fullmatch(r'[\d.\-()]+', s):
            continue
        if s.lower() in {"number", "j.", "j"}:
            continue
        sentences.append(s if s.endswith('.') else s + '.')

    return "\n".join(sentences)


#***********************
found = set()
def find_all_abbreviations(text, abbreviations_dict, contextual_abbreviations):
    found = {}

    # Static abbreviations
    for abbr, meaning in abbreviations_dict.items():
        pattern = r'(?<!\w)' + re.escape(abbr) + r'(?!\w)'
        if re.search(pattern, text, re.I):
            found[abbr.upper()] = meaning

    # Contextual abbreviations
    for abbr, meanings in contextual_abbreviations.items():
        pattern = r'(?<!\w)' + re.escape(abbr) + r'(?!\w)'
        if re.search(pattern, text, re.I):
            if abbr.upper() not in found:
                found[abbr.upper()] = "Contextual Meaning"

    return found

def extract_preamble_block(text: str) -> str:
    """
    Extracts preamble as everything before 'JUDGMENT' or 'ORDER'
    """
    lines = text.splitlines()
    preamble_lines = []

    marker_re = re.compile(r'^(?:\s*(?:j\s*u\s*d\s*g\s*m\s*e\s*n\s*t|judgment|judgement|o\s*r\s*d\s*e\s*r|order)\b).*$', re.I)
    found_marker = False
    for line in lines:
        if marker_re.match(line):
            found_marker = True
            break
        preamble_lines.append(line)

    if not found_marker:
        return ""

    return "\n".join(preamble_lines).strip()


# --- Model loading (cached) ---

@st.cache_resource(show_spinner=False)
def load_labeling_model(repo_id: str = "engineersaloni159/LegalRo-BERt_for_rhetorical_role_labeling"):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_summarizer_model(repo_id: str = "facebook/bart-large-cnn"):
    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
    # Try to use the transformers pipeline; on some environments the
    # 'summarization' task may not be registered and will raise KeyError.
    # Provide a lightweight fallback that uses model.generate directly.
    try:
        # Cast for static type checkers; transformers.pipeline overloads can be
        # too narrow for dynamically loaded Auto* models.
        hf_pipeline = cast(Any, pipeline)
        summarizer = hf_pipeline("summarization", model=model, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)
        return summarizer
    except KeyError:
        # Fallback summarizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        def _summarizer(texts, max_length=150, min_length=30, do_sample=False, **kwargs):
            single = False
            if isinstance(texts, str):
                texts = [texts]
                single = True

            results = []
            for t in texts:
                # Tokenize with truncation to a reasonable max input length
                inputs = tok(t, return_tensors='pt', truncation=True, max_length=1024).to(device)
                try:
                    gen = model.generate(**inputs, max_length=max_length, min_length=min_length, do_sample=do_sample)
                    decoded = tok.decode(gen[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                except Exception as e:
                    decoded = f"[generation failed: {e}]"
                results.append({"summary_text": decoded})

            return results[0] if single else results

        return _summarizer
 
@st.cache_data(show_spinner=False)
def load_label_mapping_from_json(url: str):
    resp = requests.get(url)
    resp.raise_for_status()
    json_data = resp.json()
    rows = []
    for document in json_data:
        doc_id = document.get("id")
        for annotation in document.get("annotations", []):
            for result in annotation.get("result", []):
                rows.append({
                    'doc_id': doc_id,
                    'text': result['value'].get('text'),
                    'label': result['value'].get('labels', [None])[0]
                })
    df = pd.DataFrame(rows)
    le = LabelEncoder()
    df['label'] = df['label'].astype(str)
    df['encoded'] = le.fit_transform(df['label'].values)  # type: ignore
    return le, df

# --- Inference helpers ---

def predict_labels_batch(sentences, tokenizer, model, label_encoder, batch_size=16, device=None):
    model.to(device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')))
    model.eval()
    labels = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(next(model.parameters()).device)
            outputs = model(**enc)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            decoded = label_encoder.inverse_transform(preds)
            labels.extend(decoded)
    return labels


@st.cache_resource(show_spinner=False)
def load_topic_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load sentence embedding model once per session for topic clustering.
    Returns None if sentence-transformers is unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception:
        return None


def _keywords_from_sentences(sentences, top_k: int = 4):
    if not sentences:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000)
        mat = vectorizer.fit_transform(sentences)
        if mat.shape[1] == 0:
            return []
        # scipy sparse stubs can be incomplete across environments; cast to Any
        # so runtime-supported operations are accepted by static type checkers.
        mat_any = cast(Any, mat)
        scores = np.asarray(mat_any.mean(axis=0)).ravel()
        terms = np.array(vectorizer.get_feature_names_out())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [str(terms[i]) for i in top_idx if scores[i] > 0]
    except Exception:
        return []


def extract_topics(topic_text: str, max_topics: int = 5):
    """
    Topic extraction for legal summaries:
    sentences -> MiniLM embeddings -> BERTopic clustering -> top keywords per cluster.
    """
    if not topic_text or not topic_text.strip():
        return []

    sentences = [s.strip() for s in sent_tokenize(topic_text) if len(s.split()) >= 5]
    if len(sentences) < 3:
        return _keywords_from_sentences(sentences, top_k=min(max_topics, 4))

    embedder = load_topic_embedding_model()
    if embedder is None:
        return _keywords_from_sentences(sentences, top_k=min(max_topics, 4))

    try:
        embeddings = embedder.encode(sentences, show_progress_bar=False)
    except Exception:
        return _keywords_from_sentences(sentences, top_k=min(max_topics, 4))

    # Try BERTopic first, fallback to HDBSCAN/DBSCAN if unavailable
    cluster_labels = None
    try:
        from bertopic import BERTopic
        import hdbscan

        # Configure BERTopic with stronger clustering
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, min(8, len(sentences) // 5)), min_samples=1)
        topic_model = BERTopic(
            embedding_model=embedder,
            hdbscan_model=hdbscan_clusterer,
            top_n_words=3,
            verbose=False,
            calculate_probabilities=False
        )
        topics, probs = topic_model.fit_transform(sentences, embeddings)
        cluster_labels = np.array(topics)
    except Exception:
        # Fallback to HDBSCAN/DBSCAN if BERTopic fails
        try:
            import hdbscan
            min_cluster_size = max(2, min(8, len(sentences) // 5))
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)
            cluster_labels = clusterer.fit_predict(embeddings)
        except Exception:
            try:
                from sklearn.cluster import DBSCAN
                clusterer = DBSCAN(eps=0.65, min_samples=2, metric="euclidean")
                cluster_labels = clusterer.fit_predict(embeddings)
            except Exception:
                return _keywords_from_sentences(sentences, top_k=min(max_topics, 4))

    clusters = {}
    for sent, lbl in zip(sentences, cluster_labels):
        if lbl == -1:
            continue
        clusters.setdefault(int(lbl), []).append(sent)

    if not clusters:
        return _keywords_from_sentences(sentences, top_k=min(max_topics, 4))

    ranked = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    topics = []
    for _, cluster_sents in ranked[:max_topics]:
        kws = _keywords_from_sentences(cluster_sents, top_k=3)
        if kws:
            title = " / ".join([kw.title() for kw in kws[:2]])
            topics.append(title)

    # De-duplicate while preserving order.
    seen = set()
    uniq = []
    for t in topics:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq[:max_topics]


def _normalize_act_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name.replace("\n", " ")).strip(" ,.-")
    lower = cleaned.lower()
    alias_map = {
        "ipc": "Indian Penal Code",
        "i.p.c": "Indian Penal Code",
        "crpc": "CrPC",
        "cr. p. c": "CrPC",
        "code of criminal procedure": "CrPC",
        "evidence act": "Indian Evidence Act",
        "indian evidence act": "Indian Evidence Act",
        "constitution": "Constitution",
    }
    return alias_map.get(lower, cleaned.title())


def extract_statutes(text: str, max_items: int = 12):
    """
    Regex + context extraction for statutes/sections/articles in legal judgments.
    """
    if not text or not text.strip():
        return []

    extracted = []
    norm_text = re.sub(r"\s+", " ", text)

    # Pattern: Section 302 of the Indian Penal Code / Evidence Act, 1872
    pattern_section_of_act = re.compile(
        r"\b(?:Section|Sec\.?)\s*(\d+[A-Za-z\-]*)\s+of\s+the\s+([A-Za-z][A-Za-z&.\- ]{2,}?(?:Act|Code|Constitution))(?:,\s*(\d{4}))?",
        re.IGNORECASE,
    )
    for sec, act_name, year in pattern_section_of_act.findall(norm_text):
        act_norm = _normalize_act_name(act_name)
        if year and "Act" in act_norm and year not in act_norm:
            act_norm = f"{act_norm}, {year}"
        extracted.append(f"{act_norm} - Sec {sec}")

    # Pattern: Indian Evidence Act, 1872
    pattern_act_year = re.compile(
        r"\b([A-Za-z][A-Za-z&.\- ]{2,}?\sAct),\s*(\d{4})\b",
        re.IGNORECASE,
    )
    for act_name, year in pattern_act_year.findall(norm_text):
        act_norm = _normalize_act_name(act_name)
        extracted.append(f"{act_norm}, {year}")

    # Pattern: Article 21 of the Constitution
    pattern_article = re.compile(
        r"\bArticle\s+(\d+[A-Za-z\-]*)\s+of\s+the\s+Constitution\b",
        re.IGNORECASE,
    )
    for article in pattern_article.findall(norm_text):
        extracted.append(f"Constitution - Art {article}")

    # Common legal drafting formats:
    # Section 302 IPC | Sec. 164 CrPC | u/s 27 Evidence Act | Section 302 of IPC
    direct_ref_pattern = re.compile(
        r"\b(?:Section|Sections|Sec\.?|S\.|u/s)\s*(\d+[A-Za-z\-\/]*)\s*(?:of\s+(?:the\s+)?)?(IPC|I\.?\s*P\.?\s*C\.?|CrPC|C\.?\s*r\.?\s*P\.?\s*C\.?|Indian Penal Code|Code of Criminal Procedure|Indian Evidence Act|Evidence Act|Constitution)\b",
        re.IGNORECASE,
    )
    for sec, act_raw in direct_ref_pattern.findall(norm_text):
        act_norm = _normalize_act_name(act_raw)
        extracted.append(f"{act_norm} - Sec {sec}")

    # Hybrid/context pattern: Act acronym near section mention.
    act_aliases = [
        (r"\bI\.?\s*P\.?\s*C\.?\b|\bIPC\b|Indian Penal Code", "Indian Penal Code"),
        (r"\bC\.?\s*r\.?\s*P\.?\s*C\.?\b|\bCrPC\b|Code of Criminal Procedure", "CrPC"),
        (r"\bEvidence Act\b|Indian Evidence Act", "Indian Evidence Act"),
    ]
    for alias_pat, canonical in act_aliases:
        near_pattern = re.compile(
            rf"(?:{alias_pat})[\s,:;\-]{{0,25}}(?:Section|Sec\.?)\s*(\d+[A-Za-z\-]*)|(?:Section|Sec\.?)\s*(\d+[A-Za-z\-]*)[\s,:;\-]{{0,25}}(?:{alias_pat})",
            re.IGNORECASE,
        )
        for m in near_pattern.findall(norm_text):
            sec = m[0] if m[0] else m[1]
            if sec:
                extracted.append(f"{canonical} - Sec {sec}")

    # Order-preserving de-duplication.
    seen = set()
    deduped = []
    for item in extracted:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped[:max_items]


def extract_party_judge_info(preamble_text: str) -> dict:
    info = {
        
        "court": "",
        "date": "",
        "bench": [],
        "petitioner": [],
        "respondent": [],
        "authors": [],
        
    }
    if not preamble_text or not preamble_text.strip():
        return info

    lines = [ln.strip() for ln in preamble_text.splitlines() if ln.strip()]
    preamble_flat = re.sub(r"\s+", " ", preamble_text).strip()

    # Court: usually in top lines in Indian judgments.
    for ln in lines[:30]:
        if re.search(r"\b(supreme court|high court|district court|court of)\b", ln, re.I):
            info["court"] = re.sub(r"\s+", " ", ln).strip(" :-")
            break

    # Parties from "A vs B"/"A v. B"/"A & B" - use greedy matching to preserve full names
    for ln in lines[:80]:
        # More robust pattern: capture everything before and after vs/v./&/and
        m = re.search(r"^\s*(.+?)\s+(?:vs|v\.?|versus|&|and)\s+(.+?)\s+(?:on\s+\d+\s+\w+\s*,?\s*\d{4})?$", ln, re.I)
        if not m:
            # Try alternative pattern without date
            m = re.search(r"^\s*(.+?)\s+(?:vs|v\.?|versus|&|and)\s+(.+)$", ln, re.I)
        if m:
            left = re.sub(r"\s+", " ", m.group(1)).strip(" ,.-")
            right = re.sub(r"\s+", " ", m.group(2)).strip(" ,.-")
            if left and not info["petitioner"]:
                info["petitioner"] = [left]
            if right and not info["respondent"]:
                info["respondent"] = [right]
            break

    # Parties from explicit tags.
    for ln in lines[:120]:
        pet = re.search(r"^(.*?)(?:\.\.\.|…|\s+-\s+)?\s*(petitioner|appellant)s?\b", ln, re.I)
        res = re.search(r"^(.*?)(?:\.\.\.|…|\s+-\s+)?\s*(respondent)s?\b", ln, re.I)
        if pet:
            name = re.sub(r"\s+", " ", pet.group(1)).strip(" ,.-")
            if name:
                info["petitioner"].append(name)
        if res:
            name = re.sub(r"\s+", " ", res.group(1)).strip(" ,.-")
            if name:
                info["respondent"].append(name)

    # Bench and authors from CORAM/BEFORE/HON'BLE lines.
    capture_bench = False
    for ln in lines[:160]:
        bench_field = re.search(r"^\s*bench\s*[:\-]\s*(.+)$", ln, re.I)
        if bench_field:
            info["bench"].append(bench_field.group(1).strip())

        if re.search(r"^\s*(coram|before)\s*:?", ln, re.I):
            capture_bench = True
            after = re.split(r":", ln, maxsplit=1)
            if len(after) == 2 and after[1].strip():
                info["bench"].append(after[1].strip())
            continue

        if capture_bench:
            if re.search(r"^\s*(for petitioner|for respondent|date|judgment|order)\b", ln, re.I):
                capture_bench = False
            elif re.search(r"(hon'?ble|justice|j\.)", ln, re.I):
                info["bench"].append(ln)

        if re.search(r"(hon'?ble|justice|j\.)", ln, re.I):
            info["authors"].append(ln)

    # Author from "Author:" style fields.
    for ln in lines[:120]:
        m = re.search(r"^\s*author\s*[:\-]\s*(.+)$", ln, re.I)
        if m:
            info["authors"].append(m.group(1).strip())

    # Date patterns.
    date_patterns = [
        r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b",
        r"\b(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})\b",
        r"\b([A-Za-z]+\s+\d{1,2},\s*\d{4})\b",
    ]
    # Common legal header form: "... vs ... on 5 August, 2022"
    m_on_date = re.search(r"\bon\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})\b", preamble_flat, re.I)
    if m_on_date:
        info["date"] = m_on_date.group(1)

    for ln in lines[:120]:
        if info["date"]:
            break
        if re.search(r"\b(date|dated|pronounced|decided on)\b", ln, re.I):
            for pat in date_patterns:
                m = re.search(pat, ln)
                if m:
                    info["date"] = m.group(1)
                    break
        if info["date"]:
            break
    if not info["date"]:
        for ln in lines[:120]:
            for pat in date_patterns:
                m = re.search(pat, ln)
                if m:
                    info["date"] = m.group(1)
                    break
            if info["date"]:
                break

    # De-duplicate and clean.
    def _dedupe(items):
        seen = set()
        out = []
        for item in items:
            cleaned = re.sub(r"\s+", " ", str(item)).strip(" ,.-")
            if not cleaned:
                continue
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                out.append(cleaned)
        return out

    info["petitioner"] = _dedupe(info["petitioner"])
    info["respondent"] = _dedupe(info["respondent"])
    info["bench"] = _dedupe(info["bench"])
    info["authors"] = _dedupe(info["authors"])

    return info


def render_party_judge_info(info: dict):
    st.markdown("### 👥 Party & Judge Extraction")
    st.markdown(f"• **Petitioner:** {', '.join(info.get('petitioner', [])) or 'Not found'}")
    st.markdown(f"• **Respondent:** {', '.join(info.get('respondent', [])) or 'Not found'}")
    st.markdown(f"• **Bench:** {', '.join(info.get('bench', [])) or 'Not found'}")
    st.markdown(f"• **Court:** {info.get('court', '') or 'Not found'}")
    st.markdown(f"• **Authors:** {', '.join(info.get('authors', [])) or 'Not found'}")
    st.markdown(f"• **Date:** {info.get('date', '') or 'Not found'}")



# --- Streamlit UI ---
st.title("Legal Rhetorical Role Labeling & Summarization")

# Ensure session state key exists
if "role_summaries" not in st.session_state:
    st.session_state["role_summaries"] = {}
if "case_topics" not in st.session_state:
    st.session_state["case_topics"] = []
if "statutes_discussed" not in st.session_state:
    st.session_state["statutes_discussed"] = []
if "party_judge_info" not in st.session_state:
    st.session_state["party_judge_info"] = {}

# Sidebar
st.sidebar.header("Settings")
# Advanced settings in a collapsed expander
with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
    # Allow user to enter HF token (password field) — used for model downloads/auth
    hf_token_input = st.text_input("Hugging Face token (optional)", type="password")
    if hf_token_input:
        st.session_state['hf_token_ui'] = hf_token_input
        st.success("HF token set for this session (used only when loading models)")
    else:
        # ensure key exists but is empty when user hasn't provided a token
        st.session_state.setdefault('hf_token_ui', None)
    
    st.markdown("---")
    st.markdown("**Model Configuration**")
    json_url_input = st.text_input("Label mapping JSON URL", value="https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/train.json")
    label_repo_input = st.text_input("Labeling model repo", value="engineersaloni159/LegalRo-BERt_for_rhetorical_role_labeling")
    summarizer_repo_input = st.text_input("Summarizer model repo", value="facebook/bart-large-cnn")

# --- Legal Terms Glossary ---
legal_glossary = {
    # Parties in a case
    "Plaintiff": "The person or party who initiates a lawsuit by filing a complaint in court seeking legal remedy",
    "Petitioner": "The person who files a petition in court seeking relief or action from the court",
    "Respondent": "The party against whom a petition or appeal is filed, required to respond to the claims",
    #"Appellant": "The party who appeals a court's decision to a higher court seeking review",
    "Defendant": "The person or party against whom a lawsuit is filed, accused of wrongdoing",
    "Accused": "A person charged with a criminal offense",
    "Complainant": "A person who files a formal complaint initiating legal proceedings",
    "Claimant": "A person who makes a claim or seeks relief in a court of law",
    "Intervener": "A person or party who intervenes in a case to protect their interest",
    "Prosecutor": "The legal representative who conducts the prosecution in a criminal case",

    # Legal documents & filings
    "Plaint": "A written document in which a plaintiff sets forth the claims and demands in a civil case",
    #"petition": "A formal written request to a court seeking judicial intervention or relief",
    "Memo": "A written statement submitted to the court presenting arguments",
    "Affidavit": "A written statement confirmed by oath or affirmation, used as evidence in court",
    "Counter Affidavit": "A written response to an affidavit filed by the opposite party",
    "Written Statement": "The defendant's written response to the plaintiff's claims",
    "Rejoinder": "The plaintiff's reply to the defendant's written statement",
    "Curative Petition": "A petition filed to correct or review a judgment after exhausting regular remedies",
    "Writ Petition": "a formal written order issued by anybody, executive or judicial, authorised to do so",
    
    # Legal concepts & doctrines
    "Ratio decidendi": "The legal principle or reason that forms the binding basis of a court's judgment",
    "Obiter dictum": "Statements in a judgment that are not essential to the decision but may have persuasive value",
    "Res judicata": "The legal doctrine that a matter already judged by a court cannot be re-litigated",
    "Prima facie": "Evidence sufficient to establish a fact unless rebutted by the other party",
    "Locus standi": "The legal right or standing to appear before a court and seek relief",
    "Factum of adoption": "The written petition or document filed in court to seek legal approval for adoption",
    "Caveat": "A notice filed in court to prevent disposal of property or relief without notice to the filer",
    "Lis pendens": "A pending lawsuit that affects the title to property",
    "Estoppel": "A legal principle preventing a person from denying facts already established",
    "Adverse possession": "Hostile possession of property without the owner's permission",
    "Due process": "The legal requirement that fair procedures must be followed in legal proceedings",
    "Natural justice": "Fair and unbiased decision-making required in legal proceedings",
    "Parole evidence": "Oral evidence given during trial to explain or vary a written document",
    "Habeas corpus": "A writ requiring a person under arrest to be brought before a judge",
    "Certiorari": "A writ seeking judicial review of a lower court's decision",
    "Mandamus": "A writ commanding a public official to perform their duty",
    "Quo warranto": "A writ questioning the legal authority of a person to hold office",
    
    # Legal roles & positions
    "Advocate": "A legal professional authorized to represent clients in court (similar to lawyer/barrister)",
    "Senior counsel": "A senior advocate designated as Senior Counsel by the High Court or Supreme Court",
    "Amicus curiae": "A person or organization not party to a case who offers information to assist the court",
    "Arbitrator": "A neutral person appointed to resolve disputes outside court",
    "Mediator": "A neutral person who helps parties reach a mutually acceptable agreement",
    
    # Court terms
    "Bench": "The judge or judges hearing a case",
    "Division bench": "A bench of two judges hearing a case",
    "Single bench": "A single judge hearing a case",
    "Full bench": "A bench of three or more judges hearing a case",
    "Chief justice": "The head of a court system who administers the court's business",
    #"judge": "An official who presides over a court of law",
    "Magistrate": "A judicial officer with limited authority to hear cases",
    "Subordinate court": "Lower courts under the supervision of higher courts",
    "Appellate court": "A court that hears appeals from lower courts",
    
    # Case types & procedures
    "Civil appeal": "An appeal in a civil matter from a lower court to a higher court",
    "Criminal appeal": "An appeal in a criminal matter from a lower court to a higher court",
    "Special leave petition": "A petition seeking special permission to appeal against a court order",
    "Writ petition": "A petition filed under Article 32 (Supreme Court) or 226 (High Court) of Constitution",
    "Revision petition": "A petition seeking revision of a lower court's order",
    "Review petition": "A petition seeking review of a judgment on grounds of error",
    "Miscellaneous application": "An application for interim relief or procedural matters",
    "Execution petition": "A petition to enforce or execute a decree or order",
    
    # Legal acts & sections
    "Section": "A specific provision of a legislative act",
    "Article": "A specific provision of the Constitution",
    "Schedule": "A list appended to an act containing supplementary provisions",
    "Clause": "A subdivision of a section in a legal document",
    "Proviso": "A condition or qualification attached to a legal provision",
    "Explanation": "A statement clarifying the meaning of a legal provision",
    
    # Procedural terms
    "Interim order": "A temporary order passed during the pendency of a case",
    "Final order": "A conclusive order deciding the matter finally",
    "Ex parte order": "An order passed without hearing the opposite party",
    "Ad interim": "Temporary order passed until further orders",
    "Maintainability": "Whether a petition is legally acceptable for hearing",
    "Limitation": "The time period within which a legal action must be filed",
    "Cause title": "The title of a case showing the parties (Petitioner vs Respondent)",
    "Memo of parties": "A document showing the names and details of parties to a case",
    
    # Evidence terms
    "Exhibit": "A document or object produced in court as evidence",
    "Material evidence": "Important evidence that influences the decision",
    "Circumstantial evidence": "Evidence from which conclusions can be drawn indirectly",
    "Direct evidence": "Evidence that directly proves a fact without inference",
    "Hearsay evidence": "Second-hand evidence not from direct witness",
    "Expert evidence": "Opinion evidence from a person with specialized knowledge",
    
    # Judgment terms
    "Judgment": "The official decision of a court on a matter",
    #"decree": "The formal expression of a court's decision in a civil case",
    "Order": "A direction issued by a court during proceedings",
    "Verdict": "The decision of a jury or judge on the matters submitted",
    "Observation": "Comments made by a judge that are not part of the binding judgment",
    "Finding": "The court's determination of facts based on evidence",
    
    # Relief & remedies
    "Injunction": "A court order prohibiting a person from doing something",
    "Declaration": "A court order determining the rights of parties without awarding damages",
    "Restitution": "Restoration of something to its rightful owner or original state",
    "Compensation": "Money awarded to make up for loss or injury",
    "Damages": "Monetary award to compensate for loss or injury",
    "Specific performance": "Court order requiring a party to perform their contractual obligations",
    "Permanent injunction": "A final injunction lasting indefinitely",
    "Temporary injunction": "A provisional injunction until final hearing",
    
    # Other common terms
    "In camera": "Proceedings held in private/closed court",
    "Verbatim": "Word for word record of proceedings",
    "Certified copy": "Official copy of a document authenticated by the court",
    "Judicial discretion": "The power of a judge to make decisions based on fairness",
    "Binding precedent": "A legal principle that must be followed in similar cases",
    "Persuasive precedent": "A legal principle from other jurisdictions that may be followed",
    "Doctrine of separation": "The constitutional principle dividing powers between legislature, executive, and judiciary",
    "Rule of law": "The principle that everyone is subject to the law",


    "Accomplice": "A person who has taken part in or aided the commission of a crime",

    "Accused person": "A person or persons accused of committing a crime but not yet tried for it",

    "Acknowledgement": "It is a method to certify or declare one’s knowledge of some document. It is a statement of acceptance.",

    "Acquaintance Rape": "When rape is being committed by a person known/related to the victim.",

    "Acquittal": "A conclusion by a judge that the accused person/s are not guilty of the commission of the charged offence/s.",

    "Actus Reus": "It is the unlawful, physical act that constitutes an essential element of a crime and which, in most cases, must be combined with mens rea (criminal intent) to prove that a crime has been committed.",

    "Adjournment": "The postponement of a case hearing to a later date. Check the wiki page",

    "Adjudication": "The legal process of deciding a dispute between two or more parties by a competent authority.",

    "Admissible Evidence": "The evidence that a trial judge may consider based on the provisions of the Indian Evidence Act. All the evidence submitted by the parties to the court may not be admissible.",

    "Admission": "The acceptance of document, fact, or statement by a party before the court.",

    #"Advocate": "A law graduate entered in any roll under the provisions of the Advocates Act, 1961.",

    #"Affidavit": "A document sworn by a party before a notary asserting that the contents of the document are made to the best of the signatory’s knowledge, information and belief. Pleadings filed in court cases usually need to be supported by affidavits. Check the wiki page",

    "Appeal": "A process by which a litigant can approach a higher court/authority challenging the order or judgment of a lower court, tribunal or authority",

    "Appearance": "A party showing up in court in response to summons or notice. A party can make an appearance either in person or through their lawyer, depending on the case. In criminal proceedings, the complainant and the accused needs to be personally present at every hearing, unless the Court exempts them.",

    "Appellant": "A person who files an appeal i.e. applies to a higher court for a reconsideration of the decision made by a lower court.",

    "Appellate (also see Jurisdiction)": "In a court, those applications that are concerned with decisions made by a lower court, tribunal or authority.",

    "Arrears": "As per the 245th Law Commission report: Some delayed cases might be in the system for longer than the normal time, for valid reasons. Those cases that show unwarranted delay will be referred to as arrears. Check the wiki page",

    "Arrest": "An arrest is an act of taking a person into custody as he/she may be suspected of a crime or an offence. It is done because a person is apprehended for doing something wrong. Check the wiki page",

    "Arrest Warrant": "An order passed by a magistrate or judge authorising a law enforcement agency to arrest a person suspected of committing a crime.",

    "Arson": "It is a voluntary act of burning a property or setting a property on fire.",

    "Assault": "It is a threat or attempt to use criminal force on an individual. Actual physical contact is not required to prove assault.",

    "Attachment": "An order seizing or attaching property/assets (including bank accounts) to satisfy the demands or claims made by a party. Courts may attach a debtor’s property to pay their creditors or to secure the creditors’ interests during the pendency of proceedings.",

    "Backlog": "As per the 245th Law Commission report, when the institution of new cases in any given time period is higher than the disposal of cases in that time period, the difference between institution and disposal is the backlog. This figure represents the accumulation of cases in the system due to the system’s inability to dispose of as many cases as are being filed. Check the wiki page",

    "Bail": "Bail is referred to as the temporary release of the accused person in a criminal case in which the trial has not started or the trial is going on and the court is yet to reach a decision. The court granting bail usually imposes conditions such as sureties, personal bond, participation in investigation, as conditions for release. Check the wiki page",   

    "Beyond Reasonable Doubt": "It is the level of proof that is required to be proved to convict an accused person in a criminal case. In criminal cases, the prosecution bears the burden of proving that the accused person is guilty beyond all reasonable doubt. The judge needs to be convinced beyond reasonable doubt, based on their consideration of the evidence, that the accused is guilty of the crime charged in order to convict them. Check the wiki page",

    "Burden of proof": "The burden of proof is the standard that the parties have to satisfy to prove a fact in court. In criminal cases, the burden of proving the accused person’s guilt is on the prosecution, and they must prove it beyond reasonable doubt.  In civil cases, the burden of proof is on the plaintiff and they have to prove their case by a preponderance of probabilities. This means that a fact is said to be proved when the court either believes it to exist or considers its existence so probable that a prudent man ought, under the circumstances of the particular case, to act upon the supposition that it exists (Narayan Ganesh Dastane v. Sucheta Narayan Dastane 1975 AIR 1534).",

    "Capital punishment": "Capital punishment or the death penalty isthe punishment for a crime which involves takingthe convicted person’s life. In India capital punishment is awarded inthe rarest of rare cases. Checkthe wiki page",

    "Case Number": "A unique identification number provided bythe court for each case, made up of three components: a case type, them said number, andthe year in whichthe case was instituted.",

    "Case Status": "The stage at which a case is, within the process in the court.",

    "Cause List": "A list issued by the registry of the matters to be heard by the court on any day. The bench, court hall number and the position of the matter are indicated on the cause list. This list appears in print form in every court, and is made available on the website of several courts. Check the wiki page",

    "Cause of action": "A set of facts and circumstances sufficient for a party to initiate legal action against another party.",

    "Charge sheet": "Charge sheet refers to a formal police record presented to the court showing the names of each person accused of the criminal offence/s, the nature of the accusations and the crimes, and the evidence. If the person accused of a crime is in prison, the police has to file a chargesheet in 60 days (where the punishment for the crime is less than 10 years) or 90 days (where the punishment of the crime is more than 10 years). Check the wiki page",

    "Circumstantial Evidence": "Circumstantial evidence is indirect evidence that is not based on direct observation. On its face circumstantial evidence does not prove a fact in issue but gives rise to a logical inference that the fact exists. A person can be convicted on the basis of circumstantial evidence only if the circumstances taken cumulatively form a chain so complete that there is no escape from the conclusion that in all human probability, the crime was committed by them.",

    "Civil Procedure Code": "Codified procedural law related to administration of Indian civil law.",

    "Civil": "That part of the law that encompasses business, contracts, estates, domestic (family) relations, accidents, negligence, and everything related to legal issues, statutes, and lawsuits, that is not criminal law.",

    "Commissions": "A commission is appointed by a court to ascertain or investigate facts needed to decide a case. A commission is usually given specific terms of reference. Members of a commission can be academics, social activists/workers, advocates, or judges.",

    "Commutation": "The action of an executive officer to substitute a punishment given to a convicted person, with a less severe punishment. Under the Constitution, the President and Governor have the power to commute sentences.",

    "Complaint": "Any allegation made orally or in writing to the police or a magistrate stating that a criminal offence has been committed, with a view to them taking action to investigate the alleged offence. Check the wiki page",

    "Conviction": "It is a final adjudication of finding an accused person guilty of the commission of s crime by a Court.",

    "Counterclaim": "A claim made by the defendant against the plaintiff in answer to the claim raised by the plaintiff.",

    "Court Hall": "The room in which the judicial proceedings of the court are carried out. Court halls are usually described by the numbers assigned to them e.g. Courthall No. 3.",

    "Court Notice/Summons": "An official document that a court sends to a party informing them that a case has been filed against them, and which indicates the date and time of the next hearing. Check the wiki page",

    "Criminal Procedure Code": "The main legislation on procedure for administration of substantive criminal law in India.",

    "Criminal": "That which pertains to crimes, and requires the administration of penal justice. Involving those cases that deal with a violation of a law in which a citizen inflicts injury upon another citizen or the state. Punishable with the curtailment of liberty, via imprisonment or detention, or fines.",

    "Cross-examination": "The examination of witness by the opposite party shall be called a cross-examination. Cross-examination gives the opposing party an opportunity to point out the weaknesses of a witness’ testimony. The lawyer conducting the cross-examination cannot ask questions outside the scope of the witness’s prior direct examination. Check the wiki page",

    "Culpable homicide not amounting to murder": "An act which has caused death done with the intention of causing death, or causing such bodily injury which is likely to cause death, or done by someone having the knowledge that they can, by their act, likely cause death, amounts to culpable homicide.",

    "Date of Hearing": "The date on which a case is heard in court.",

    "Date of Institution": "The date when a case is filed and registered in a court.",

    "Decree Holder": "The person in favour of whom the judgment and decree is given by a court, directing the other party to take action or desist from acting in terms of the decree.",

    "Decree": "The formal order of a court that conclusively resolves the issues in a case and determines the rights of the parties.",

    #"Defendant": "The party against whom a civil suit is filed",

    "Deferred Sentence": "Postponement or delay of a sentence to a future date.",

    "Delay": "As per the 245th Law Commission report: A case that has been in the court or judicial system for longer than the normal time that it should take for a case of that type to be disposed of.",

    "Disposal": "The resolution of a case. This could either be the dismissal or a charge, or a final judgement. Court records often specify the nature of resolution.",

    "Dispute": "A conflict of claims or rights that has given rise to the subject of litigation.",

    "District": "An administrative unit within a state. Each district is headed by an official called the District Magistrate/Deputy Commissioner/Collector.",

    "Evidence": "Evidence are things or information brought before the court to prove a fact. Evidence can be documents, computers, witness testimony, videos, audio recordings, weapons etc. The Indian Evidence Act governs the admissibility of evidence and burden of proof.",

    "Ex Parte": "A hearing or trial conducted in the absence of one party. Check the wiki page",

    "Examination in chief": "Examination-in-chief is the examination of a witness by the lawyer of the side who called that witness. It is usually followed by a cross-examination of the witness by the lawyers of the opposite side. Check the wiki page",

    "Exculpatory Evidence": "Evidence that establishes the innocence of the defendant.",

    "Executing court": "The court which executes the decree. It is generally the court which passed the decree.",

    "Execution": "In a civil case execution is the process of enforcement of the decree to enable the decree-holder to claim the benefits of the decree. In a criminal case execution is the implementation of death sentence given by a court. Check the wiki page",

    "Executive": "The executive is the part of the government that has sole authority and responsibility for the effecting and enforcing laws.",

    #"Exhibit": "A document, electronic device or other item introduced in evidence during a trial or hearing.",

    "Fine": "A fine is a  sum of money imposed on a convicted person by a court as a punishment.",

    "FIR": "First Information Report (FIR) is a written document prepared by the police when they first receive information about the commission of an offence. Check the wiki page",

    "Forgery": "The act of making false documents or false electronic records to cause damage or injury to the public or any person.",

    "Fundamental Rights": "A charter of rights contained in the Constitution of India which can be enforced against the state.",

    "Garnishee": "Garnishee is a debtor of judgment debtor whose debt is attached by the court.",

    "Gram Nyayalayas": "Village courts instituted by the Gram Nyayalayas Act, 2008 for speedy and easy access to justice system in the rural areas of India. Check the wiki page",

    "Habeas Corpus": "This is a writ that can be filed before the High Court or Supreme Court when a person is in unlawful detention. A writ of habeas corpus is used to bring a prisoner or other detainee before the court to determine if their imprisonment or detention is lawful.",

    "Hand Summons/dasti summons": "Giving summons “by hand” by the plaintiff or his lawyer personally to the defendant.",

    "Hearsay": "A testimony or evidence by an individual not from his personal knowledge but what he heard another person saying.  Hearsay evidence is usually not admissible in a trial.",

    "Homicide": "The act of killing a human being.",

    "Implied Bar": "A claim/right barred by necessary implication of statute, contract, or general principles of law.",

    "Indigent Person": "An individual who does not possess the financial means to afford the court fees to be paid with the plaint in a particular suit.",

    "Inquiry": "It is every inquiry, other than a trial, conducted by a Magistrate.",   

    "Indian Penal Code": "Main criminal code of India that covers all aspects of substantive criminal law in India.",

    "Interpretation": "The process of determining the intended meaning of a written document, such as the constitution, or a statute.",

    "Interlocutory Application": "An application filed by either party during the pendency of a civil proceeding seeking relief of an interim/temporary nature. Check the wiki page",

    "Interrogation": "The process of law enforcement agencies questioning a person accused of a crime. The accused person is not obligated to answer the questions asked by the law enforcement agency, and the fact that they have remained silent generally cannot be used by the prosecution to help prove guilt. It is illegal for the police to use violence in the process of interrogation.",

    "Investigation": "Investigation includes all the proceedings required for the collection of evidence to ascertain whether a crime has been committed, who has committed the crime and to provide evidence to prove the guilt of the accused person. Investigation usually involves collecting physical evidence like fingerprints, computers, weapons etc and information from people like witnesses, informants and suspects. It is conducted by the law enforcement agency or by any person other than a magistrate, who has been authorized by the magistrate on this behalf.",

    "Issues": "The question of fact or law that is in dispute.",

    "Judge": "A public official authorised to hear and decide cases in a court of law. Check the wiki page",

    "Judgment debtor": "The party against whom a court has given a judgement and who is liable to pay/perform in terms of the decree passed by the court in favour of the decree holder.",

    #"Judgment": "The final decision in a case determining the rights of parties and the reasoning for this decision given by a court. Check the wiki page",

    "Judicial Custody": "Judicial custody means the detention of an accused person in the custody of the concerned Magistrate. The accused person is lodged in a prison. Magistrates usually order judicial custody after the accused person has been in police custody for fourteen days or less.",

    "Judiciary": "The judiciary is one of the three main organs of the government (also known as the judicial system or court system). It is the system of courts that interprets and applies the law in the name of the state.",

    "Jurisdiction": "Jurisdiction refers to the legal authority or power of a court to hear and decide a case. It is the power to interpret and apply the law, and to determine the facts of a case and to deliver a binding judgment and enforce it. Jurisdiction can be based on a number of factors, including the location of the parties or the subject matter of the dispute, and may be limited by geography, subject matter, and other legal principles. Check the wiki page",

    "Juvenile": "A juvenile is a person who is under the age of 16 years in the case of boys, or the age of 18 years in the case of girls. A juvenile has to serve their sentence until the age of twenty-one years in a remand home. A person between the ages of sixteen and eighteen, accused of committing ‘heinous offences’ may be tried as an adult",

    "Lawyer": "A person who practices law.",

    "Legal representative": "The legal heir(s) of a deceased person, or the person(s) who represents the deceased person after his death.",

    "Legislative Assembly": "The name given in some countries to either a legislature, or to one of its branch. In India, legislative assembly usually refers to the legislature at the state-level.",

    "Legislature": "A branch of the government that has the power to make laws in a country.",

    #"Limitation": "The maximum time from the date of the offence that parties have to initiate legal proceedings. Limitation does not apply to criminal cases involving offences punishable with imprisonment of more than 3 years.",

    "Litigant": "A party to a lawsuit in a court.",

    "Lok Adalats": "A system of alternative dispute resolution developed in India. It roughly means people’s court and it is a forum where disputes/cases pending in the court of law or at pre-litigation stage are settled/ compromised amicably. Lok Adalats have been given statutory status under the Legal Services Authorities Act, 1987. Check the wiki page",

    "Malice": "Intention to commit a criminal offence.",

    "Mandatory minimum sentence": "It is a minimum sentence prescribed for certain serious offences irrespective of the circumstances of the crime. It reduces a judge’s discretion since they cannot reduce the sentence below the minimum.  E.g. Section 376-DB of IPC provides for a mandatory minimum punishment of life imprisonment without remission for persons convicted for rape/gangrape of a girl under the age of 12 years.",

    "Mediation": "It is a form of alternative dispute resolution where parties choose a third party who is neutral to help them to resolve their dispute through compromise and discussion.",

    "Mens Rea": "It means guilty mind in Latin. Mens rea refers to criminal intent required in order to convict a person, and it is an essential element that has to be proved in criminal proceedings.",

    "Merits": "The substantive grounds of dispute between the parties.",

    "Mitigating Circumstances": "Mitigating circumstances are facts that appear to mitigate the seriousness of a crime by rendering the actions of the accused less severe and influences the sentence given by the judge. E.g. the age of the accused person and their previous criminal record are considered mitigating cricumstances.",

    "Modus Operandi": "The mode or way in which a person commits a crime.",

    "Murder": "An act committed with an intention to cause death. The act is done with the intention of causing such bodily injury which the offender has knowledge that it would result in death.",

    "Notice": "The legal notification by which a party or person is made aware of a legal process affecting their rights, obligations, or duties.",

    "Objection": "An argument or contention raised in response to the actions of the other party in a court or tribunal.",

    #"Order": "A direction issued by a court against one of more parties to a case. Orders can be given during the proceedings or after the case has been decided. Check the wiki page",

    "Original Jurisdiction": "A court’s power to hear a case in the first instance. It is the first court that hears a matter. This depends on the nature of the case and the value of the dispute in the case.",

    "Pecuniary Jurisdiction": "The pecuniary jurisdiction of a court refers to the monetary value of cases that can be adjudicated by it.",

    "Pendency": "As per the 245th Law Commission report pendency refers to all cases instituted but not disposed of, regardless of when the case was instituted. Check the wiki page",  

    "Personal Bond": "A personal bond is an agreement by the accused person as a condition for release on bail that they will appear for court hearings and will comply with the conditions placed on their release.",

    "Petition": "A formal written request presented to a court of law.",

    #"Petitioner": "The party who presents a petition in court of law. Check the wiki page",

    #"Plaintiff": "The party who initiates a lawsuit (also known as an action) before a court of law.",

    "Plea Bargain": "Plea bargaining is a negotiation between the accused and the prosecution where the accused agrees to plead guilty in exchange for certain concessions by the prosecution. It is a bargain where an accused person pleads guilty to a lesser charge and the prosecutors in return drop more serious charges. Check the wiki page",

    "Plea": "In a court of law, a person’s plea is the answer that they give when they have been charged with a crime, saying whether or not they are guilty of that crime. It is usually taken at the initial stage of the trial.",

    "Pleadings": "Written presentation by litigants in a case, setting forth the facts upon which they are claiming legal relief or challenging the claims of the opposite party.",

    "Police Custody": "Police custody means that the physical custody of the accused person is with the police. The accused person is lodged in a lock-up of a police station. Police custody is ordered for a maximum of fourteen days after which the accused person has to be kept in judicial custody. Check the wiki page",

    "Prayer": "A prayer for relief is a portion of a complaint in which the plaintiff describes the remedies that they seek from the court. Check the wiki page",

    "Precedent": "A precedent or authority is a principle or rule established in a previous legal case that is either binding on a court or other tribunal when deciding subsequent cases with similar issues or facts.",

    "Pre-emption": "It is the preferential right to purchase or enjoy property before another.",

    "Prima facie": "It means at first sight in Latin. It is a way to evaluate a case at an initial stage to see if there is enough material for it to go to trial.",

    "Probation": "The release of a convict from prison subject to good behaviour and any other conditions. A violation of probation conditions can lead to its revocation and to imprisonment. Check the wiki page",

    "Procedural": "Procedural law comprises the rules by which a court prescribes the steps for having a right or duty that is judicially enforced, and determines what happens in civil lawsuit, criminal or administrative proceedings.",

    "Proceedings": "It is the form and manner in which a court of law or judicial officer conducts business.",

    "Public Prosecutor": "A lawyer representing the state in a criminal trial. Since a criminal offence is regarded as a public wrong, which has been committed not only against the victim, but also against society as a whole, the case is prosecuted by the state. Check the wiki page",

    "Rape": "Sexual intercourse with a woman against her will, without her consent, by coercion, misrepresentation or fraud or at a time when she has been intoxicated or duped, or is of unsound mental health and in any case if she is under the age of 18 years.",

    "Record": "It is the record maintained by the court for each case, including pleadings, documents and evidence." ,

    "Recusal": "It is an action of a judge to withdraw himself from a case in which his bias to one of the parties may be called into question. Check the wiki page",   

    "Registry": "An office of the court which receives and maintains documents for filing with the court.",

    #"Rejoinder": "A pleading in which the plaintiff explains or rebuts the additional facts brought up by the defendant in their written statement. Check the wiki page",

    "Remand": "Police custody of an accused person ordered by a magistrate when the investigation is not completed within 24 hours of arrest.",

    #"Remand": "To send back the case to the lower court. Check the wiki page",

    "Representative suit": "It is a suit that is filed by one person on behalf of themselves and on behalf of others who have the same interest in the suit.",

    #"Respondent": "A party against whom a petition is filed. This term is generally used in appeals. A respondent can be the plaintiff or defendant from lower court. Check the wiki page",

    "Review": "A proceeding in a court to review its own judgement in case of any error or mistake made with regard to the decision rendered, to rectify the same. Check the wiki page",

    "Right": "A legally protected interest.",

    "Robbery": "Theft is a robbery when in order to commit theft, the offender voluntarily causes or attempts to cause to any person death, subject him to wrongful restraint, cause hurt or induce fear of instant death, instant wrongful restraint or cause instant hurt.",

    "Search Warrant": "An order signed by a judge for probable cause that directs owners of private property to allow the police to enter and search for items named in the warrant.",

    "Separation of Powers": "A constitutional government with three separate branches: the legislative, executive, and judicial.",

    "Set off": "It is a kind of cross-claim for the recovery of money which the defendant raises against the claim of the plaintiff subtract dues owed to the defendant by the plaintiff from the dues claimed by the plaintiff.",

    "Settlement": "It is an agreement reached by the parties in a case to resolve their dispute",

    "Special Courts": "Bodies within the judicial branch of government that generally address only one area of law or have specifically defined powers. Check the wiki page",

    "Stage of the Case": "The steps in process through which a case passes.",

    "Statute": "Any law passed by a legislative body at the municipal, state, or central level.",

    "Substantive": "A statutory, or written law, that creates and defines rights and duties and powers of parties, such as crimes and punishments in criminal law, civil rights and responsibilities in civil law.",

    "Suit": "A civil action brought by a party/parties against another in a court of law. Check the wiki page", 

    "Summons": "Summon is a document that commands a person to whom it is served to appear before the court. and to answer the complaint made against him",

    "Surety": "A surety is a person that guarantees the accused person will attend their court hearing after being granted bail. The surety is required to deposit a security which is forfeited if the accused person fails to appear in court.",

    "Testimony": "It is evidence presented under oath by a witness in court",   

    "Theft": "Theft means the dishonest removal of moveable property out of the possession of any person without their consent",

    "Tort": "A civil wrong in which an injury is caused by one person to another, and which may be intentional or unintentional. Prominent examples of torts include negligent injury, battery, deceit, and defamation.",

    "Transcript": "It is a record of official proceedings of a hearing.",

    "Trial": "It is a process to determine the guilt or innocence of the accused person. It is a structured process where the facts of a case are presented to the judge, and they decide if the accused person is guilty or not.",

    "Tribunals": "Tribunals are dispute resolution institutions established for discharging judicial or quasi-judicial duties related to certain areas of law. Check the wiki page"  ,

    "Vakalatnama": "A document by which an advocate is empowered to appear or plead before any court, tribunal or other authority on behalf of a party. Check the wiki page",

    "Waiver": "It is a voluntary renunciation of a person’s rights, claims or privileges.",

    #"Written statement": "It is a statement of defence filed by the defendant countering the allegations of the plaintiff.",

    "Writs": "A writ is a formal written order issued by a high court or the Supreme Court asking the state to refrain from or perform a specific act. Check the wiki page",

    #The following are the five types of writs.

    #"Mandamus": "An order that is issued by a court of superior jurisdiction to ask a lower court, tribunal, commission, or individual, to perform or refrain from performing an action that is required by law.",

    "Habeas Corpus": "A court order that commands an individual or a government official who has restrained another to produce the prisoner at a designated  time and place so that the court can determine the legality of custody.",

    "Prohibition": "An extraordinary writ issued by a higher court commanding an inferior court or quasi-judicial body to keep within its jurisdiction.",

   # "Quo Warranto": "A writ issued with a view to restrain a person from holding a civil office to which he/she is not entitled",

    "Certiorari": "A writ issued by the Supreme Court or High Court to quash the order already passed by an inferior court, tribunal or quasi-judicial body."

}

# --- Sidebar Glossary Panel for Legal Terms ---
with st.sidebar.expander("📚 Legal Glossary (Common Legal Terms)", expanded=True):
    st.markdown("## 📚 Legal Glossary (Common Legal Terms)")
    st.markdown("_Browse or search for legal terms to understand their meaning while reading the summary._")
    glossary_search = st.text_input("Search legal glossary", key="sidebar_legal_glossary_search")
    # Get terms from uploaded text
    raw_text = st.session_state.get('raw_text', '')
    terms_in_text = set()
    for term in legal_glossary.keys():
        if re.search(r'\b' + re.escape(term) + r'\b', raw_text, re.I):
            terms_in_text.add(term.lower())
    base_filtered = {k: v for k, v in legal_glossary.items() if k.lower() in terms_in_text}
    # Apply search filter
    if glossary_search:
        filtered_terms = {k: v for k, v in base_filtered.items() if glossary_search.lower() in k.lower() or glossary_search.lower() in v.lower()}
    else:
        filtered_terms = base_filtered
    if filtered_terms:
        for term, definition in sorted(filtered_terms.items()):
            st.markdown(f"**{term}**: {definition}")
    else:
        st.info("No matching terms found.")



# Use inputs from advanced settings or defaults
json_url = json_url_input
label_repo = label_repo_input
summarizer_repo = summarizer_repo_input

# Load label mapping
with st.spinner('Loading label mapping...'):
    try:
        label_encoder, df_map = load_label_mapping_from_json(json_url)
        label_classes = list(label_encoder.classes_)
    except Exception as e:
        st.error(f"Failed to load label mapping JSON: {e}")
        label_classes = []

# Upload PDF
uploaded_file = st.file_uploader("Upload a legal PDF", type=['pdf'])



raw_text = ""
if uploaded_file is not None:
    raw_text = extract_text_from_pdf_filelike(uploaded_file)
    st.session_state['raw_text'] = raw_text     



found_abbreviations = find_all_abbreviations(
    raw_text,
    abbreviations_dict,
    contextual_abbreviations
)

if raw_text.strip():
    st.subheader("📘 Abbreviations Detected")

    if found_abbreviations:
        items = list(found_abbreviations.items())

        cols = st.columns(3)   # 3 columns

        for i, (abbr, meaning) in enumerate(items):
            with cols[i % 3]:
                st.markdown(f"**{abbr}** → {meaning}")
    else:
        st.info("No abbreviations detected.")
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Extracted text")
    st.text_area("Extracted", value=raw_text, height=300)

# Preprocess
preprocess_button = st.button("Refine text")
if preprocess_button:
    cleaned_text = preprocess_text(raw_text)
    cleaned_text = expand_abbreviations_safe(cleaned_text, abbreviations_dict)
    # Only remove square brackets and quotes — NOT parentheses
    cleaned_text = re.sub(r"[\[\]\"]", " ", cleaned_text)

    cleaned_text = cleaned_text.strip()


    st.session_state['cleaned_text'] = cleaned_text
else:
    cleaned_text = st.session_state.get('cleaned_text', '')

with col2:
    st.subheader("Refined text")
    st.text_area("Refined", value=cleaned_text, height=300)




# Label selection
st.sidebar.subheader("Select labels to include")
if label_classes:
    default_labels = label_classes
    selected_labels = st.sidebar.multiselect("Labels", options=label_classes, default=default_labels)
else:
    selected_labels = []

# Load models on demand
load_models_btn = st.sidebar.button("Load models")
models_loaded = False
if load_models_btn:
    # Only attempt Hugging Face login if the user explicitly provided
    # a token via the sidebar. Do NOT use environment variables.
    hf_ui = st.session_state.get('hf_token_ui')
    if hf_ui:
        try:
            login(token=hf_ui)
            st.sidebar.info("Logged into Hugging Face using sidebar token")
        except Exception as e:
            st.sidebar.warning(f"Hugging Face login failed: {e}")

    with st.spinner('Loading labeling model... this may take a while'):
        tokenizer_label, model_label = load_labeling_model(label_repo)
    with st.spinner('Loading summarizer...'):
        summarizer = load_summarizer_model(summarizer_repo)
    models_loaded = True
    st.sidebar.success('Models loaded')
else:
    # try load lazily if present in cache
    try:
        # Only login if user explicitly provided a UI token
        hf_ui = st.session_state.get('hf_token_ui')
        if hf_ui:
            try:
                login(token=hf_ui)
            except Exception as e:
                st.sidebar.warning(f"Hugging Face login failed: {e}")

        tokenizer_label, model_label = load_labeling_model(label_repo)
        summarizer = load_summarizer_model(summarizer_repo)
        models_loaded = True
    except Exception:
        models_loaded = False

preamble_text = extract_preamble_block(raw_text)
#remaining_text = raw_text.replace(preamble_text, "")

# Run labeling & summarization
if st.button("Label Sentences"):
    st.session_state["role_summaries"] = {}
    st.session_state["case_topics"] = []
    st.session_state["statutes_discussed"] = []
    st.session_state["party_judge_info"] = {}
    if not cleaned_text:
        st.warning("Preprocess text first")
    elif not models_loaded:
        st.warning("Load models from the sidebar first (button 'Load models')")
    else:
        sentences = [s.strip() for s in cleaned_text.splitlines() if len(s.split()) > 3]
        # 🔹 Extract PREAMBLE from RAW TEXT (not cleaned text)
        #preamble_text = extract_preamble_block(raw_text)

        # 🔹 Store PREAMBLE directly (NO ML)
        st.session_state["role_summaries"]["PREAMBLE"] = preamble_text

        # 🔹 Remove PREAMBLE from body before ML
        if preamble_text:
            idx = raw_text.find(preamble_text)
            if idx != -1:
                body_text = raw_text[idx + len(preamble_text):].strip()
            else:
                body_text = raw_text.strip()
        else:
            body_text = raw_text.strip()

        # If no judgment/order marker is found, keep the full text as body.
        if not body_text:
            body_text = raw_text.strip()

        # 🔹 Preprocess only BODY
        cleaned_body = preprocess_text(body_text)
        cleaned_body = expand_abbreviations_safe(cleaned_body, abbreviations_dict)
        cleaned_body = re.sub(r"[\[\]\"]", " ", cleaned_body)

        sentences = [s.strip() for s in cleaned_body.splitlines() if len(s.split()) > 3]


        if not sentences:
            st.warning("No sentences found after preprocessing")
        else:
            prog = st.progress(0)
            with st.spinner('Predicting labels...'):
                predicted = predict_labels_batch(sentences, tokenizer_label, model_label, label_encoder, batch_size=16)
            

            prog.progress(50)

            # Collect sentences by label and filter by selected labels
            grouped = {}
            for sent, lab in zip(sentences, predicted):
                if lab in selected_labels:
                    grouped.setdefault(lab, []).append(sent)

            st.subheader("Labeled sentences (filtered)")
            for lab, sents in grouped.items():
                st.markdown(f"### {lab} — {len(sents)} sentences")
                # Show first 10 sentences with colors (simple HTML)
                color = "black"
                # user had a color dictionary - use fallback
                # Map some default colors (limited set)
                color_map = {
                    "ANALYSIS": "red",
                    "ARG_PETITIONER": "blue",
                    "ARG_RESPONDENT": "green",
                    "FAC": "orange",
                    "ISSUE": "purple",
                    "NONE": "gray",
                    "PREAMBLE": "brown",
                    "PRE_NOT_RELIED": "pink",
                    "PRE_RELIED": "yellow",
                    "RATIO": "cyan",
                    "RLC": "magenta",
                    "RPC": "lime",
                    "STA": "gold"
                }
                color = color_map.get(lab, 'black')

                for s in sents[:200]:
                    #st.markdown(f"- {s}", unsafe_allow_html=True)
                    #st.markdown(f"<div style='color:{color}; padding:4px'>{s}</div>", unsafe_allow_html=True) #without bullet
                    st.markdown(f"<div style='color:{color};'>• {s}</div>", unsafe_allow_html=True) #with bullete   
            prog.progress(75)
            # Generate summaries but don't display them to user (used only in overall summary)
            for lab, sents in grouped.items():
                long_text = ' '.join(sents)
                if len(long_text.split()) < 20:
                    continue
                # Summarize (beware of long inputs) Doing the Recursive summarization for the long inputs
                try:
                    # Use pipeline for summarization directly
                    summary_result = summarizer(long_text, max_length=150, min_length=30, do_sample=False)  # type: ignore
                    # Handle both list and dict return types from pipeline
                    if isinstance(summary_result, list) and len(summary_result) > 0:
                        summary_text = summary_result[0].get("summary_text", "")  # type: ignore
                    elif isinstance(summary_result, dict):
                        summary_text = summary_result.get("summary_text", "")  # type: ignore
                    else:
                        summary_text = str(summary_result)
                    
                    st.session_state["role_summaries"][lab] = summary_text
                except Exception as e:
                    pass

            # Topic modeling from key rhetorical roles.
            important_roles = ["ISSUE", "ANALYSIS", "FAC"]
            topic_text = " ".join(" ".join(grouped.get(r, [])) for r in important_roles)
            st.session_state["case_topics"] = extract_topics(topic_text)
            st.session_state["statutes_discussed"] = extract_statutes(body_text)
            st.session_state["party_judge_info"] = extract_party_judge_info(preamble_text)

            prog.progress(100)
            st.success('Done')
            st.session_state['show_sections'] = True


def generate_overall_summary(role_summaries: dict) -> str:
    ORDER = [
        ("PREAMBLE", "Case Background"),
        ("FAC", "Facts of the Case"),
        ("RLC", "Ruling by Lower Court"),
        ("ISSUE", "Issues Before the Court"),
        ("ARG_PETITIONER", "Arguments by the Petitioner"),
        ("ARG_RESPONDENT", "Arguments by the Respondent"),
        ("ANALYSIS", "Court’s Analysis"),
        ("STA", "Statutory Provisions Discussed"),
        ("PRE_RELIED", "Precedents Relied Upon"),
        ("PRE_NOT_RELIED", "Precedents Not Relied Upon"),
        ("RATIO", "Ratio Decidendi"),
        ("RPC", "Final Decision of the Court"),
    ]

    summary_parts = []

    for role, heading in ORDER:
        content = role_summaries.get(role, "").strip()
        if content:
            paragraph = (
                f"{heading}: {content}"
            )
            summary_parts.append(paragraph)

    return "\n\n".join(summary_parts)

st.markdown("---")
st.subheader("📁 Key Information")

if st.session_state.get('show_sections', False):
    existing_topics = st.session_state.get("case_topics", [])
    existing_statutes = st.session_state.get("statutes_discussed", [])

    col1, col2, col3 = st.columns(3)
    with col1:
        existing_party_judge_info = st.session_state.get("party_judge_info", {})
        if existing_party_judge_info:
            render_party_judge_info(existing_party_judge_info)
    with col2:
        if existing_statutes:
            st.markdown("### ⚖️ Statutes Discussed")
            for statute in existing_statutes:
                st.markdown(f"• {statute}")
    with col3:
        if existing_topics:
            st.markdown("### 🧭 Case Topics")
            for topic in existing_topics:
                st.markdown(f"• {topic}")

if st.button("Generate Overall Summary"):
    role_summaries = st.session_state.get("role_summaries", {})

    if not role_summaries:
        st.warning("Please generate rhetorical summaries first.")
    else:
        overall_summary = generate_overall_summary(role_summaries)
        st.text_area(
            "Overall Summary",
            value=overall_summary,
            height=400
        )
        st.success('Done')
        st.session_state['show_sections'] = False


st.markdown("---")
#st.caption("Built from the user's Tkinter app — adapted for Streamlit. Models can be large; running locally with a GPU is recommended.")
st.caption("Built for the Supreme Court of India Judgements dataset. Models can be large; running locally with a GPU is recommended.")

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
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

        # If only one page or empty, short-circuit to original behavior
        if len(pages) <= 1:
            return "\n".join(pages)

        # Heuristic: detect repeated top/bottom lines across pages (headers/footers)
        from collections import Counter
        from difflib import SequenceMatcher

        def _normalize_for_match(l: str) -> str:
            s = l.strip()
            # remove explicit page number tokens
            s = re.sub(r"Page\s*\d+(?:\s*of\s*\d+)?", "", s, flags=re.I)
            # remove common date forms like 'on 8 August, 2023' and '8 August 2023'
            s = re.sub(r"\bon\s+\d{1,2}\s+[A-Za-z]+,?\s+\d{4}\b", "", s, flags=re.I)
            s = re.sub(r"\b\d{1,2}\s+[A-Za-z]+,?\s+\d{4}\b", "", s, flags=re.I)
            # strip punctuation for fuzzy matching
            s = re.sub(r"[^\w\s]", " ", s)
            s = re.sub(r"\s+", " ", s)
            return s.strip().lower()

        pages_lines = [[ln for ln in p.splitlines() if ln.strip()] for p in pages]
        top_n = 3
        bottom_n = 3

        top_candidates = []
        bottom_candidates = []
        for lines in pages_lines:
            if not lines:
                continue
            for i in range(min(top_n, len(lines))):
                top_candidates.append(lines[i].strip())
            for i in range(1, min(bottom_n, len(lines)) + 1):
                bottom_candidates.append(lines[-i].strip())

        def cluster_similar(strings, min_count, ratio=0.75):
            groups = []
            for s in strings:
                norm_s = _normalize_for_match(s)
                if not norm_s:
                    continue
                placed = False
                for g in groups:
                    rep = g[0]
                    if SequenceMatcher(None, _normalize_for_match(rep), norm_s).ratio() >= ratio:
                        g.append(s)
                        placed = True
                        break
                if not placed:
                    groups.append([s])
            reps = []
            for g in groups:
                if len(g) >= min_count:
                    rep = Counter(g).most_common(1)[0][0]
                    reps.append(rep)
            return reps

        min_pages = max(2, int(len(pages_lines) * 0.5))
        header_reps = cluster_similar(top_candidates, min_pages, ratio=0.72)
        footer_reps = cluster_similar(bottom_candidates, min_pages, ratio=0.72)

        cleaned_pages = []
        page_num_re = re.compile(r"^\s*Page\s*\d+(?:\s*of\s*\d+)?\s*$", re.I)
        for lines in pages_lines:
            if not lines:
                continue
            # Remove leading header-like lines (fuzzy)
            start = 0
            while start < len(lines):
                ln = lines[start]
                norm_ln = _normalize_for_match(ln)
                if not norm_ln:
                    start += 1
                    continue
                match_found = False
                for rep in header_reps:
                    if SequenceMatcher(None, _normalize_for_match(rep), norm_ln).ratio() >= 0.72:
                        match_found = True
                        break
                if match_found:
                    start += 1
                    continue
                break

            # Remove trailing footer-like lines (fuzzy)
            end = len(lines)
            while end - 1 >= start:
                ln = lines[end - 1]
                norm_ln = _normalize_for_match(ln)
                if not norm_ln:
                    end -= 1
                    continue
                match_found = False
                for rep in footer_reps:
                    if SequenceMatcher(None, _normalize_for_match(rep), norm_ln).ratio() >= 0.72:
                        match_found = True
                        break
                if match_found:
                    end -= 1
                    continue
                break

            candidate = lines[start:end]
            # Also remove explicit page-number-only lines and very short noise
            filtered = []
            for ln in candidate:
                if page_num_re.match(ln):
                    continue
                if len(ln.strip()) <= 2:
                    continue
                filtered.append(ln)

            if filtered:
                cleaned_pages.append("\n".join(filtered))

        # Final post-processing: aggressively remove running headers
        all_text = "\n".join(cleaned_pages)
        lines = all_text.splitlines()
        
        # Pattern 1: Remove lines matching "X vs Y on <date>"
        header_date_re = re.compile(
            r"(.+?)\s+(?:vs|v\.?|versus)\s+(.+?)\s+on\s+\d{1,2}\s+[A-Za-z]+,?\s+\d{4}",
            re.I
        )
        # Pattern 2: Also catch partial headers that start with case name patterns
        partial_header_re = re.compile(
            r"^\s*[A-Z][A-Za-z\.\s]+(?:vs|v\.?|versus)[\s\.]{0,5}$|^\s*[A-Z][A-Za-z\.\s]+vs[\s\.]*$|^\s*(?:on|April|May|June|July|August|September|October|November|December|January|February|March),?\s+\d{4}\s*$",
            re.I
        )
        # Pattern 3: Remove legal citation lines like "2023 INSC 682" or "2019 SC 249"
        citation_re = re.compile(r"^\s*\d{4}\s+[INSC]+\s+\d+\s*$")
        
        # Remove header lines
        filtered_lines = []
        prev_line = ""
        for ln in lines:
            stripped = ln.strip()
            
            # Skip if matches full header pattern with date
            if header_date_re.search(ln):
                continue
            
            # Skip if matches partial header pattern (incomplete headers across lines)
            if partial_header_re.match(ln):
                continue
            
            # Skip if matches legal citation pattern (e.g. "2023 INSC 682")
            if citation_re.match(ln):
                continue
            
            # Skip if it's a duplicate of the previous line (catches repeated headers)
            if stripped and stripped == prev_line:
                continue
            
            filtered_lines.append(ln)
            prev_line = stripped
        
        return "\n".join(filtered_lines)
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
    "Petitioner": "The person who files a petition in court seeking relief or action from the court",
    "Respondent": "The party against whom a petition or appeal is filed, required to respond to the claims",
    "Defendant": "The person or party against whom a lawsuit is filed, accused of wrongdoing",
    "Accused": "A person charged with a criminal offense",
    "Complainant": "A person who files a formal complaint initiating legal proceedings",
    "Claimant": "A person who makes a claim or seeks relief in a court of law",
    "Intervener": "A person or party who intervenes in a case to protect their interest",
    "Prosecutor": "The legal representative who conducts the prosecution in a criminal case",
    "Plaint": "A written document in which a plaintiff sets forth the claims and demands in a civil case",
    "Memo": "A written statement submitted to the court presenting arguments",
    "Affidavit": "A written statement confirmed by oath or affirmation, used as evidence in court",
    "Counter Affidavit": "A written response to an affidavit filed by the opposite party",
    "Written Statement": "The defendant's written response to the plaintiff's claims",
    "Rejoinder": "The plaintiff's reply to the defendant's written statement",
    "Curative Petition": "A petition filed to correct or review a judgment after exhausting regular remedies",
    "Writ Petition": "a formal written order issued by anybody, executive or judicial, authorised to do so",
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
    "Natural justice": "Fair and unbiased decision-making required in legal proceedings",
    "Parole evidence": "Oral evidence given during trial to explain or vary a written document",
    "Habeas corpus": "A writ requiring a person under arrest to be brought before a judge",
    "Certiorari": "A writ seeking judicial review of a lower court's decision",
    "Mandamus": "Latin: “We command.”A writ commanding a public official to perform their duty",
    "Quo warranto": "A writ questioning the legal authority of a person to hold office",
    "Advocate": "A legal professional authorized to represent clients in court (similar to lawyer/barrister)",
    "Senior counsel": "A senior advocate designated as Senior Counsel by the High Court or Supreme Court",
    "Amicus curiae": "A person or organization not party to a case who offers information to assist the court",
    "Arbitrator": "A neutral person appointed to resolve disputes outside court",
    "Mediator": "A neutral person who helps parties reach a mutually acceptable agreement",
    "Bench": "The judge or judges hearing a case",
    "Division bench": "A bench of two judges hearing a case",
    "Single bench": "A single judge hearing a case",
    "Full bench": "A bench of three or more judges hearing a case",
    "Chief justice": "The head of a court system who administers the court's business",
    "Magistrate": "A judicial officer with limited authority to hear cases",
    "Subordinate court": "Lower courts under the supervision of higher courts",
    "Appellate court": "A court that hears appeals from lower courts",
    "Civil appeal": "An appeal in a civil matter from a lower court to a higher court",
    "Criminal appeal": "An appeal in a criminal matter from a lower court to a higher court",
    "Special leave petition": "A petition seeking special permission to appeal against a court order",
    "Revision petition": "A petition seeking revision of a lower court's order",
    "Review petition": "A petition seeking review of a judgment on grounds of error",
    "Miscellaneous application": "An application for interim relief or procedural matters",
    "Execution petition": "A petition to enforce or execute a decree or order",
    "Section": "A specific provision of a legislative act",
    "Article": "A specific provision of the Constitution",
    "Schedule": "A list appended to an act containing supplementary provisions",
    "Clause": "A subdivision of a section in a legal document",
    "Proviso": "A condition or qualification attached to a legal provision",
    "Explanation": "A statement clarifying the meaning of a legal provision",
    "Interim order": "A temporary order passed during the pendency of a case",
    "Final order": "A conclusive order deciding the matter finally",
    "Ex parte order": "An order passed without hearing the opposite party",
    "Ad interim": "Temporary order passed until further orders",
    "Maintainability": "Whether a petition is legally acceptable for hearing",
    "Limitation": "The time period within which a legal action must be filed",
    "Cause title": "The title of a case showing the parties (Petitioner vs Respondent)",
    "Memo of parties": "A document showing the names and details of parties to a case",
    "Exhibit": "A document or object produced in court as evidence",
    "Material evidence": "Important evidence that influences the decision",
    "Circumstantial evidence": "Evidence from which conclusions can be drawn indirectly",
    "Direct evidence": "Evidence that directly proves a fact without inference, e.g., eyewitness testimony.",
    "Hearsay evidence": "Second-hand evidence not from direct witness",
    "Expert evidence": "Opinion evidence from a person with specialized knowledge",
    "Judgment": "The official decision of a court on a matter",
    "Observation": "Comments made by a judge that are not part of the binding judgment",
    "Finding": "The court's determination of facts based on evidence",
    "Declaration": "A court order determining the rights of parties without awarding damages",
    "Restitution": "Restoration of something to its rightful owner or original state",
    "Compensation": "Money awarded to make up for loss or injury",
    "Damages": "Monetary award to compensate for loss or injury",
    "Specific performance": "Court order requiring a party to perform their contractual obligations",
    "Permanent injunction": "A final injunction lasting indefinitely",
    "Temporary injunction": "A provisional injunction until final hearing",
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
    "Adjournment": "The postponement of a case hearing to a later date.  ",
    "Adjudication": "The legal process of deciding a dispute between two or more parties by a competent authority.",
    "Admissible Evidence": "The evidence that a trial judge may consider based on the provisions of the Indian Evidence Act. All the evidence submitted by the parties to the court may not be admissible.",
    "Admission": "The acceptance of document, fact, or statement by a party before the court.",
    "Appeal": "A process by which a litigant can approach a higher court/authority challenging the order or judgment of a lower court, tribunal or authority",
    "Appearance": "A party showing up in court in response to summons or notice. A party can make an appearance either in person or through their lawyer, depending on the case. In criminal proceedings, the complainant and the accused needs to be personally present at every hearing, unless the Court exempts them.",
    "Appellant": "A person who files an appeal i.e. applies to a higher court for a reconsideration of the decision made by a lower court.",
    "Appellate": "In a court, those applications that are concerned with decisions made by a lower court, tribunal or authority.",
    "Arrears": "As per the 245th Law Commission report: Some delayed cases might be in the system for longer than the normal time, for valid reasons. Those cases that show unwarranted delay will be referred to as arrears.  ",
    "Arrest Warrant": "An order passed by a magistrate or judge authorising a law enforcement agency to arrest a person suspected of committing a crime.",
    "Arson": "It is a voluntary act of burning a property or setting a property on fire.",
    "Assault": "It is a threat or attempt to use criminal force on an individual. Actual physical contact is not required to prove assault.",
    "Backlog": "As per the 245th Law Commission report, when the institution of new cases in any given time period is higher than the disposal of cases in that time period, the difference between institution and disposal is the backlog. This figure represents the accumulation of cases in the system due to the system’s inability to dispose of as many cases as are being filed.  ",
    "Bail": "Bail is referred to as the temporary release of the accused person in a criminal case in which the trial has not started or the trial is going on and the court is yet to reach a decision. The court granting bail usually imposes conditions such as sureties, personal bond, participation in investigation, as conditions for release.  ",   
    "Beyond Reasonable Doubt": "It is the level of proof that is required to be proved to convict an accused person in a criminal case. In criminal cases, the prosecution bears the burden of proving that the accused person is guilty beyond all reasonable doubt. The judge needs to be convinced beyond reasonable doubt, based on their consideration of the evidence, that the accused is guilty of the crime charged in order to convict them.  ",
    "Burden of proof": "The burden of proof is the standard that the parties have to satisfy to prove a fact in court. In criminal cases, the burden of proving the accused person’s guilt is on the prosecution, and they must prove it beyond reasonable doubt.  In civil cases, the burden of proof is on the plaintiff and they have to prove their case by a preponderance of probabilities. This means that a fact is said to be proved when the court either believes it to exist or considers its existence so probable that a prudent man ought, under the circumstances of the particular case, to act upon the supposition that it exists (Narayan Ganesh Dastane v. Sucheta Narayan Dastane 1975 AIR 1534).",
    "Capital punishment": "Capital punishment or the death penalty isthe punishment for a crime which involves takingthe convicted person’s life. In India capital punishment is awarded inthe rarest of rare cases. Checkthe wiki page",
    "Case Number": "A unique identification number provided bythe court for each case, made up of three components: a case type, them said number, andthe year in whichthe case was instituted.",
    "Case Status": "The stage at which a case is, within the process in the court.",
    "Cause List": "A list issued by the registry of the matters to be heard by the court on any day. The bench, court hall number and the position of the matter are indicated on the cause list. This list appears in print form in every court, and is made available on the website of several courts.  ",
    "Charge sheet": "Charge sheet refers to a formal police record presented to the court showing the names of each person accused of the criminal offence/s, the nature of the accusations and the crimes, and the evidence. If the person accused of a crime is in prison, the police has to file a chargesheet in 60 days (where the punishment for the crime is less than 10 years) or 90 days (where the punishment of the crime is more than 10 years). ",
    "Civil Procedure Code": "Codified procedural law related to administration of Indian civil law.",
    "Civil": "That part of the law that encompasses business, contracts, estates, domestic (family) relations, accidents, negligence, and everything related to legal issues, statutes, and lawsuits, that is not criminal law.",
    "Commissions": "A commission is appointed by a court to ascertain or investigate facts needed to decide a case. A commission is usually given specific terms of reference. Members of a commission can be academics, social activists/workers, advocates, or judges.",
    "Conviction": "It is a final adjudication of finding an accused person guilty of the commission of s crime by a Court.",
    "Court Hall": "The room in which the judicial proceedings of the court are carried out. Court halls are usually described by the numbers assigned to them e.g. Courthall No. 3.",
    "Court Notice/Summons": "An official document that a court sends to a party informing them that a case has been filed against them, and which indicates the date and time of the next hearing.  ",
    "Criminal Procedure Code": "The main legislation on procedure for administration of substantive criminal law in India.",
    "Criminal": "That which pertains to crimes, and requires the administration of penal justice. Involving those cases that deal with a violation of a law in which a citizen inflicts injury upon another citizen or the state. Punishable with the curtailment of liberty, via imprisonment or detention, or fines.",
    "Cross-examination": "The examination of witness by the opposite party shall be called a cross-examination. Cross-examination gives the opposing party an opportunity to point out the weaknesses of a witness’ testimony. The lawyer conducting the cross-examination cannot ask questions outside the scope of the witness’s prior direct examination.  ",
    "Culpable homicide not amounting to murder": "An act which has caused death done with the intention of causing death, or causing such bodily injury which is likely to cause death, or done by someone having the knowledge that they can, by their act, likely cause death, amounts to culpable homicide.",
    "Date of Hearing": "The date on which a case is heard in court.",
    "Date of Institution": "The date when a case is filed and registered in a court.",
    "Decree Holder": "The person in favour of whom the judgment and decree is given by a court, directing the other party to take action or desist from acting in terms of the decree.",
    "Decree": "The formal order of a court that conclusively resolves the issues in a case and determines the rights of the parties.",
    "Deferred Sentence": "Postponement or delay of a sentence to a future date.",
    "Delay": "As per the 245th Law Commission report: A case that has been in the court or judicial system for longer than the normal time that it should take for a case of that type to be disposed of.",
    "Disposal": "The resolution of a case. This could either be the dismissal or a charge, or a final judgement. Court records often specify the nature of resolution.",
    "Dispute": "A conflict of claims or rights that has given rise to the subject of litigation.",
    "District": "An administrative unit within a state. Each district is headed by an official called the District Magistrate/Deputy Commissioner/Collector.",
    "Ex Parte": "A hearing or trial conducted in the absence of one party.  ",
    "Examination in chief": "Examination-in-chief is the examination of a witness by the lawyer of the side who called that witness. It is usually followed by a cross-examination of the witness by the lawyers of the opposite side.  ",
    "Exculpatory Evidence": "Evidence that establishes the innocence of the defendant.",
    "Executing court": "The court which executes the decree. It is generally the court which passed the decree.",
    "Execution": "In a civil case execution is the process of enforcement of the decree to enable the decree-holer to claim the benefits of the decree. In a criminal case execution is the implementation of death sentence given by a court.  ",
    "Executive": "The executive is the part of the government that has sole authority and responsibility for the effecting and enforcing laws.",
    "FIR": "First Information Report (FIR) is a written document prepared by the police when they first receive information about the commission of an offence.  ",
    "Forgery": "The act of making false documents or false electronic records to cause damage or injury to the public or any person.",
    "Fundamental Rights": "A charter of rights contained in the Constitution of India which can be enforced against the state.",
    "Garnishee": "Garnishee is a debtor of judgment debtor whose debt is attached by the court.",
    "Gram Nyayalayas": "Village courts instituted by the Gram Nyayalayas Act, 2008 for speedy and easy access to justice system in the rural areas of India.  ",
    "Hand Summons/dasti summons": "Giving summons “by hand” by the plaintiff or his lawyer personally to the defendant.",
    "Hearsay": "A testimony or evidence by an individual not from his personal knowledge but what he heard another person saying.  Hearsay evidence is usually not admissible in a trial.",
    "Implied Bar": "A claim/right barred by necessary implication of statute, contract, or general principles of law.",
    "Indigent Person": "An individual who does not possess the financial means to afford the court fees to be paid with the plaint in a particular suit.",
    "Inquiry": "It is every inquiry, other than a trial, conducted by a Magistrate.",   
    "Indian Penal Code": "Main criminal code of India that covers all aspects of substantive criminal law in India.",
    "Interpretation": "The process of determining the intended meaning of a written document, such as the constitution, or a statute.",
    "Interlocutory Application": "An application filed by either party during the pendency of a civil proceeding seeking relief of an interim/temporary nature.  ",
    "Interrogation": "The process of law enforcement agencies questioning a person accused of a crime. The accused person is not obligated to answer the questions asked by the law enforcement agency, and the fact that they have remained silent generally cannot be used by the prosecution to help prove guilt. It is illegal for the police to use violence in the process of interrogation.",
    "Investigation": "Investigation includes all the proceedings required for the collection of evidence to ascertain whether a crime has been committed, who has committed the crime and to provide evidence to prove the guilt of the accused person. Investigation usually involves collecting physical evidence like fingerprints, computers, weapons etc and information from people like witnesses, informants and suspects. It is conducted by the law enforcement agency or by any person other than a magistrate, who has been authorized by the magistrate on this behalf.",
    "Issues": "The question of fact or law that is in dispute.",
    "Judge": "A public official authorised to hear and decide cases in a court of law.  ",
    "Judgment debtor": "The party against whom a court has given a judgement and who is liable to pay/perform in terms of the decree passed by the court in favour of the decree holder.",
    "Judicial Custody": "Judicial custody means the detention of an accused person in the custody of the concerned Magistrate. The accused person is lodged in a prison. Magistrates usually order judicial custody after the accused person has been in police custody for fourteen days or less.",
    "Judiciary": "The judiciary is one of the three main organs of the government (also known as the judicial system or court system). It is the system of courts that interprets and applies the law in the name of the state.",
    "Juvenile": "A juvenile is a person who is under the age of 16 years in the case of boys, or the age of 18 years in the case of girls. A juvenile has to serve their sentence until the age of twenty-one years in a remand home. A person between the ages of sixteen and eighteen, accused of committing ‘heinous offences’ may be tried as an adult",
    "Lawyer": "A person who practices law.",
    "Legal representative": "The legal heir(s) of a deceased person, or the person(s) who represents the deceased person after his death.",
    "Legislative Assembly": "The name given in some countries to either a legislature, or to one of its branch. In India, legislative assembly usually refers to the legislature at the state-level.",
    "Legislature": "A branch of the government that has the power to make laws in a country.",
    "Lok Adalats": "A system of alternative dispute resolution developed in India. It roughly means people’s court and it is a forum where disputes/cases pending in the court of law or at pre-litigation stage are settled/ compromised amicably. Lok Adalats have been given statutory status under the Legal Services Authorities Act, 1987.  ",
    "Malice": "Intention to commit a criminal offence.",
    "Mandatory minimum sentence": "It is a minimum sentence prescribed for certain serious offences irrespective of the circumstances of the crime. It reduces a judge’s discretion since they cannot reduce the sentence below the minimum.  E.g. Section 376-DB of IPC provides for a mandatory minimum punishment of life imprisonment without remission for persons convicted for rape/gangrape of a girl under the age of 12 years.",
    "Mediation": "It is a form of alternative dispute resolution where parties choose a third party who is neutral to help them to resolve their dispute through compromise and discussion.",
    "Merits": "The substantive grounds of dispute between the parties.",
    "Mitigating Circumstances": "Mitigating circumstances are facts that appear to mitigate the seriousness of a crime by rendering the actions of the accused less severe and influences the sentence given by the judge. E.g. the age of the accused person and their previous criminal record are considered mitigating cricumstances.",
    "Modus Operandi": "The mode or way in which a person commits a crime.",
    "Murder": "An act committed with an intention to cause death. The act is done with the intention of causing such bodily injury which the offender has knowledge that it would result in death.",
    "Objection": "An argument or contention raised in response to the actions of the other party in a court or tribunal.",
    "Original Jurisdiction": "A court’s power to hear a case in the first instance. It is the first court that hears a matter. This depends on the nature of the case and the value of the dispute in the case.",
    "Pecuniary Jurisdiction": "The pecuniary jurisdiction of a court refers to the monetary value of cases that can be adjudicated by it.",
    "Pendency": "As per the 245th Law Commission report pendency refers to all cases instituted but not disposed of, regardless of when the case was instituted.  ",  
    "Personal Bond": "A personal bond is an agreement by the accused person as a condition for release on bail that they will appear for court hearings and will comply with the conditions placed on their release.",
    "Petition": "A formal written request presented to a court of law.",
    "Plea": "In a court of law, a person’s plea is the answer that they give when they have been charged with a crime, saying whether or not they are guilty of that crime. It is usually taken at the initial stage of the trial.",
    "Police Custody": "Police custody means that the physical custody of the accused person is with the police. The accused person is lodged in a lock-up of a police station. Police custody is ordered for a maximum of fourteen days after which the accused person has to be kept in judicial custody.  ",
    "Prayer": "A prayer for relief is a portion of a complaint in which the plaintiff describes the remedies that they seek from the court.  ",
    "Pre-emption": "It is the preferential right to purchase or enjoy property before another.",
    "Probation": "The release of a convict from prison subject to good behaviour and any other conditions. A violation of probation conditions can lead to its revocation and to imprisonment.  ",
    "Procedural": "Procedural law comprises the rules by which a court prescribes the steps for having a right or duty that is judicially enforced, and determines what happens in civil lawsuit, criminal or administrative proceedings.",
    "Proceedings": "It is the form and manner in which a court of law or judicial officer conducts business.",
    "Public Prosecutor": "A lawyer representing the state in a criminal trial. Since a criminal offence is regarded as a public wrong, which has been committed not only against the victim, but also against society as a whole, the case is prosecuted by the state.  ",
    "Rape": "Sexual intercourse with a woman against her will, without her consent, by coercion, misrepresentation or fraud or at a time when she has been intoxicated or duped, or is of unsound mental health and in any case if she is under the age of 18 years.",
    "Recusal": "It is an action of a judge to withdraw himself from a case in which his bias to one of the parties may be called into question.  ",   
    "Registry": "An office of the court which receives and maintains documents for filing with the court.",
    "Representative suit": "It is a suit that is filed by one person on behalf of themselves and on behalf of others who have the same interest in the suit.",
    "Right": "A legally protected interest.",
    "Robbery": "Theft is a robbery when in order to commit theft, the offender voluntarily causes or attempts to cause to any person death, subject him to wrongful restraint, cause hurt or induce fear of instant death, instant wrongful restraint or cause instant hurt.",
    "Separation of Powers": "A constitutional government with three separate branches: the legislative, executive, and judicial.",
    "Set off": "It is a kind of cross-claim for the recovery of money which the defendant raises against the claim of the plaintiff subtract dues owed to the defendant by the plaintiff from the dues claimed by the plaintiff.",
    "Settlement": "It is an agreement reached by the parties in a case to resolve their dispute",
    "Special Courts": "Bodies within the judicial branch of government that generally address only one area of law or have specifically defined powers.  ",
    "Stage of the Case": "The steps in process through which a case passes.",
    "Statute": "Any law passed by a legislative body at the municipal, state, or central level.",
    "Substantive": "A statutory, or written law, that creates and defines rights and duties and powers of parties, such as crimes and punishments in criminal law, civil rights and responsibilities in civil law.",
    "Suit": "A civil action brought by a party/parties against another in a court of law.  ", 
    "Testimony": "It is evidence presented under oath by a witness in court",   
    "Theft": "Theft means the dishonest removal of moveable property out of the possession of any person without their consent",
    "Trial": "It is a process to determine the guilt or innocence of the accused person. It is a structured process where the facts of a case are presented to the judge, and they decide if the accused person is guilty or not.",
    "Tribunals": "Tribunals are dispute resolution institutions established for discharging judicial or quasi-judicial duties related to certain areas of law.  "  ,
    "Vakalatnama": "A document by which an advocate is empowered to appear or plead before any court, tribunal or other authority on behalf of a party.  ",
    "Writs": "A writ is a formal written order issued by a high court or the Supreme Court asking the state to refrain from or perform a specific act.  ", 
    "Prohibition": "An extraordinary writ issued by a higher court commanding an inferior court or quasi-judicial body to keep within its jurisdiction.",



    "ABROGATE": "To annul, cancel or repeal an order or rule.",
    "ABSOLUTE IMMUNITY": "A total exemption from civil liability.",
    "ABSTRACT OF RECORD": "1. An impartial summary of the most important parts of the pleadings, testimony, exhibits and other matters from the trial court record of a case on appeal." ,

    "ABSTRACT OF TITLE": "A condensed history of landownership.",
    "ABUSE OF PROCESS": "A tort claiming that a legal process or procedure has been used for an improper purpose.",
    "ACCESSORY AFTER THE FACT": "One who assisted a person who has committed a felony from being apprehended, arrested or convicted.",
    "ACCESSORY BEFORE THE FACT": "One who acted or contributed as an assistant or instigator to the commission of a crime.",

    "ACCORD AND SATISFACTION": "Discharge of a claim by full payment or use of instrument.",
    "ACCOUNTING FOR COSTS": "A clerk’s itemized statement of costs incurred in a civil action submitted to the parties as set forth in M.R.C.P. 3(d).",
    "ACKNOWLEDGMENT": "A formal statement, usually before an authorized official such as a notary public, acknowledging voluntary execution of a document.",
    "ACQUIT": "1. To render a verdict of not guilty. 2. To release from an obligation or accusation.",

    "ACT OF GOD": "A defense that applies where an injury is attributable solely to a natural cause without any human intervention, which the exercise of prudent care could not have prevented.",
    "ADDITUR": "An increase by the trial court in the amount of damages awarded by the jury.",
    "AD HOC": "Latin: “To this.” For this particular purpose or occasion.",
    "AD HOMINEM": "Latin: “To the person.” Hostile accusations unsupported by logic or reason.",
    "ADMINISTRATION OF ESTATE": "The management and settlement of the estate of a deceased person who has died intestate, or with no named executor, for the main purpose of: ascertaining, collecting, and caring for the non-exempt assets of the estate; ascertaining the debts" "of the estate in the manner prescribed by statute; paying all just debts that are duly probated; and distributing remaining assets to the heirs.",

    "ADMINISTRATOR, -TRIX": "One who administers a decedent’s estate.",
    "ADOPTION": "Legal process granting parental status to a party for the purpose of rearing a child whose natural parents are deceased, unfit or unwilling to do so.",    
    "ADULT": "One who has reached the legal age of majority.",
    "AD VALOREM TAX": "A tax or duty upon the value of the article or thing subject to taxation.",
    "ADVERSARY": "A party opponent in a civil action.",
   # "ADVERSE POSSESSION": "Acquiring ownership of real property by uninterrupted occupying or possessing it for a statutorily prescribed period of time.",
    "AFFIANT": "The person who makes and signs an affidavit.",
    "AFFIRMATION": "A solemn and formal declaration or assertion that the witness will tell the truth, that an affidavit is true, etc.; given in place of an oath.",
    "A FORTIORI": "Latin: “With greater force.”",
    "AGENT": "One authorized to act on behalf of a particular entity or person.",
    "AID AND ABET": "One who incites, encourages, or counsels another in the commission of a crime.",
    "ALIAS": "Latin: “Otherwise.” A name other than a person’s legal name.",
    "ALIAS SUMMONS": "Process issued when the original summons has not been effective; supercedes the original.",
    "ALIBI": "Latin: “Elsewhere.” A defense that places the defendant elsewhere at the time of the crime.",
    "ALIENATION OF AFFECTION": "A tort claiming that one has intentionally and wrongfully interfered in another’s marriage.",
    "ALIMONY": "A sum of money which a court orders one spouse to pay the separated or former spouse for support, aid and maintenance.",
    "ALIMONY PENDENTE LITE": "A temporary support order to maintain the status quo during the course of divorce proceedings. Such is now referred to as temporary alimony.",
    "ALLEGATION": "A formal assertion set out in a pleading.",
    "AMENDMENT TO CONFORM TO THE EVIDENCE": "An amendment of the pleadings to conform to the evidence presented at trial.",
    #"AMICUS CURIAE": "Latin: “A friend of the court.”", 
    "ANNOTATIONS": "Summaries of cases interpreting constitutional or statutory provisions.",
    "ANNULMENT": "A court order declaring that a marriage,contract, or other agreement is void.",
    "ANSWER": "A legal pleading in which the defendant responds to the plaintiff’s claims.",
    "ANTE": "Latin: “Before.”",
    "ANTENUPTIAL AGREEMENT": "See, PRENUPTIAL AGREEMENT.",
    #"APPEAL": "Procedures allowing a higher court to review alleged errors committed at the trial court level.",
    "APPELLEE": "One against whom an appeal is taken.",
    "APROPOS": "Pertinent to time, place or occasion.",
    "ARGUENDO": "Latin: “For the sake of argument.”",
    "ARRAIGNMENT": "Procedure whereby a criminal defendant comes before the court to hear the charge and to enter a plea.",
    "ARREST": "Taking into custody a person to answer accusations of criminal conduct.",
    #"ARREST WARRANT": "A writ issued by a judge, based upon probable cause, to a law enforcement officer to take into custody the person named in it.",
    #"ASSAULT": "1. Civil. An intentional and unlawful attempt or threat, either by words or acts, to inflict injury upon another. 2. Criminal. Simple and aggravated assault are defined under Miss. Code Ann. Section 97-3-7.",
    "ASSIGNEE": "One to whom an assignment is made.",
    "ASSIGNMENT": "Voluntary transfer of rights to property.",
    "ASSIGNOR": "One who makes an assignment.",
    "ASSUMPSIT": "Latin: “He promised.” An action for the recovery of damages for the nonperformance of an implied contract.",
    "ATTACHMENT": "Legal process by which property is seized and brought within the custody of the court to secure satisfaction of a judgment.",
    "ATTESTATION": "The formal authentication of an act or instrument by a subscribing witness or an official.",
    "ATTRACTIVE NUISANCE": "Doctrine that requires a landowner to exercise reasonable care in maintaining an inherently dangerous instrumentality if such is easily accessible to trespassing children.",
    "AUTHENTICATION": "Proof of extrinsic evidence as a condition precedent to admissibility to ensure that a matter in question is what its proponent claims it to be.",
    "AUTOMATIC STAY": "Procedure that automatically delays the execution or enforcement of a civil judgment until the expiration of ten days after the later of its entry or the disposition of a motion for a new trial.",
    "AUTRE DROIT": "Acting in the right of another.",
    "AVERMENT": "A formal assertion stating a claim or defense.",
    "BAD FAITH": "Lack of honesty; intentional dishonest action.",
    "BAD FAITH REFUSAL TO PAY INSURANCE CLAIM": "An action against an insurer alleging that the insurer acted without reason and fair dealing in refusing to pay an insured’s claim.",
    #"BAIL": "Security required by the court to guarantee the defendant’s presence at trial.",
    "BAIL BOND": "A contract between a licensed surety and the defendant for the payment of bail.",
    "BAILEE": "One who receives personal property under a bailment.",
    "BAILIFF": "A court attendant whose primary dutiesvinclude keeping order in the courtroom and attending to the jury.",
    "BAILMENT": "An express or implied contract for the storage or safekeeping of personal property.",
    "BAILOR": "One who delivers personal property under a bailment.",
    "BANC": "The place where a court regularly conducts judicial business.",
    "BARRISTER": "An English trial lawyer. Compare, SOLICITOR.",
    "BASTARD": "An illegitimate person; one born out of wedlock.",
    "BATSON HEARING": "Legal proceeding that requires the trial court to determine whether the exercise of a peremptory strike was purposefully discriminatory.",
    "BATTERY": "1. Civil. Any unlawful and willful use of force or violence on the person of another. 2. Criminal. Simple and aggravated assault are defined under Miss. Code Ann. Section 97-3-7.",
    "BENCH WARRANT": "Process issued by the court itself for the arrest of someone.",
    "BEQUEATH": "To give personal property to another by a will.",
    "BEQUEST": "A gift of personal property by a will; a legacy.",
    "BEST EVIDENCE RULE": "Primary proof, as distinguished from secondary; original as distinguished from a copy; applied only to documents, never to testimony.",
    "BEYOND A REASONABLE DOUBT": "Proof to the exclusion of every reasonable hypothesis except that of guilt.",
    "BIFURCATED": "Latin: “Two-pronged.” A trial or hearing separated into distinct phases, usually as to issues of guilt and punishment, to safeguard against undue prejudice or otherwise ensure justice.",
    "BILL OF EQUITY": "The initial pleading in former Mississippi chancery practice. See, COMPLAINT.",
    "BILL OF EXCEPTIONS": "1. An appellate record, especially of a judgment or decision of municipal authorities. 2. A formal statement of objections to be included in the record.",
    "BINDING INSTRUCTION": "Directions to the jury that if it finds certain conditions to be true, it must find for the plaintiff or the defendant, as the case may be.",
    "BIND OVER": "To hold an accused for action by the grand jury after a finding at a preliminary hearing that there is probable cause to believe that the accused committed a crime.",
    "BLOG": "A website where the author writes about subjects such as news, politics, or the legal system; it is displayed in reverse chronological order.",
    "BONA FIDE": "Latin: “In good faith.” Genuine.",
    "BOUNDARY": "The physical limits of a parcel of real estate as described in a deed; a property line.",
    "BOUNDARY LINE AGREEMENT": "An agreement by and between adjacent landowners fixing the property line.",
    "BREACH OF CONTRACT": "A failure without legal excuse to perform any promise which forms the whole or part of a contract.",
    "BRIEF": "A document presented to the court that contains facts and law supporting a client’s position.",
    "BURGLARY-BREAKING": "Any act or force, however slight, used to unlawfully enter a structure.",
    "BURGLARY-ENTERING": "The act of unlawfully making one’s way into a structure.",
    #"BURDEN OF PROOF": "Standard of requisite proof necessary to prevail on the merits of the case. Criminal law requires proof beyond a reasonable doubt. Civil law ordinarily requires proof by a preponderance of the evidence.",
    "BURGLARY": "Breaking and entering the dwelling house or inner door of such dwelling house of another, whether armed with a deadly weapon or not, and whether there shall be at the time some human being in such dwelling house or not, with intent to commit some crime therein.",
    "CALLING THE DOCKET": "The public calling of the list of pending cases at the beginning of a court term for the primary purpose of setting trial dates, entering orders, or hearing preliminary motions.",
    "CANONS OF ETHICS": "Standards of ethical conduct governing ju  dges and lawyers.",
    "CAPACITY": "The legal qualification or ability to sue or be sued or be brought into court.",
    "CAPIAS": "Latin: “To seize.” An arrest warrant, especially if issued from the bench or on an indictment.",
    "CAPITAL OFFENSE": "Offense which is punishable by death or life imprisonment.",
    "CAPTION": "The heading or introductory part of a pleading, motion or other legal document which shows the names of the parties, name of the court, title of the action, file number, etc.",
    "CARELESS DRIVING": "Driving a vehicle in a careless or imprudent manner, without due regard for the width, grade, curves, corner, traffic and use of the streets and highways and all other attendant circumstances.",
    "CARNAL KNOWLEDGE": "Sexual intercourse; the slightest penetration by the male’s sexual organ of a female’s sexual organs.",
    "CASE": "A legal dispute brought into court; a lawsuit.",
    "CAUSE OF ACTION": "The legal basis for bringing a lawsuit.",
    "CAVEAT ACTOR": "Latin: “Let the doer beware.”",
    "CAVEAT EMPTOR": "Latin: “Let the buyer beware.”",
    "CERTIFICATION": "1. Order to transfer jurisdiction from youth court to circuit court upon conducting a bifurcated transfer hearing. 2. The act of attesting.",
    #"CERTIORARI": "A discretionary review of a lower court decision confined to questions of law arising or appearing on the face of the record and proceedings.",
    "CESTUIQUE (QUI) TRUST": "Beneficiary of a trust; one having equitable and beneficial interest in an estate, the legal title to which is vested in another.",
    "CESTUIQUE VIE": "One whose life measures the duration of an estate, trust, gift or insurance contract.",
    "CHAIN OF CUSTODY": "Proving that the integrity of evidence has not been compromised, i.e., no indication or reasonable inference of probable tampering with the evidence or substitution of the evidence, by showing continuous custodial possession.",
    "CHAIN OF TITLE": "Conveyances or other property transfers, arranged consecutively, from the government or original source of title down to the present holder.",
    "CHALLENGE FOR CAUSE": "Requesting the court to exclude a prospective juror whose answers and demeanor on voir dire clearly indicate an inability to fairly try the case.",
    "CHALLENGE TO THE ARRAY": "Questioning the qualifications of an entire panel summoned for jury duty, usually because of some deficiency in the manner in which the panel was selected and summoned.",
    "CHAMBERS": "The private office or room of a judge.",
    "CHAMPERTY": "The purchase of an interest in something in dispute in order to maintain or take part in litigation concerning it; illegal in Mississippi.",
    "CHANCELLOR": "A judge of the chancery court.",
    "CHANCERY COURT": "Established by the Mississippi Constitution with jurisdiction to hear, among other issues: all matters in equity; divorce and alimony; matters testamentary and of administration; minor’s business; cases of idiocy, lunacy, and persons of unsound mind; and real property disputes.",
    "CHANGE OF VENUE": "The removal of a case from one venue to another for trial.",
    "CHARGE": "An accusation of a crime by a formal complaint, information, or indictment.",
    "CHASTE": "No previous instances of consensual sexual intercourse.",
    "CHATTEL": "An article of personal property.",
    "CHILD": "One who has not reached the legal age of majority.",
    "CHILD ABUSE": "Causing or allowing the sexual abuse, sexual exploitation, emotional abuse, mental injury, nonaccidental physical injury or other maltreatment upon a child in one’s legal custody or care.",
    "CHILD NEGLECT": "Neglecting or refusing to provide for the necessary physical, medical, or educational needs of a child in one’s legal custody or care.",
    "CHILD SUPPORT": "Court-ordered periodic payments of funds for the support of a child.",
    "CIRCUIT COURT": "Established by the Mississippi Constitution with jurisdiction to hear all matters civil and criminal not exclusively cognizable in some other court.",
    #"CIRCUMSTANTIAL EVIDENCE": "Evidence which, without going directly to prove the existence of a fact, gives rise to a logical inference that such fact exists.",
    "CITATION": "1. Reference to an authority, e.g., a case or statute, that supports the textual statement or from which a quote is taken. 2. Often used as a synonymous term for traffic ticket or summons.",
    "CIVIL ACTION": "A lawsuit.",
    "CIVIL LAW": "Substantive and procedural laws pertaining to civil matters.",
    "CIVIL RIGHTS": "Personal rights guaranteed and protected by the U.S. Constitution or federal law.",
    "CLAIM": "1. Any demand to recover damages from a governmental entity as compensation for injuries. 2. A right to payment, whether or not the right is reduced to judgment, liquidated, unliquidated, fixed, contingent, matured, unmatured, disputed, undisputed, legal, equitable, secured, or unsecured.",
    #"CLAIMANT": "One asserting a claim through a civil action.",
    "CLEAR AND CONVINCING EVIDENCE": "That weight of proof which produces in the mind of the trier of fact a firm belief or conviction as to the truth of the allegations sought to be established; evidence so clear, direct and weighty and convincing as to enable the fact finder to come to a clear conviction, without hesitancy, of the truth of the precise facts of the case.",
    "CLEAR TITLE": "A good and marketable title to real property; a title free from encumbrance, burden, or limitation.",
    "CLOSING": "Final steps in a real estate transaction where consideration is paid, the mortgage is secured, and the deed is delivered.",
    "CLOSING ARGUMENT": "Argument before a jury summarizing the evidence presented at trial, along with any reasonable deductions and conclusions.",
    "CLOUD ON TITLE": "The semblance of an outstanding claim or encumbrance that casts a doubt as to the validity of the record title. A legal recourse is to attain a decree in chancery court that removes the cloud on title.",
    "CODE": "An authorized collection, compendium or revision of laws systematically arranged into titles, chapters, and sections.",
    "CODICIL": "A supplement or addition to a will.",
    "COLLUSION": "A secret agreement between two or more persons for fraudulent or deceitful purposes.",
    "COMMIT": "1. To order a person to a state institution such as a prison or mental health facility. 2. To engage in a criminal act.",
    "COMMON CARRIER": "A person or business that carries or transports people or property for money.",

    "COMMON LAW": "Law derived from the English legal system applicable as precedent in the absence of overriding Constitutional law or legislative enactments.",
    "COMMON LAW MARRIAGE": "A marriage not solemnized by legal ceremony, but instead created by an agreement to marry followed by cohabitation. Mississippi does not recognize common law marriages contracted after April 5, 1956.",
    "COMMUNITY PROPERTY": "A statutory mandate requiring an equal division of all marital property upon divorce regardless of the respective contributions or circumstances. Not applicable in Mississippi. Instead, our state applies a system of equitable distribution.",
    "COMMUTATION": "Reduction in severity of a previously imposed penalty, e.g., commuting a death sentence to life imprisonment.",
    "COMPARATIVE NEGLIGENCE": "Legal principal, now codified in Mississippi, that permits the jury to diminish personal injury damages in proportion to the amount of negligence attributable to the injured party.",
    "COMPETENCY": "1. The mental capacity to understand the nature and effects of one’s actions. A defendant in a criminal case must have a rational and factual understanding of the proceedings to enter a plea or to stand trial. 2. Evidentiary finding that a witness is legally qualified to give testimony.",
    "COMPLAINT": "The first pleading which begins a civil action.",
    "COMPOSITION": "Agreement whereby a creditor accepts the immediate payment of a percentage of the total amount owed as discharge of the entire debt.",
    "CONCLUSIVE EVIDENCE": "Evidence that is incontrovertible or from which only one reasonable conclusion can be drawn taking all the facts and surroundings into consideration. It is synonymous with manifest, plain, clear, and obvious.",
    "CONCUBINE": "A woman who lives with a man without the benefit of marriage.",
    "CONDEMNATION": "Legal process by which real property of a private owner is taken for public use upon the award of due compensation. Compare, EMINENT DOMAIN.",
    "CONDONATION": "Defense in a divorce proceeding on grounds that the offended spouse either expressly or impliedly forgave, upon a condition of future good behavior, a marital wrong. Merely not leaving the marital domicile or separating from the offending spouse does not constitute a condonation.",
    "CONFIRMATION OF TITLE": "An action in chancery court to clear title in the ownership of real property.",   
    "CONFLICTS OF LAW": "Differences or inconsistencies between the laws of different jurisdictions., and the rules for determining which jurisdiction’s law applies in a given case.",
    "CONNIVANCE": "The secret or indirect consent or permission of one person to the commission of an unlawful or criminal act by another; an intentional failure to discover or prevent the wrong.",
    "CONSANGUINITY": "Kinship; blood relationship; the connection or relation of persons descended from the same stock or common ancestor.",
    "CONSENT DECREE": "A decree agreed upon by the parties as a way to settle a controversy, or substantial part of it, without further litigation.",
    "CONSERVATOR": "1. One appointed by the chancery court to manage the estate of a person who is found incapable of doing so for reasons of advanced age, physical incapacity, or mental weakness.",
    "CONSIDERATION": "The inducement to a contract; the promise, price or other value given to persuade another to enter into the contract.",
    "CONSORTIUM": "A protected interest arising out of the marriage covenant such as society, companionship, love, affection, aid, services, support, and sexual relations.",
    "CONSORTIUM, LOSS OF": "A claim for damages as a result of a party’s loss of his or her spouse’s consortium.",
    "CONSPIRACY": "When two (2) or more people agree and plan to commit a crime.",
    "CONSTABLE": "As set forth in Miss. Code Ann. Section 1919-5, an officer whose responsibilities include preserving the peace, aiding and assisting in executing the criminal laws of the state, and serving process.",
    "CONTRACT-ACCEPTANCE": "An agreement to the conditions or terms stated in an offer.",
    "CONTRACT-OFFER": "A proposal to enter into a contract; an offer contains conditions or terms.",
    "CONTEMPT": "Conduct or words disruptive to the orderly administration of justice.",
    "CONTINUANCE": "Postponement of a court proceeding to a later date.",
    "CONTRABAND": "Property subject to lawful seizure.",
    "CONTRA BONOS MORES": "Latin: “Against good morals.”",
    "CONTRACT": "A legally enforceable exchange of promises.",
    "CONTRIBUTORY NEGLIGENCE": "Defense to negligence action barring recovery if injured person was partially at fault in proximately contributing to the injury. Not applicable in Mississippi.",
    "CONTROLLED SUBSTANCE": "Any substance regulated by law as to its possession and use.",
    "CONVERSION": "The unauthorized exercise of ownership over personal property belonging to another.",
    "CONVEYANCE": "The transfer of title to property from one person to another; the written instrument which effects the transfer of title.",
    "CONVICT": "1. To find one guilty of a criminal charge as a result of a trial or plea. 2. A prisoner.",
    #"CONVICTION": "An adjudication of guilt.",
    "COPYRIGHT": "The right to a particular expression of ideas, such as literature, music, art, etc.; the right to control its reprinting.",
    "CORPUS": "Latin: “The body.” The principal substance of a thing; the principal of a fund or estate, as opposed to interest, income, dividends or the like.",
    "CORPUS DELECTI": "Latin: “The body of the crime.” Essential facts which must be established by the prosecution in a criminal case to prove that a crime has in fact been committed.",
    "CORPUS JURIS": "Latin: “The body of the law.” A comprehensive collection of the law of a country or jurisdiction.",
    "CORROBORATING EVIDENCE": "Proof which supplements that already given and which tends to confirm or strengthen it.",
    "COSTS BILL": "The payment that must accompany the filing of a complaint in a civil action.",   
    "COUNSEL": "1. An attorney. 2. To provide legal advice.",
    "COUNT": "1. Civil. Each separate claim in the pleadings. 2. Criminal. Each separate charge in an indictment.",
    "COUNTERCLAIM": "A claim asserted by the defendant in a civil action.",
    "COUNTY COURT": "A statutorily created court with jurisdiction to hear both civil actions up to $200,000 and misdemeanor offenses. County courts also serve as special courts of eminent domain.",
    "COURT OF RECORD": "Courts with inherent powers to correct clerical errors and enter judgments, along with incidental power to fine and imprison for contempt.",
    "COURT REPORTER": "One who makes a record of judicial proceedings using shorthand, steno-type machines, or electronic recording devices.",
    "CREDITOR": "One to whom a debt is owed.",
    "CRIMINAL LAW": "Substantive and procedural laws pertaining to felonies and misdemeanors.",
    "CROSS-CLAIM": "As set forth in M.R.C.P. 13, any claim by one party against a co-party arising out of the transaction or occurrence that is the subject matter either of the original action or of a counterclaim therein or relating to any property that is the subject matter of the original action.",
    "CROSS EXAMINATION": "Questioning a witness who has testified for the opposing side on direct examination. Ordinarily the purpose of cross examination is to discredit the witness’ perception, memory, narration, or sincerity through the use of leading questions.",
    "CULPABLE NEGLIGENCE": "Negligence of a degree so great as to be equal to a complete disregard or indifference to the safety of human life.",
    "CURATOR BONIS": "A guardian or trustee appointed to take care of property.",
    "CUSTODIAL INTERROGATION": "Questioning initiated by law enforcement officers of a person in custody. “In custody” means that from the “totality of the circumstances” a reasonable person would feel arrested as opposed to being temporarily detained.",
    "CUSTODY": "1. Restraint of a person to the extent of constituting an arrest. 2. Care and supervision over a person or thing.",
    #"DAMAGES": "Monetary amount recoverable in a civil action to compensate one who has suffered loss, detriment, or injury.",
    "DEADLY WEAPON": "An object or weapon reasonably capable of producing death or serious bodily injury.",
    "DEBTOR": "One who owes a debt.",
    "DECEDENT": "A dead person.",
    "DECISION": "1. The court’s findings of fact and conclusions of law. 2. The court’s written disposition of a case.",
    #"DECREE": "An equitable decision or order of a chancery court.",
    "DECREE PRO CONFESSO": "An entry of default as provided in M.R.C.P. 55.",
    "DEED": "A conveyance of realty; a written instrument transferring title and ownership of real property. See also, QUITCLAIM DEED; WARRANTY DEED.",
    "DEED OF TRUST": "An instrument by which legal title to real property is transferred to one or more trustees to secure the payment of money or the performance of other conditions. Compare, MORTGAGE.",
    "DE FACTO": "Latin: “In fact.” Acting or existing without specific legal authority, but of recognized legal effect or consequences. Compare, DE JURE.",
    "DEFALCATION": "A misuse of funds, especially as it pertains to public or corporate accounts.",
    "DEFAMATION": "A false statement that harms another’s reputation. Libel is a written defamatory statement, while slander is a spoken one.",
    "DEFAULT": "Occurs when a party against whom a judgment for affirmative relief is sought fails to plead or otherwise defend the action.",
    "DEFAULT, ENTRY OF": "See, ENTRY OF DEFAULT JUDGMENT.",
    "DEFAULT JUDGMENT": "Judgment rendered as a result of a party’s default.",
    #"DEFENDANT": "1. Civil. One against whom a lawsuit is initiated. 2. Criminal. One accused of a crime.",
    "DEFICIENCY JUDGMENT": "A judgment in favor of a creditor for the difference between the amount of the debt owed and the amount received from a judicial sale.",
    "DELIBERATE DESIGN": "Unlawfully deciding to kill another without a legally justifiable or excusable reason.",
    "DELINQUENT ACT": "Any act, which if committed by an adult, is designated as a crime under state or federal law, or municipal or county ordinance other than offenses punishable by life imprisonment or death. A delinquent act includes escape from lawful detention, violations of the Uniform Controlled Substances Law, and violent behavior.",
    "DELINQUENT CHILD": "A child who has reached his tenth birthday and who has committed a delinquent act; a child adjudicated delinquent by the youth court.",
    "DEMAND NOTE": "A note that becomes due and payable as of the date of execution, no demand being necessary.",
    "DEMURRER": "1. Civil. No longer applicable in Mississippi to civil cases. Such is now understood to mean a motion to strike as set out in M.R.C.P. 12(f). 2. Criminal. A defendant’s formal objection to an alleged defect in an indictment.",
    "DE NOVO": "Latin: “Anew.” A new trial or hearing as if the original trial or hearing had not taken place.",
    "DE NOVO Latin": "“Anew.”",
    "DEPONENT": "One who gives a deposition.",
    "DEPOSE": "To give sworn testimony at an informal proceeding, usually without the presence of a judge; the act of obtaining such testimony.",
    "DEPOSITION": "Sworn testimony given in accordance with the rules of discovery.",
    "DEPRAVED HEART": "Acting in a highly dangerous way that shows a lack of care for the safety of human life.",
    "DERAIGN": "Tracing the history of a land title, beginning with the grant by the government and concluding with the last recorded conveyance of the property. Compare, ABSTRACT OF TITLE.",
    "DETENTION": "1. The care of children in physically restrictive facilities. 2. The temporary care of juveniles and adults who require secure custody for their own or the community's protection in a physically restrictive facility prior to adjudication, or retention in a physically restrictive facility upon being taken into custody after an alleged parole or probation violation. 3. A brief restraint by law enforcement of one’s liberty or freedom.  ",

    "DEVISE": "A gift of real property under a will.",
    "DEVISEE": "One given real property under a will.",
    "DEVISOR": "One who gives real property by means of a will.",
    "DICTUM": "A comment or remark in an appellate decision having persuasive or suggestive influence but not binding as legal precedent.",

    "DIRECT CONTEMPT": "Contempt committed in the presence of the judge presiding in court or so near the judge as to interrupt the court’s proceedings.",
    #"DIRECT EVIDENCE": "1. Evidence which, if believed, proves the fact without inference or presumption. 2. Evidence not circumstantial,",
    "DIRECT EXAMINATION": "Questioning of one’s own witness at trial who afterwards is subject to cross examination. Ordinarily direct examination precludes the use of leading questions. Compare, CROSS EXAMINATION.",
    "DISABILITY": "1. Any physical, mental or neurological impairment which severely restricts a person's mobility, manual dexterity or ability to climb stairs; substantial loss of sight or hearing; loss of one or more limbs",
    "DISCOVERY": "Procedures whereby each party, to avoid unfair surprise at trial, may discover beforehand certain information accessible to the opposition. Examples of discovery in civil cases includes depositions, written interrogatories, production of documents or things, and admissions.",
    "DISCOVERY CONFERENCE": "A conference held by the court in a civil action for the purpose of: fixing the issues to be tried; establishing a plan and schedule of discovery; setting limitations upon discovery, if any; and determining such other matters, including the allocation of expenses, as are necessary for the proper management of discovery in the case.",
    "DISMISSAL": "Procedure that concludes a civil action prior to a trial on the merits. A voluntary dismissal is ordinarily without prejudice. An involuntary dismissal, unless otherwise specified in the court order, is ordinarily with prejudice.",
    "DISMISSAL ON CLERK’S MOTION": "Procedure that allows clerk, upon proper notice to the attorneys of record, to move for the dismissal of a civil action in which there has been no action for twelve months. A subsequent dismissal by the court is without prejudice.",
    "DISMISSAL WITH PREJUDICE": "A dismissal that operates as an adjudication upon the merits of the case. Such precludes a refiling of the claim.",
    "DISMISSAL WITHOUT PREJUDICE": "A dismissal that does not operate as an adjudication upon the merits. Such does not preclude a refiling of the claim.",
    "DISORDERLY CONDUCT": "Offensively disruptive behavior constituting a breach of the public peace and safety.",
    "DISTRIBUTION": "Apportionment and division of an intestate’s estate to the rightful heirs after payment of the estate’s debts and charges.",
    "DIVERSION": "Unauthorized use of funds.",
    "DIVERSION PROGRAM": "See, PRETRIAL INTERVENTION PROGRAM.",
    "DIVORCE": "Legal termination of a marriage.",
    "DOCKET": "A chronological log of activities maintained by the clerk on each civil action or criminal case.",
    "DOMESTIC ANIMAL": "An animal that is customarily owned or used by people.",
    "DOMESTIC DOCUMENT": "An official record, or entry in it, kept within the United States or any state, district, commonwealth, territory, etc.",
    "DOMICILE": "One’s primary place of abode and to which, upon a departure therefrom, there is a present intention of returning.",
    "DOUBLE JEOPARDY": "Constitutional protections prohibiting: a second prosecution after acquittal; a second prosecution after conviction; and multiple punishments for the same offense. In Mississippi, jeopardy attaches when a jury is empaneled and sworn, or for a non-jury trial, when the first witness is sworn.",
    "DRUG": "Any substance recognized or designated as a drug by law. Such would include controlled substances and over-the-counter medicines.",
    "DUE PROCESS": "Rules of procedure necessary to ensure a fair and just trial.",
    "DURANTE VIVA": "Latin: “During life.”",
    "DWELLING HOUSE": "A structure where one lives or where one intends to live.",
    "EASEMENT": "A right or interest in real property for use of a particular purpose, e.g., a right of way. It may be created by grant, implication, or prescription.",
    "EJECTMENT": "A civil action to recover possession of land and damages from one in unlawful retention.",    
    "EJUSDEM GENERIS": "Latin: “Of the same kind.”",
    "ELECTRONIC DISCOVERY": "(E-DISCOVERY) Discovery of data and information stored electronically.",
    "ELECTRONIC FILING": "(E-FILING) Documents filed in an electronic format.",
    "ELECTRONIC MAIL": "(E-MAIL) Electronic communications conveyed to an addressee by means of a computer or like device.",
    "EMINENT DOMAIN": "Power of a governmental entity, subject to the award of due compensation, to take the real property of a private owner for public use. Compare, CONDEMNATION.",
    "ENTRY OF DEFAULT JUDGMENT": "A default entered by the clerk when a party against whom a judgment is sought has failed to plead or otherwise defend the action. Such precedes a default judgment.",
    "EQUITABLE ESTOPPEL": "Doctrine that forbids one who, by words or conduct, induced another to detrimentally rely upon a material fact from later taking a contrary position.",
    "EQUITY": "That system of justice which was administered by the high court of chancery in England. Courts of equity proceed according to equitable rules and principals not available to courts of law, e.g., the clean hands doctrine, laches, etc.",
    "ESCAPEE": "One who escapes from lawful incarceration, confinement or custody.",
    "ESCHEAT": "Reversion of property to the state when a person dies without any heirs.",
    "ESCROW": "Conditional delivery of something to a third party to be held until the occurrence of some event or the performance of some act.",
    "ESTATE": "The total interest one has in real and personal property.",
    "ET ALIA (ET AL.)": "Latin: “And others.”",
    "ET SEQUENTES (ET SEQ.)": "Latin: “And the following.”",
    "ET UXOR (ET UX.)": "Latin: “And wife.”",
    "ET VIR": "Latin: “And husband.”",
    "EVICTION PROCEEDING": "Civil action to remove the tenant from the premises.",
    #"EVIDENCE": "Proof such as testimony and tangible objects offered during a trial or hearing for the purpose of proving or disproving some fact.",
    "EXCLUSIONARY RULE": "Rule that excludes from the prosecutor’s case-in-chief the admissibility of evidence directly or derivatively obtained by exploitation of an illegal search or seizure.",
    "EX CONTRACTU": "Latin: “From a contract.” Rights and claims arising from a contract.",
    "EXCULPATORY": "Tending to clear or excuse from fault or guilt.",
    "EX DELICTO": "Latin: “From a wrong.” Rights and claims arising from a tort.",
    "EXECUTOR": "One named in a will whose duty is to carry out its provisions.",
    "EXEMPLI GRATIA (E.G.)": "Latin: “For example.”",
    #"EXHIBIT": "A document or other tangible evidence produced during a trial or hearing.",
    "EX MERO MOTO": "Latin: “On the court’s own motion.” A phrase often occurring in grants, charters, etc.",
    "EX OFFICIO": "Latin: “By virtue of the office.”",
    "EXONERATE": "To relieve of liability; to excuse.",
    "EX PARTE": "Latin: “By one side.” Proceeding in which only one party is being heard.",
    "EXPERT TESTIMONY": "Testimony relating scientific, technical, or other specialized knowledge by one qualified to do so.",
    "EX POST FACTO": "Latin: “After the fact.”",
    "EXPUNGE": "To erase, as authorized by law, information contained in a record.",
    "EX RELATIONE (EX REL.)": "Latin: “By or on the relation of.”",
    "EXTENUATING CIRCUMSTANCES": "Unusual circumstances supporting a position for leniency.",
    "EXTRADITION": "Summary procedure for the surrender of a fugitive to the authorities of the offended jurisdiction.",
    "EYEWITNESS": "One who actually saw a particular event as it took place.",
    "FACSIMILE": "An exact copy or reproduction of something.",
    "FAILURE OF SERVICE": "Inability to serve a copy of the summons on a defendant.",
    "FALSE ARREST": "An arrest not authorized by law.",
    "FALSE IMPRISONMENT": "When one holds or imprisons another unlawfully or when one causes another to do so.",
    "FALSE PRETENSES": "Knowingly making a false representation of a material fact thereby obtaining something of value without compensation.",
    "FAMILY MASTER": "A qualified person appointed to hear certain referred cases involving support and paternity matters.",
    "FEE SIMPLE": "Absolute ownership of real property, usually with unconditional powers of disposition.",
    "FELONY": "An offense punishable by death or confinement in the penitentiary; an indictable offense.",
    "FEME COVERT": "A married woman.",
    "FEME SOLE": "An unmarried woman.",
    "FIAT": "Latin: “Let it be done.” A short order or warrant of a judge or other competent authority directing some legal act to be done.",
    "FIERI FACIAS": "Latin: “Cause to be done.” Directing an execution to be levied on the goods of a judgment debtor.",
    "FILE": "1. To deliver a document to the clerk for filing into the official record. 2. The official record of a case kept and preserved as provided by law.",
    "FILING FEES": "Fees paid to the clerk of the court upon initiating a civil action.",
    "FILING OF JUDGMENT": "Delivery of the judgment to the clerk for filing into the official record.",
    "FINE": "A monetary punishment or penalty.",
    "FINIS": "Latin: “The end.”",
    "FLAGRANTE DELICTO": "Latin: “While the offense is blazing.”Caught in the act of committing the offense.", 
    "FORECLOSURE": "To shut out; a termination of the borrower’s rights in property covered by a mortage.",
    "FOREIGN CORPORATION": "A corporation created under the laws of another state, government or country.",
    "FORENSIC": "Describing a discipline readily applicable to evidentiary matters, e.g., forensic medicine, forensic chemist, forensic pathologist, etc.",
    "FORFEITURE": "1. The failure to recognize and assert a right. 2. A divestiture of specific property without compensation as the consequence of some default or act forbidden by law.",
    #"FORGERY": "To falsely make or materially alter a document with intent to defraud.",
    "FORUM": "A place where issues are litigated and resolved; a jurisdiction; a court.",
    "FRAUD": "Knowingly misrepresenting a material fact to induce another to detrimentally act upon it in the manner reasonably contemplated.",
    "FRONTAGE": "That portion of real property abutting a street or road.",
    "FUGITIVE": "One who flees from justice upon being charged with a criminal offense. Compare, ESCAPEE.",
    "FUGITIVE WARRANT": "A warrant for the arrest of one who has fled to another state to avoid prosecution.",
    "FULL FAITH AND CREDIT": "U.S. Constitutional requirement that each state fully recognize and enforce all legitimate and final judgments of other states and federal courts.",
    "FUTURE INTERESTS": "An existing interest in real or personal property, ordinarily freely transferable, in which the privilege of possession or enjoyment is future and not present.",
    #"GARNISHEE": "One upon who a writ of garnishment is served.",
    "GARNISHMENT": "Statutory process of enforcing a judgment by attaching monies or property owed to the defendant, such as employment wages.",
    "GENERAL VERDICT": "A verdict requiring no special form.",
    "GIFT": "A voluntary transfer of property.",
    "GIFT CAUSA MORTIS": "A gift made in prospect of imminent death.",
    "GOOD FAITH": "Honest and trustworthy action.",
    "GOVERNMENTAL IMMUNITY": "Exemption from civil liability of a governmental entity absent its consent.",
    "GRAND JURY": "Impaneled group of men and women convened to determine whether probable cause exists to return an indictment. Compare PETIT JURY.",
    "GRANT": "To agree to; to make a concession; to convey, especially real property.",
    "GRANTEE": "One to whom a grant is made.",
    "GRANTOR": "One who makes a grant.",
    "GRANTOR/GRANTEE INDEX": "Index to property titles in which the records are kept by reference to grantor’s and grantee’s names.",
    "GRATUITOUS": "Without valuable or legal consideration.",
    "GRAVAMEN": "The principle or most important part of a complaint or argument.",
    "GROSS": "1. Total amount, e.g., gross earnings. 2. Flagrant or shameful.",
    "GROSS NEGLIGENCE": "Negligence of a degree so great that it shows a reckless disregard for the safety or rights of others.",
    "GUARDIAN": "Legally recognized custodian of the person or property of another with prescribed fiduciary duties and responsibilities under court authority and direction.",
    "GUARDIAN AD LITEM": "A representative of the court appointed to assist in properly protecting the best interests of a child or incompetent person by means of investigations, recommendations, and reports.",
    "GUARDIANSHIP": "The legal relationship created by the appointment of a guardian to care for the person or property of another.",
    "HABEAS CORPUS": "Latin: “You shall have the body.” A writ used to bring a prisoner before the court to determine whether the prisoner’s detention is lawful.",
    "HABEAS CORPUS PETITION": "A petition filed by a prisoner seeking relief from unlawful detention.",
    "HABENDUM CLAUSE": "A clause in a deed defining the extent of ownership in the thing granted. Such is usual in a mineral deed.",
    "HARMLESS ERROR": "Error of insufficient prejudicial effect to warrant a reversal. Such is apparent if a fair minded juror, even if the error were rectified, could only have arrived at a  verdict of guilt.",
    "HEARING": "A legal proceeding before the court in which testimony is presented, e.g., preliminary hearing, plea hearing, suppression hearing, trial, sentencing, etc.",
    #"HEARSAY": "A statement that: (1) the declarant does not make while testifying at the current trial or hearing; and (2) a party offers in evidence to prove the truth of the matter asserted in the statement.” M.R.E. 801(c).",
    "HEAT OF PASSION": "Unlawfully acting in a state of violent and uncontrollable rage.",
    "HEIR": "One who inherits or receives property from another who has died; such person may be male (heir) or female (heiress).",
    "HOLIDAY": "See, LEGAL HOLIDAY.",
    "HOLGAPHIC WILL": "A last will and testament entirely in the handwriting of the maker and signed at the end; valid in Mississippi even when made without witnesses.",
    "HOMESTEAD EXEMPTION": "Statutory right of householder to hold exempt from seizure or sale, under execution or attachment, a certain value and acreage of the personal residence.",
    "HOMICIDE": "The killing of one human being by another.",
    "HOUSE ARREST": "The confinement of a person convicted or charged with a crime to that person’s residence under the terms and conditions established by the department of corrections or court.",
    "HUNG JURY": "A jury which after extensive deliberations cannot agree upon a verdict.",
    "HYPOTHECATE": "To guarantee a debt by pledging one’s property as security.",
    "IBIDEM": "Latin: “In the same place.”",
    "IDEM": "Latin: “The same.”",
    "ID EST": "Latin: “That is.”",
    "IMPEACHMENT": "An attack upon the credibility of a witness.",
    "IMMUNITY": "Exemption from liability. See also, ABSOLUTE IMMUNITY; QUALIFIED IMMUNITY; GOVERNMENTAL IMMUNITY.",
    "IN CAMERA": "Latin: “In chambers.” A judicial act done outside the presence of the public or jury.",
    "INCARCERATION": "Confinement to a jail or prison.",
    "IN CURIA": "Latin: “In the court.”",
    "INDEFEASIBLE": "That which is irrevocable; something   which cannot be defeated or voided; usually applied to ownership of an estate or right.",
    "INDICTMENT": "Formal charge of a felony returned by a grand jury.",
    "INDIGENT": "Poor; without funds.",
    "INDIRECT CONTEMPT": "An act done beyond the presence of the court which is calculated to impede, embarrass, obstruct, defeat or corrupt the orderly administration of justice.",
    "IN ESSE": "Latin: “In being.”",
    "IN EXTREMIS": "Latin: “In the extreme.” One who is near death and with no hope of recovery.",
    "INFAMOUS CRIME": "An offense punishable by death or confinement in the penitentiary; a felony.",
    "IN FORMA PAUPERIS": "Latin: “In the manner of a pauper.” One who for reasons of poverty is relieved from paying fees and costs of a lawsuit or appeal.",
    "INFORMATION": "Formal charge of a felony issued by a prosecutor. Such is permissible if the defendant validly waives the indictment.",
    "INFRA": "Latin: “Below.” Compare, SUPRA.",
    "INHERITANCE": "Property received by will or by law from someone who has died.",
    "IN HOC": "Latin: “In this.” In this respect.",
    "INJUNCTION": "An order issued by a court that requires someone to do or not to do something.",
    "IN LIMINE": "Latin: “At the threshold.” A motion in limine seeks a ruling at the beginning of a trial to exclude the use of certain evidence.",
    "IN LOCO PARENTIS": "Latin: “In place of the parent.”",
    "IN PARI DELICTO": "Latin: “In equal fault.”",
    "IN PERSONAM": "Latin: “Against the person.” The power of a court to hear claims for or against a particular person. Compare, IN REM.",
    "IN POSSE": "Latin: “In possible existence.”",
    "INQUEST": "A legal inquiry into the circumstances of the death of a human being; generally held before a court of law or an official legally empowered to hold such inquiries.",
    "IN RE": "Latin: “In the matter of.”",
    "IN REM": "Latin: “Against the thing.” The power of a court to hear claims involving a particular thing or property. Compare, IN PERSONAM.",
    "IN SITU": "Latin: “In place.” In its original site or place.",
    "INSOLVENCY": "The condition of being unable to pay one’s debts.",
    "INTELLECTUAL PROPERTY": "Property originating from human intellect and the property rights obtained therein, such as copyright, patent, and trademark.",
    "INTENTIONAL INFLICTION OF EMOTIONAL DISTRESS": "A tort claiming that one has intentionally acted in an extreme and outrageous way and has caused another to have emotional distress.",
    "INTER ALIA": "Latin: “Among other things.”",
    "INTER ALIOS": "Latin: “Among other persons.”",
    "INTERLOCUTORY": "Provisional; temporary; not final.",
    "INTERNET": "The computer network of federal and nonfederal interoperable packet switched data networks.",
    "INTERPLEADER": "Procedure that permits a stakeholder of money or property to join potential claimants to avoid double or multiple liability.",
    "INTERROGATORIES": "Written questions served upon an opposing party to be returned with sworn answers.",
    "INTERVENOR": "One who by right or permission intervenes in a civil action.",
    "INTERVENTION OF RIGHT": "Procedure that gives one claiming an interest relating to the property or transaction of a civil action the right to intervene to protect such interest.",
    "INTER VIVOS": "Latin: “Between the living.” From one living person to another.",
    "INTESTATE": "Without a will.",
    "IN TOTO": "Latin: “In the whole.” Entirely.",
    "INVASION OF PRIVACY": "A tort claiming that one has intentionally  intruded on another’s right to privacy or seclusion; intentionally used another’s likeness or photograph for business purposes without permission.",
    "INVEIGLE": "To lure or entice.",
    "INVITEE": "One who enters premises at the express or implied invitation of the owner.",
    "INVITEE, BUSINESS": "One who enters and remains on a business’s property by express or implied invitation of the business owner.",
    "IPSO FACTO": "Latin: “By the fact itself.”",
    "IPSO JURE": "Latin: “By the law itself.”",
    "ISSUANCE": "Sending out orders or papers relating to the business of the court.",
    "JOINDER": "Procedure that permits one to join claims or persons to an action. Such allows for the efficient administration of justice by eliminating piecemeal litigation and balancing the rights of all persons whose interests are involved in an action.",
    "JOINT TENANCY": "Co-ownership of property by two or more persons with the right of survivorship. The deed or other instrument must clearly indicate the intent to create a joint tenancy with the right of survivorship, and not as tenants in common.",
    #"JUDGMENT": "A final decision or order from which an appeal may be taken; the final determination of an action.",
    "JUDGMENT NISI": "A judgment that becomes final upon compliance with certain statutory procedures",
    "JUDGMENT NOTWITHSTANDING THE VERDICT (JNOV)": "A judgment that sets aside the verdict.",
    "JUDICIAL NOTICE": "A court accepting into evidence, without requiring proof, an adjudicative fact not subject to reasonable dispute.",
    "JUDICIAL SALE": "A court ordered sale.",
    "JURAT": "Clause written at the bottom of an affidavit stating when, where and before whom the affidavit was sworn.",
    "JURISDICTION": "The power of the court to hear and decide a particular matter.",
    "JUROR": "A member of a jury.",
    "JURY COMMISSIONER": "An official responsible for selecting potential jurors.",
    "JURY INSTRUCTIONS": "Instructions given to the jury of the law pertaining to the case.",
    "JURY PANEL": "See, PANEL.",
    "JURY TAX": "Costs collected by the clerk or sheriff, as set forth in Miss. Code Section 9-7-133, as a fund for the payment of jurors.",
    "JUSTICE COURT": "Established by the Mississippi Constitution with limited criminal and civil jurisdiction, essentially misdemeanor offenses and small claims.",
    "KICKBACK": "Remuneration in return for unlawfully soliciting business.",
    "KIDNAPPING": "To seize or inveigle forcibly with intent to confine or imprison.",
    "KILL": "To terminate a life. Compare, MURDER.",
    "LACHES": "Equitable defense barring recovery if a party has inexcusably delayed in asserting a right or claim which thereby resulted in undue prejudice.",
    "LANDLORD": "1. As used in Mississippi’s Residential Landlord and Tenant Act, the owner, lessor or sublessor of the dwelling unit or the building of which it is a part, or the agent representing such owner, lessor or sublessor. 2. One with legal standing to sue the tenant for breach of the rental agreement.",
    "LAND PATENT": "A governmental conveyance of public land to a private individual.",
    "LARCENY": "Stealing the personal property of another.",
    "LAWSUIT": "A case or controversy brought before a court.",
    "LEAD COUNSEL": "The lawyer who is principally in charge of a case.",
    "LEASE": "An agreement to rent real or personal property, usually for a specified time; creates a legal relationship known as landlord and tenant or lessor and lessee.",
    "LEAVE OF COURT": "Permission of the court necessary to proceed in way that otherwise would be prohibited or limited by the rules of procedure.",
    "LEGACY": "A disposition of personal property by will.",
    "LEGAL": "Of or relating to the law.",
    "LEGAL HOLIDAY": "Days declared a legal holiday under Miss. Code Ann. Section 3-3-7 or as otherwise provided by law. The courthouse is officially closed on such days.",
    "LEGATEE": "One who receives personal property under a will.",
    "LESSEE": "Someone who leases or rents property, real or personal, from another. Compare, TENANT.",
    "LESSOR": "One whose property, real or personal, is rented or leased to another. Compare, LANDLORD.",
    "LETTERS OF ADMINISTRATION": "A formal document issued by the chancery court authorizing one to act as the administrator of a decedent’s estate.",
    "LETTERS OF CONSERVATORSHIP": "A formal document issued by the chancery court authorizing one to act as the conservator of an estate.",
    "LETTERS OF GUARDIANSHIP": "A formal document issued by the chancery court authorizing one to act as the guardian of a minor or specified ward of the court.",
    "LETTERS ROGATORY": "Procedure to obtain testimony from a witness residing in a foreign jurisdiction.",
    "LETTERS TESTAMENTARY": "A formal document issued by the chancery court authorizing one to act as the executor of a decedent’s estate.",
    "LEVY": "1. To impose a tax. 2. Legal process of satisfying a judgment by the seizure and sale of property.",
    "LEX Latin": "The law.",
    "LEX LOCI": "The law of the place. The local law or custom.",
    "LIBEL": "A written defamatory statement. Compare, DEFAMATION.",
    "LICENSE": "1. Permission by the applicable governing authorities to engage in certain activity or conduct upon meeting specific criteria. 2. Permissive use of land by which the owner allows another to come onto the owner’s land for a specific purpose.",
    "LICENSEE": "One who enters and remains on another’s property for one’s own benefit, with the owner’s consent or permission.",
    "LIEN": "A claim against property to secure a debt or other obligation.",
    "LIENHOLDER": "One holding a claim against property to secure a debt or other obligation.",
    "LIFE ESTATE": "An estate in which the duration is limited to a specified person’s lifetime, usually the possessor. Compare, PER AUTRE VIE.",
    "LIMITATION OF ACTIONS": "A time limit set by law within which certain legal actions must be brought.",
    "LINEUP": "A police identification procedure in which the suspect is presented alongside others of similar general appearance and stature. Compare, SHOWUP.",
    #"LIS PENDENS": "Latin: “A pending lawsuit.” A notice of lis pendens is filed of record to warn the public that certain property is involved in litigation.",
    "LITIGANT": "A party in a civil action.",
    "LITIGATION": "The lawsuit process.",
    "LIVESTOCK": "Animals, such as cattle and swine, produced for profit.",
    "LUCID INTERVAL": "A temporary restoration of sanity.",
    #"MAGISTRATE": "An informal term describing one authorized by law to perform judicial functions.",
    "MAKE A RECORD": "To preserve for appellate review an argument or proof.",
    "MALFEASANCE": "An act by a public official that is positively wrong or unlawful. Compare, MISFEASANCE, NONFEASANCE.",
   # "MALICE": "Intentionally acting wrongfully without having a valid reason or excuse.",
    "MALICE AFORETHOUGHT": "Deliberate design.",
    "MALICIOUS MISCHIEF": "When one intentionally and maliciously damages or destroys another’s property.",
    "MALICIOUS PROSECUTION": "A tort claiming that one has intentionally instituted a civil or criminal action against another without a reasonable basis for the action.",
    "MALPRACTICE": "Failure to provide the degree of care, skill and diligence expected of a minimally competent and reasonably prudent professional of the same specialty.",
    "MALPRACTICE, LEGAL": "Failure by an attorney to provide the degree of knowledge, skill, and diligence expected of a minimally competent and reasonably prudent attorney practicing in the same community.",
    "MALPRACTICE, MEDICAL": "Failure by a physician to act with the same degree of attention, skill, ability, and caution expected of a minimally competent and reasonably prudent physician practicing in the same medical specialty or general field of medicine.",
    #"MANDAMUS": "Latin: “We command.” An order issued by a higher court commanding an inferior tribunal, corporation, board, officer, or person to fulfill a specific responsibility.",
    "MANDATE": "1. A judicial, legislative, or executive command or directive. 2. A formal issuance by an appellate court of its decision.",
    "MANUFACTURE OF A CONTROLLED SUBSTANCE": "To unlawfully produce or prepare a controlled substance.",
    "MENS REA": "Latin: “Guilty knowledge.” With criminal intent.",
    "METADATA": "LITERALLY, DATA ABOUT DATA; INFORMATION ABOUT THE DATA SOUGHT AND ITS TYPE.",
    "MINUTE BOOK": "An official record of all significant court proceedings kept by the clerk.",
    "MIRANDA WARNINGS": "Warnings required to be given by law enforcement when subjecting a suspect to custodial interrogation, i.e., “You have the right to remain silent, . . . .”",
    "MISDEMEANOR": "A criminal offense punishable by a maximum possible sentence of confinement for one year or less, a fine, or both.",
    "MISFEASANCE": "Performing a legal duty in a wrongful manner.",
    "MISJOINDER": "The improper joining of a party in an action.",
    "MISSISSIPPI UNIFORM POST CONVICTION COLLATERAL RELIEF ACT": "(§ 99-39-1 et seq.) Exclusive and uniform procedure to review objections, defenses, claims, questions, issues or errors which could not be or should not have been raised at trial or on direct appeal.",
    "MISTRIAL": "A trial declared invalid by the court because of a fundamental error in the proceedings or the inability of the jury to reach a verdict.",
   # "MITIGATING CIRCUMSTANCES": "Facts and conditions which do not constitute a justification or excuse for an offense but which may be considered as reducing the degree of blame or fault.",
    "MITIGATION OF DAMAGES": "A doctrine that requires an injured party to take reasonable precautions to limit damages resulting from a tort or breach of contract.",
    "MITTIMUS": "Latin: “We send.” A writ to commit an offender to prison or to direct the transfer of records from one court to another. ",
    "MISJOINDER": "The improper joining of a party in an action.",
    "MISSISSIPPI UNIFORM POST CONVICTION COLLATERAL RELIEF ACT": "Exclusive and uniform procedure to review objections, defenses, claims, questions, issues or errors which could not be or should not have been raised at trial or on direct appeal.",
    "MISTRIAL": "A trial declared invalid by the court because of a fundamental error in the proceedings or the inability of the jury to reach a verdict.",
   # "MITIGATING CIRCUMSTANCES": "Facts and conditions which do not constitute a justification or excuse for an offense but which may be considered as reducing the degree of blame or fault.",
    "MITIGATION OF DAMAGES": "A doctrine that requires an injured party to take reasonable precautions to limit damages resulting from a tort or breach of contract.",
    "MITTIMUS": "Latin: “We send.” A writ to commit an offender to prison or to direct the transfer of records from one court to another.",
    "MODUS OPERANDI (M.O.)": "Latin: “Manner of operation.”",
    "MOOT": "A legal controversy rendered pointless because of a subsequent event; a theoretical or hypothetical issue.",
    "MORTGAGE": "A lien on real property to secure the performance of some obligation which is discharged upon payment or performance as required. Compare, DEED OF TRUST.",
    "MORTGAGEE": "One to whom the obligation on a mortgage or deed of trust is owed.",
    "MORTGAGOR": "The maker of a mortgage or a deed of trust; the one who owes the obligation on a mortgage.",
    "MOTION": "A formal application to the court seeking an order or relief.",
    "MOTION DAY": "A day designated by a court to hear motions.",
    "MOTION FOR JUDGMENT ON THE PLEADINGS": "A request for judgment based solely on the face of the pleadings.",
    "MOTION FOR MORE DEFINITE STATEMENT": "A request that a vague or ambiguous pleading to which a response is permitted be made more definite and specific.",
    "MOTION FOR ORDER COMPELLING DISCOVERY": "A request that the court force an opposing party to cooperate in discovery.",
    "MOTION FOR RELIEF FROM JUDGMENT OR ORDER": "A request to correct clerical mistakes in judgments or orders, or to seek relief from a final judgment, order or proceeding from errors such as fraud, newly discovered evidence, misrepresentation, etc.",
    "MOTION FOR SECURITY OF COSTS": "A request by the clerk or a party which, if granted, requires the plaintiff to deposit monies with the court to be used to pay the costs of the action if the plaintiff should not prevail.",
    "MOTION TO DISMISS FOR FAILURE TO STATE A CLAIM UPON WHICH RELIEF CAN BE GRANTED": "A request to dismiss a case on the basis that the plaintiff would not be entitled to relief even if all the facts alleged in the complaint were proved true.",
    "MOTION TO STRIKE": "A request that the court delete from a pleading any insufficient defense or material which is redundant, immaterial, impertinent or scandalous.",
    "MOTION TO SUPPRESS": "A request in a criminal case to keep certain facts or evidence from being brought out at trial.",
    "MOTION TO TERMINATE OR LIMIT EXAMINATION": "A request made by a party or the deponent during the taking of a deposition that the court end or restrict an examination that is being conducted in bad faith or in a manner calculated merely to annoy, embarrass or oppress.",
    "MULTIPLICITY OF ACTIONS": "Multiple litigation against a single defendant involving the same legal issue.",
    "MUNICIPAL COURT": "A statutory court with jurisdiction to hear and determine, without a jury, all cases charging municipal ordinance violations and state misdemeanor laws made offenses against the municipality.",
    "MURDER": "Murder is defined under Miss. Code Ann. Section 97-3-19(1). Generally, it is a deliberate or depraved killing of a human being without legal excuse or justifiable cause. Compare, KILL.",
    "MULTIPLICITY OF ACTIONS": "Multiple litigation against a single defendant involving the same legal issue.",
    "MUNICIPAL COURT": "A statutory court with jurisdiction to hear and determine, without a jury, all cases charging municipal ordinance violations and state misdemeanor laws made offenses against the municipality.",
    #"MURDER": "Murder is defined under Miss. Code Ann. Section 97-3-19(1). Generally, it is a deliberate or depraved killing of a human being without legal excuse or justifiable cause. Compare, KILL.",
    "NE EXEAT": "Latin: “Let him not go out.” A writ prohibiting a particular person from leaving the jurisdiction of the court.",
    "NEGLIGENCE": "A failure to act as a reasonably prudent person would act under similar circumstances.",
    "NEGLIGENCE, GROSS": "Negligence of a degree so great that it shows a reckless disregard for the safety or rights of others.",
    "NEGLIGENCE PER SE": "Negligence as a matter of law.",
    "NEXT FRIEND": "An adult who, in the absence of an appointed guardian, sues on behalf of an infant or incompetent person.",
    "NIL": "Latin: “Nothing.” A thing of no value.",
    "NO BILL": "Opinion of the grand jury that evidence is insufficient to warrant the finding of an indictment. Compare, TRUE BILL.",
    "NO FAULT": "A method of resolving disputes without considering the issue of fault.",
    "NOLLE PROSEQUI": "Latin: “I am unwilling to prosecute.” A formal dismissal of a criminal indictment.",
    "NOLLO CONTENDERE": "Latin: “I will not contest it.” A plea whereby the defendant neither admits nor denies guilt, but instead accepts a judgment of guilt by choosing not to contest the allegations underlying the charge.",
    "NOLO CONTENDERE": "Latin: “I will not contest it.” A plea whereby the defendant neither admits nor denies guilt, but instead accepts a judgment of guilt by choosing not to contest the allegations underlying the charge.",
    "NOMINAL DAMAGES": "A small and trivial sum awarded for a technical injury due to a violation of some legal right.",
    "NONADJUDICATION": "Withholding adjudication of guilt of an eligible defendant pursuant the statutory requirements.",
    "NON COMPOS MENTIS (N.C.M.)": "Latin: “Not of sound mind.”",
    "NONFEASANCE": "Failing to perform a legal duty.",
    "NON-JOINDER": "Failure to join a party needed for a just adjudication.",
    "NON SEQUITUR": "Latin: “It does not follow.” An unwarranted or illogical conclusion.",
    "NONSUIT": "A plaintiff’s voluntary dismissal of a lawsuit prior to an adjudication on the merits.",
    "NOTARY PUBLIC": "A bonded public officer who may administer oaths and affirmations, receive the proof or acknowledgment of all instruments of writing relating to commerce and navigation, and such other writings as are commonly proved or acknowledged before notaries.",
    "NOTICE": "Notification to a party or witness as required by law.",
    "NOTICE TO QUIT": "A written notice by the landlord demanding the tenant to quit the premises.",
    "NULLA BONA": "Latin: “Nothing collected.” A form of return by a sheriff or constable upon an execution when a judgment debtor has no seizable property within the jurisdiction.",
    "NUNC PRO TUNC": "Latin: “Now for then.” To supply omissions in the record of what had previously been done, but for reasons of mistake or neglect had not been entered.",
    "OATH": "A sworn pledge, e.g., an oath to tell the truth prior to giving testimony.",
    #"OBJECTION": "Protocol for requesting the trial court to rule on the admissibility of a particular question, statement, or exhibit.",
    "OFFENDER": "One charged or convicted of a crime under the laws of the State.",
    "OFFENSE": "A violation of a criminal law.",
    "OFFER OF JUDGMENT": "A formal offer to take an adverse judgment conditioned upon certain specified terms.",
    "OPEN ACCOUNT": "A type of credit extended through an advance agreement by a seller to a buyer which permits the buyer to make purchases without a note of security and is based on an evaluation of the buyer's credit.",
    "OPEN PLEA": "A plea in which the State does not make any recommendation regarding sentencing.",
    "OPINION TESTIMONY BY LAY WITNESSES": "Testimony by a non-expert that is rationally based on the perception of the witness, helpful to a clear understanding of the testimony, and not based on scientific, technical, or other specialized knowledge.",
    "ORDER": "A formal command of a court, usually in writing.",
    "ORDINANCE": "A municipal law.",
    "ORE TENUS": "Latin: “By word of mouth.” Orally.",
    "OUTRAGEOUS CONDUCT": "Conduct which exceeds all possible bounds of decency.",
    "PANEL": "A group of jurors chosen to serve in a specific court; those selected to hear a trial of a certain action; denotes either the whole body of persons summoned for a particular court term or those selected at random.",
    "PAR": "Latin: “Equal.”",
    "PARITY": "An equitable term denoting equality in amount, status or character.",
    "PARAMOUR": "A lover to whom one is not married.",
    "PARAPHERNALIA": "1. Personal belongings. 2. Any type of equipment or accessory utilized for illicit drug use.",
    "PARENS PATRIA": "Latin: “Parent of the country.” Doctrine that refers to a State’s sovereign power to act in protecting its more vulnerable citizens, such as children or incompetent adults.",
    "PARISH": "In Louisiana, the equivalent of what in Mississippi would be a county.",
    "PAROLE": "The conditional release of a prisoner.",
    "PAROL EVIDENCE RULE": "An evidentiary rule which forbids the introduction of oral evidence to modify the terms of a written contract.",
    "PARTITION": "The court supervised division of real or personal property.",
    "PARTY": "One who is directly involved in a lawsuit, e.g., plaintiff, defendant; appellant, appellee; petitioner, respondent; etc.",
    "PARTY WALL": "A wall constructed on a property line.",
    "PATENT": "Right held by patent holder that protects against the infringement of a particular invention or discovery.",
    "PATERNITY SUIT": "A court proceeding to prove the father of an illegitimate child.",
    "PAUPER’S OATH": "An affidavit seeking a waiver of costs and security for reasons of poverty.",
    "PENDENTE LITE": "Latin: “While the action is pending.”",
    "PER AUTRE VIE": "Latin: “For or during a period measured by another’s life.”",
    "PER CAPITA": "Latin: “By the head.” Share and share alike. A per capita distribution is an equal division of an estate among descendants who enjoy the same degree of kinship to the decedent. Compare, PER STIRPES.",
    "PER CURIAM": "Latin: “By the court.” A per curiam opinion is one that speaks in unison for all members of the court.",
    "PEREMPTORY CHALLENGE": "Requesting the court to exclude a prospective juror for reasons that are nondiscriminatory. Each side is afforded a limited number of peremptory challenges.",
    "PERJURY": "To deliberately make false statements under oath.",
    "PERMISSIVE INTERVENTION": "Procedure that permits, within the discretion of the court, one to intervene if asserting a claim or defense with a common question of law or fact in the civil action.",
    "PER SE": "Latin: “By itself.”",
    "PERSONAL PROPERTY": "Property that is not realty.",
    "PERSONAL RECOGNIZANCE": "Release of a defendant charge with a criminal offense without any condition relating to, or a deposit of, security.",
    "PERSONALTY": "Personal property.",
    "PERSONA NON GRATA": "Latin: “An unacceptable person.”",
    "PER STIRPES": "Latin: “By roots or stocks.” By representation. A per stirpes distribution is where a class or group of individuals or distributees take the share which their “stock” (deceased ancestor) would have been able to take in a per capita distribution.",
    "PETIT JURY": "The jury selected to hear the trial of a criminal or civil case. Compare, GRAND JURY.",
    "PHYSICAL EVIDENCE": "Tangible proof, e.g., document, x-ray, weapon, etc.; also called real evidence.",
    "PLAINTIFF": "One who initiates a legal action.",
    #"PLEA": "1. Civil. Obsolete; replaced in civil practice by a motion or answer. 2. Criminal. A defendant’s formal response to a criminal charge.",
    "PLEA BARGAIN": "A negotiated plea between the prosecuting entity and the defendant but subject to the court’s approval. Compare, OPEN PLEA.",
    "PLEAD": "To answer or respond to an indictment; to answer an allegation.",
    "PLEADINGS": "The process by which parties to an action alternately present written statements of their contentions of the case.",
    "PLEA IN ABATEMENT": "In civil practice, the same as a motion. See, MOTION.",
    "PLURIES SUMMONS": "A third summons issued when the original and alias summonses have been ineffective.",
    "POLLING THE JURY": "Procedure in which each juror is asked by the court if the verdict rendered is that juror’s verdict.",
    "POST-NUPTIAL": "Latin: “After marriage.”",
    "POST-RELEASE SUPERVISION": "A conditional suspension of a prison sentence as set forth in Miss. Code Ann. Section 47-7-34.",
    "POWER OF ATTORNEY": "A document empowering another person to act as one’s legal representative or attorney.",
    "PRECEDENT": "An appellate decision that carries authoritative weight in deciding later cases involving similar legal issues.",
    "PREJUDICIAL ERROR": "An error which warrants the appellate court to reverse the judgment of a lower court; reversible error.",
    "PRELIMINARY HEARING": "A hearing conducted pursuant to Rule 6 of the Mississippi Rules of Criminal Procedure for determining whether there is probable cause to believe that a felony has been committed and probable cause to believe that the defendant committed it. A defendant who has been indicted by a grand jury is not entitled to a preliminary hearing.",
    "PRENUPITIAL AGREEMENT": "A premarital contract that operates in the event of divorce or death. Such are enforceable in Mississippi provided there is fairness in execution and full disclosure.",
    "PREPONDERANCE OF THE EVIDENCE": "Evidence which is of greater weight than that offered against it; more probable than not.",
    "PRESENTENCE REPORT": "A written report submitted to the court prior to sentencing that consists of an offender's criminal, educational, and social history. It also contains other pertinent information, such as victim impact statements.",
    "PRESENTMENT": "An instruction presented by a grand jury for an indictment to be drawn.",
    "PRESIDING JUDGE": "The judge who directs, controls or regulates the proceedings in a court.",
    "PRESUMPTION OF LAW": "A presumption that the law expressly directs to be made from particular facts in the absence of contrary evidence.",
    "PRETERMITTED HEIR": "A child born after the making and publishing of a will but who is still entitled to a share of the testator’s estate.",
    "PRETRIAL CONFERENCE": "A conference held by the judge and attorneys prior to trial for the purpose of considering various ways to expedite or resolve the case.",
    "PRINCIPAL": "Civil. One who authorizes another to act as an agent.",
    "PROBABLE CAUSE": "Standard for issuing an arrest warrant or search warrant upon reasonably trustworthy information regarding criminal activities or contraband.",  
    "PROBATE": "The act or process of proving the validity of a will and disposing of the estate.",
    #"PROBATION": "A period of time whereby a defendant is not incarcerated but must abide by certain terms and conditions imposed by the court.",
    "PROCEEDING": "The form and manner of conducting judicial business.",
    "PROCESS": "Formal procedures a court uses to acquire or exercise jurisdiction over persons or property, e.g., a summons or subpoena.",
    "PROCESS SERVER": "One employed to deliver a summons, subpoena or other document.",
    "PROOF OF SERVICE": "Evidence that process has been returned.",
    "PROPERTY": "Something, such as land or an item, which one has the right to own, possess, and use.",
    "PRO SE": "Latin: “For himself.” Self-representation; representing oneself without the assistance of an attorney.",
    "PRO TANTO": "Latin: “For so much.”",
    "PROTECTIVE ORDER": "1. Domestic relations. An order issued by the chancery, circuit, or county court to bring about the cessation of abuse of the petitioner, any minor children, or any person alleged to be incompetent. 2. Discovery. An order issued by the court to protect a party or witness from discovery abuses.",
    "PRO TEMPORE (PRO TEM.)": "Latin: “For the time being.”",
    "PROXIMATE CAUSE": "A necessary element in proving negligence that is comprised of two distinct concepts: “cause in fact” and “foreseeability.”",
    "PUBLIC OFFICIAL": "One who is elected or appointed to any office or position where the salary or fee of such office or position is paid by the State or any political subdivision.",
    "PUNITIVE DAMAGES": "Damages awarded to punish the wrongdoer.",
    "QUAERE": "Latin: “A query.” Questions; doubt.",
    "QUALIFIED IMMUNITY": "Exemption from civil liability for public officials acting within the course and scope of their employment.",
    "QUANTUM": "Quantity; amount.",
    "QUASH": "To annul or make void.",
    "QUID PRO QUO": "Latin: “One thing for another.” A fair exchange.",
    "QUIET AND CONFIRM TITLE": "Decree validating the title to real property.",
    "QUITCLAIM DEED": "A deed that conveys, without warranty, whatever title, interest, or claim the grantor may have in the described real property.",
    "SPECIAL WARRANTY DEED": "A deed that provides limited warranties regarding the grantor's title to the property.",
    #"QUO WARRANTO": "Latin: “By what authority.” A statutory mechanism for trying, among other things, a person’s right to political office.",
    "REAL PROPERTY, REALTY": "Land and generally anything affixed to land or erected upon it.",
    "REASONABLE SUSPICION": "A particularized and objective basis for suspecting criminal activity sufficient to justify an investigatory stop.",
    "RECEIVER": "One appointed by the court to take and manage the property or money which is the subject matter of litigation.",
    "RECESS": "A short break ordered by the court during the course of the trial.",
    "RECKLESS DISREGARD": "When one knows that a risk of emotional distress probably would result from one’s conduct, and then disregards that risk and the harm that may occur as a result.",
    "RECKLESS DRIVING": "Driving a vehicle in such a manner as to indicate either a wilful or a wanton disregard for the safety of persons or property.",
    "RECORD": "1. The act of filing a written instrument. 2. A complete transcript of all trial proceedings, along with any pleadings and exhibits. Compare, ABSTRACT OF RECORD.",
    "RECOUPMENT": "See, COUNTERCLAIM.",
    "REDACT": "To remove text from a document, such as personal information; to edit.",
    "REDEMPTION": "The buying back or repurchasing of something, e.g., the redemption of property by paying off the mortgage.",
    "REDIRECT EXAMINATION": "Questioning of one’s own witness at trial after the opposing side has finished its cross-examination.",
    "REFORMATION OF INSTRUMENTS": "The correction or modification of written documents to make them conform to the original intent of the parties.",
    "RELEASE": "1. The discharge of a particular right or claim. 2. Procedures under Rule 8 of the Mississippi Rules of Criminal Procedure governing the release of a defendant from custody pending trial.",
    "REMAINDER": "A future interest in a life estate or estate for years.",
    "REMAND": "To send back for further action consistent with the accompanied directives or instructions, e.g., a remand for new trial.",
    "REMITTITUR": "Latin: “It is sent back.” A court order reducing the amount of damages awarded by the jury.",
    "REPLEVIN": "An action brought to recover possession of goods unlawfully taken.",
    "REPORTS": "Published judicial cases arranged according to jurisdiction, court, period of time subject matter or case significance.",
    "RES": "Latin: “A thing.” The thing over which a court exercises in rem jurisdiction.",
    "RESCIND": "To abrogate, annul, or cancel, especially as to contracts.",
    "RESIDENCE": "The place where one presently lives.",
    "DOMICILE": "The place where one has their permanent home or principal establishment.",
    "RES IPSA LOQUITUR": "Latin: “The thing speaks for itself.”",
   # "RES JUDICATA": "Latin: “A thing adjudicated.” A doctrine that precludes parties from relitigating the same controversy.",
    "RESPONDEAT SUPERIOR": "Latin: “Let the master answer.” A doctrine holding employers liable for the negligence of its employees.",
    #"RESPONDENT": "The party against whom a petition is filed.",
    "RETAINER FEE": "An advanced payment to an attorney for legal representation.",
    "RETURN": "Documentation delivered to the court showing execution of process.",
    "REVIEW": "To carefully consider a legal or factual issue.",
    "RULE AGAINST PERPETUITIES": "A common law rule that invalidates interests in real estate that vest too remotely in time.",
    "RULE OF SEQUESTRATION": " (THE RULE) The practice of excluding witnesses from the courtroom prior to the time for them to testify.",
    "SANCTION": "A judicial disciplinary action.",
    "SCIENTER": "Latin: “Knowingly.”",
    "SCIRE FACIAS": "Latin: “Cause it to be known.” A writ requiring the surety to show cause why a judgment nisi should not be made final.",
    "SEARCH WARRANT": "An order issued by a judge upon probable cause directing an officer to search a specified place for a specified thing.",
    "SELF DEFENSE": "Justifiably protecting oneself or others against an assault.",
    "SENTENCE": "Punishment imposed by the court upon a criminal defendant who has been convicted.",
    "SEPARATE MAINTENANCE": "A decree granting an allowance for the support of the spouse and any children during a period of separation.",
    "SEQUESTRATION": "1. The isolation of the jury or witnesses during a trial. 2. Authorized seizure of property pertinent to a lawsuit to prevent its removal, concealment, or transfer.",
    "SERVICE OF PROCESS": "The delivery of a summons, subpoena, etc., by an authorized person; official notification of a legal action or proceeding.",
    "SET-OFF": "See, COUNTERCLAIM.",
    #"SETTLEMENT": "An agreement that resolves the claims and issues between the parties.",
    "SETTLEMENT, STRUCTURED": "An agreement where one agrees to pay sums of money to another over a specified period of time.",
    "SHOW CAUSE": "Procedure that affords a person the opportunity to give a satisfactory reason why the court should not make final a particular judgment, e.g. a show cause hearing on a judgment nisi.",
    "SHOWUP": "A police identification procedure in which the suspect is presented alone. Compare, LINEUP.",
    "SINE DIE": "Latin: “Without date.”",
    "SINE QUA NON": "Latin: “That without which the thing cannot be.” An indispensable thing or condition.",
    "SITUS": "Latin: “Place.”",
    "SLANDER": "A spoken defamatory statement. Compare, DEFAMATION.",
    "SOCIAL GUEST": "One who goes onto and remains on another’s property at the property owner’s invitation to enjoy hospitality or an event.",
    "SOCIAL MEDIA": "Formats for users to communicate electronically.",
    "SOFTWARE": "Computer programs such as operating systems and applications.",
    "SOLICITOR": "An English legal practitioner.",
    "SPECIAL COMMISSIONER": "A non-lawyer appointed by the court to conduct a judicially ordered sale or partition of real or personal property.",
    "SPECIAL MASTER": "A qualified person appointed, upon written consent of the parties or a showing that an exceptional condition requires it, to perform a specified act. ",
    "SPECIAL VENIRE": "The list of jurors summoned for a capital case.",
    "SPECIAL VERDICT": "A verdict requiring a special written finding upon each issue of fact.",
    "SPECIAL WARRANTY DEED": "A deed where grantor specially warrants to defend title only to those claims of grantor and those claiming through grantor. Compare, QUITCLAIM DEED.",
    #"SPECIFIC PERFORMANCE": "An equitable remedy for a breach of contract compelling the performance of the terms of the contract.",
    "STALKING": "Any person who willfully, maliciously and repeatedly follows or harasses another person, or who makes a credible threat, with the intent to place that person in reasonable fear of death or great bodily injury.",
    "STARE DECISIS": "Latin: “To stand by the thing decided.”",
    "STATUS OFFENSE": "Conduct subject to adjudication by the youth court that would not be a crime if committed by an adult.",
    "STATUS QUO": "Latin: “The situation that currently exists.”",
    #"STATUTE": "A law enacted by the legislature or Congress.",
    "STATUTE OF FRAUDS": "Statutory requirement that certain contracts be in writing and signed, e.g., the sale of lands.",
    "STATUTE OF LIMITATIONS": "The time period within which a lawsuit must be filed.",
    "STAY": "The halting of a judicial process by court order.",
    "STIPULATION": "An agreement between attorneys on opposite sides of a case allowing a certain fact to be established in evidence without the necessity of further proof.",
    "SUA SPONTE": "Latin: “Of one’s own will.” Without promptng or suggestion; voluntarily.",
    "SUB JUDICE": "Latin: “Under judicial consideration.”",
    "SUBLEASE": "A lease whereby the tenant rents an interest in the leasehold property to a third party; creates a legal relationship known assublessor and sublessee. Compare, LEASE.",
    "SUBPOENA": "Process requiring a witness to appear and give testimony at a deposition, hearing or trial.",
    "SUBPOENA DUCES TECUM": "Process requiring a witness to produce certain documents, records, or other tangible evidence at a deposition, hearing or trial.",
    "SUBROGATION": "One’s right to sue on the claim of another.",
    "SUBSCRIPTION": "Signature on a legal document.",
    "SUI GENERIS": "Latin: “Of its own kind.”",
    "SUI JURIS": "Latin: “Of one’s own right.” Full legal capacity.",
    "SUMMARY JUDGMENT": "A judgment made on the pleadings where there is no genuine issue of material fact requiring a trial and the prevailing party is entitled to a judgment as a matter of law.",
    "SUMMONS": "Legal notification of a lawsuit as set forth in M.R.C.P. 4 or as otherwise required by law."    ,
    "SUPERSEDEAS": "Latin: “You must desist.” A stay of legal proceedings pending an appeal, e.g., a supersedeas of a money judgment.",
    "SUPERVISED PROBATION": "A conditional suspension of a prison sentence as set forth in Miss. Code Ann. Section 47-7-33.",
    "SUPRA": "Latin: “Above.” Compare, INFRA.",
    "SURETY": "One who is liable for the debt of another in the event of default, e.g., a bail bondsman.",
    "SUSPENDED SENTENCE": "A prison sentence that a defendant does not have to serve upon successful completion of probation.",
    "TALESMAN": "A bystander summoned by the court for jury service.",
    "TAX TITLE": "Title to land purchased at a tax sale.",
    "TENANCY BY THE ENTIRETY": "Co-ownership of property by husband and wife with the right of survivorship. The deed or other instrument must clearly indicate the intent to create a tenancy by the entirety with the right of survivorship, and not as tenants in common.",
    "TENANCY IN COMMON": "Co-ownership of property by two or more persons without the right of survivorship. Compare, JOINT TENANCY.",
    "TENANT": "One who rents property from another.",
    "TENANT AT WILL": "One who rents property from another without a fixed term.",
    "TENDER": "A monetary offer, usually to settle a claim.",
    "TERMINATION OF PARENTAL RIGHTS": "Procedure, as set forth in the “Mississippi Termination of Parental Rights Law,” for the termination of all parental rights regarding a child.",
    "TERM OF COURT": "Time during which the court legally conducts business. Compare, VACATION.",
    "TESTAMENTARY": "Pertaining to a will.",
    "TESTATOR, TESTATRIX": "A person who has made a will; one who has died having left a will; may be a testator (male) or testatrix (female).",
    "TESTIMONIUM CLAUSE": "A part of a document, usually a deed, which gives the date on which the writing was executed and by whom.",
    #"TESTIMONY": "Spoken evidence given under oath or affirmation.",
    "TITLE": "The right to, or ownership in, real or personal property; the document which is evidence of this right.",
    "TORT": "Latin: “Twisted.” A negligent or intentional act that causes harm for which there is liability.",
    "TORTFEASOR": "One who commits a tort; a wrongdoer.",
    "TRADEMARK": "A word, phrase, symbol or design which identifies a product as belonging to its owner.",
    "TRANSCRIPT": "The official verbatim record of legal proceedings.",
    "TRANSFER OF VENUE": "The transfer of a case from one venue to the proper county of venue.",
    "TRAUMA": "Any injury to the body caused by external violence; a wound.",
    "TRESPASS": "An unlawful act against another’s property.",
    "TRESPASSER": "One who commits a trespass; one who goes onto and remains on another’s property without the property owner’s permission or consent.",
    "TRUE BILL": "A grand jury’s endorsement upon a bill of indictment when sufficient evidence is found to warrant a criminal charge. Compare, NO BILL.",
    "TRUSTEE": "One who holds property in trust for the benefit of another.",
    "TRUSTY STATUS": "A prisoner status, as defined by the classifications board of the Mississippi Department of Corrections, having certain benefits and privileges.",
    "TURNKEY": "A jailor, especially one in charge of the keys; an officer who is primarily assigned to the custody of prisoners.",
    "ULTRA VIRES": "Latin: “Beyond the power.” Transcending legal power or authority, especially if by an officer of a corporation.",
    "UNDUE INFLUENCE": "Exerting influence or control over another to the extent of destroying free agency or voluntary consent.",
    "UNLAWFUL ENTRY AND DETAINER": "A summary remedy to rightfully restore possession of real property.",
    "UNSUPERVISED PROBATION": "A conditional suspension of a prison sentence under the supervision of the judge.",
    "USURY": "A higher rate of interest charged on loans or accounts than allowed by law.",
    "VACATION": "The period between terms of court.",
    "VENDEE": "A purchaser.",
    "VENDOR": "A seller.",
    "VENIRE": "Technically, a writ summoning persons to court to serve as jurors; commonly used to refer to the entire group of jurors summoned.",
    "VENIRE, SPECIAL": "See, SPECIAL VENIRE.",
    "VENIREMEN": "Members of a panel of jurors.",
    "VENUE": "The particular geographical area, such as a county, in which a court with jurisdiction may hear and determine a case.",
    "VERDICT": "A formal decision or finding by a jury.",
    "VICTIM": "1. One who has been harmed by a wrongful act. 2. As set forth in Miss. Code Ann. Section 99-43-3, a person against whom the criminal offense has been committed, or if the person is deceased or incapacitated, the lawful representative.",
    "VOID": "Of no legally binding effect.",
    "VOIDABLE": "Capable of being declared void.",
    "VOIR DIRE": "French: “To speak the truth.” The preliminary examination by the court and attorneys as to the qualifications of jurors or witnesses.",
    "VULNERABLE PERSON": "One who is not able to lead a normal daily life or is not able to take care of oneself due to a mental, emotional, physical, or developmental state, or as a result of aging or brain damage.",
    "WAIVER": "Intentional relinquishment or abandonment of a known right.",
    "WILD ANIMAL": "An animal that is not customarily owned or used by people.",
    "WAIVER OF IMMUNITY": "1. Statutory provision that waives the immunity of the state and its political, subdivisions from certain tort claims.",
    "WAIVER OF PROCESS": "Procedure whereby a party to lawsuit waives service of process as allowed by law.",
    "WARRANTY DEED": "Conveyance of clear, good title to real property, which especially has the effect of embracing all of the five covenants known to common law, to wit: seizin, power to sell, freedom from encumbrance, quiet  enjoyment and warranty of title, including defending title against any claims.",
    "WILL": "A properly executed document that directs the distribution of real and personal property of an estate to the heirs.",
    "WILLFUL, WILLFULLY": "Intentionally doing, or failing to do, an act.",
    "WITNESS": "One who testifies under oath in a legal proceeding.",
    "WRIT OF ERROR CORAM NOBIS": "Latin: “The error before us.” A device allowing the court to correct its own judgment upon the discovery of substantial factual errors.",
    "YOUTH COURT": "Statutorily created court with exclusive original jurisdiction in all proceedings concerning: a delinquent child; an abused or neglected child; a child in need of supervision; and a dependent child.",
    "ZONING": "A municipal action which defines or  restricts the acceptable use of real property."

}

# --- Sidebar Glossary Panel for Legal Terms ---
with st.sidebar.expander("📚 Legal Glossary", expanded=True):
    st.markdown("## 📚 Common Legal Terms")
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
                    "ISSUE": "orange",
                    "NONE": "gray",
                    "PREAMBLE": "brown",
                    "PRE_NOT_RELIED": "pink",
                    "PRE_RELIED": "darkcyan",
                    "RATIO": "blueviolet",
                    "RLC": "magenta",
                    "RPC": "teal",
                    "STA": "brown"
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

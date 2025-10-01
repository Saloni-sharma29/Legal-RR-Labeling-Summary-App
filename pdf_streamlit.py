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

This file bundles the preprocessing, abbreviation search, model loading (cached),
labeling and summarization UI in Streamlit.
"""

from urllib import response
import streamlit as _st_mod

# Robust wrapper around the streamlit module that forwards attributes to the
# real module but provides safe fallbacks for missing caching APIs. This is
# necessary because some older Streamlit versions (e.g. 1.7) don't expose
# `cache_resource` or `cache_data`, and assigning attributes directly to the
# module can be unreliable across different import/runtime contexts.
class _StreamlitProxy:
    def __init__(self, mod):
        object.__setattr__(self, '_mod', mod)

    def __getattr__(self, name):
        mod = object.__getattribute__(self, '_mod')
        # If attribute exists on real module, return it
        if hasattr(mod, name):
            return getattr(mod, name)
        # Provide sensible fallbacks for newer APIs
        if name == 'cache_resource':
            if hasattr(mod, 'experimental_singleton'):
                return getattr(mod, 'experimental_singleton')
            if hasattr(mod, 'cache'):
                return getattr(mod, 'cache')
            # no-op decorator
            def _noop(*a, **k):
                def _d(f):
                    return f
                return _d
            return _noop
        if name == 'cache_data':
            if hasattr(mod, 'cache'):
                return getattr(mod, 'cache')
            def _noop(*a, **k):
                def _d(f):
                    return f
                return _d
            return _noop
        # Default: raise AttributeError like normal module access
        raise AttributeError(f"module 'streamlit' has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Forward assignments to the underlying module
        mod = object.__getattribute__(self, '_mod')
        setattr(mod, name, value)

# Expose the proxy as `st` for the rest of the file
st = _StreamlitProxy(_st_mod)
import PyPDF2
import re
import nltk
from nltk.tokenize import sent_tokenize
import requests
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline
import json

torch.classes.__path__ = [] # add this line to manually set it to empty.


# --- NLTK setup ---
# NLTK punkt downloader (kept separate to avoid side effects during import).
# If download fails (common on systems without proper SSL certs), fall back to
# a simple regex-based sentence splitter so the app still runs.
try:
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize
except Exception:
    # fallback simple sentence tokenizer (keeps common sentence-ending punctuation)
    import re as _re
    def sent_tokenize(text):
        # split on [.?!]\s+ but keep the punctuation
        pieces = _re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in pieces if p.strip()]


# --- Abbreviations dictionary (from user's original code) ---
abbreviations_dict = {
"(P) Ltd.": "private limited",
"§": "section",
"§§": "multiple sections",
"¶": "paragraph",
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
"b.s.p.": "bahujan samaj party",
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
"cr. pc": "criminal procedure code",
"cr.": "civil revision",
"cr.p.c.":"criminal procedure code",
"crc": "camera ready copy",
"crl. a.": "criminal appeal",
"crl.": "criminal",
"crl.o.p.": "criminal original petition",
"crl.r.c.": "crl. revision case",
"crm.": "client relationship management",
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
"jj": "judges",
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
"m.p.": "member of parliament",
"m.p.": "miscellaneous petition",
"m.p.c": "model penal code",
"m.p.l.a.d.s.": "member of parliament local area development scheme",
"m.p.p.": "manipur people's party",
"m.r.c.a.": "malaysian rubber board",
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
"mp": "member of parliament",
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
"o": "on behalf of",
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
"p.s.p.": "praja socialist party",
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
"r.s.": "rajya sabha",
"r.s.p.": "revolutionary socialist party",
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
"s": "section",
"s.ct.": "supreme court reports",
"s.": "section",
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
"s.p.": "samajwadi party",
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
"vide": "see",
"vip": "very important person",
"viz.": "videlicet",
"voc": "victim of crime",
"vol.": "volume",
"vols.": "volumes",
"vs": "versus",
"vs.": "verses",
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
"you've": "you have"
}

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

def preprocess_text(text: str) -> str:
    # A cleaned-up adaptation of the user's preprocessing
    if not text:
        return ""
    text = re.sub(r"\xa0", " ", text)
    lines = [re.sub(r'[^a-zA-Z0-9.,)\-(/?\t ]', '', l) for l in text.splitlines()]
    lines = [re.sub(r'(?<=[^0-9])/(?=[^0-9])', ' ', l) for l in lines]
    lines = [re.sub(r"\t+", " ", l) for l in lines]
    lines = [re.sub(r" +", " ", l) for l in lines]
    lines = [re.sub(r"\.{2,}", "", l) for l in lines]
    lines = [re.sub(r"^ ?", "", l) for l in lines]
    lines = [l for l in lines if (len(l) != 1 and not re.fullmatch(r"(\d|\d\d|\d\d\d)", l))]

    # Remove lines referencing Indian Kanoon (user-specific)
    lines = [l for l in lines if not re.search(r"Indian Kanoon", l, re.I)]

    text = "\n".join(lines)
    text = re.sub(r"[()\[\]\"]", " ", text)
    text = re.sub(r" no\.", " number", text)
    text = re.sub(r" nos\.", " numbers", text)
    text = re.sub(r" co\.", " company", text)
    text = re.sub(r" ltd\.", " limited", text)

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
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s if s.endswith('.') else s + '.' for s in sentences]
    return "\n".join(sentences)


#***********************

def find_abbreviations(text: str, abbrev_dict: dict):
    """
    Finds abbreviations present in the given text based on the provided abbreviation dictionary.

    Parameters:
        text (str): The input text to search for abbreviations.
        abbrev_dict (dict): A dictionary mapping abbreviations to their full forms.

    Returns:
        dict: A dictionary of matched abbreviations and their corresponding full forms found in the text.
    """
    t = text.lower()
    matched = {}
    for abbr, full in abbrev_dict.items():
        # Match abbreviation as a standalone token, allowing for non-word characters
        pattern = r'(?<!\w)' + re.escape(abbr.lower()) + r'(?!\w)'
        if re.search(pattern, t):
            matched[abbr] = full
    return matched

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
    # Use a pipeline wrapper for convenience
    summarizer = pipeline("summarization", model=model, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)
    return summarizer

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
    df['encoded'] = le.fit_transform(df['label'])
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

# --- Streamlit UI ---
st.set_page_config(page_title="Legal Rhetorical Role Labeling", layout='wide')
st.title("Legal Rhetorical Role Labeling & Summarization")

# Sidebar
st.sidebar.header("Settings")
json_url = st.sidebar.text_input("Label mapping JSON URL", value="https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/train.json")
label_repo = st.sidebar.text_input("Labeling model repo", value="engineersaloni159/LegalRo-BERt_for_rhetorical_role_labeling")
summarizer_repo = st.sidebar.text_input("Summarizer model repo", value="facebook/bart-large-cnn")

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

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Extracted text")
    st.text_area("Extracted", value=raw_text, height=300)

# Preprocess
preprocess_button = st.button("Preprocess text")
if preprocess_button:
    cleaned_text = preprocess_text(raw_text.lower())
    st.session_state['cleaned_text'] = cleaned_text
else:
    cleaned_text = st.session_state.get('cleaned_text', '')

with col2:
    st.subheader("Preprocessed text")
    st.text_area("Preprocessed", value=cleaned_text, height=300)

# Abbreviations
if st.button("Show abbreviations found"):
    if not cleaned_text:
        st.warning("Please preprocess the text first")
    else:
        matches = find_abbreviations(cleaned_text, abbreviations_dict)
        if matches:
            for k, v in matches.items():
                st.markdown(f"**{k}**: {v}")
        else:
            st.info("No abbreviations found")

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
    with st.spinner('Loading labeling model... this may take a while'):
        tokenizer_label, model_label = load_labeling_model(label_repo)
    with st.spinner('Loading summarizer...'):
        summarizer = load_summarizer_model(summarizer_repo)
    models_loaded = True
    st.sidebar.success('Models loaded')
else:
    # try load lazily if present in cache
    try:
        tokenizer_label, model_label = load_labeling_model(label_repo)
        summarizer = load_summarizer_model(summarizer_repo)
        models_loaded = True
    except Exception:
        models_loaded = False

# Run labeling & summarization
if st.button("Label & Summarize"):
    if not cleaned_text:
        st.warning("Preprocess text first")
    elif not models_loaded:
        st.warning("Load models from the sidebar first (button 'Load models')")
    else:
        sentences = [s for s in cleaned_text.splitlines() if s.strip()]
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
            st.subheader("Summaries by label")
            for lab, sents in grouped.items():
                long_text = ' '.join(sents)
                if len(long_text.split()) < 20:
                    st.info(f"Not enough content to summarize for {lab}")
                    continue
                # Summarize (beware of long inputs) Doing the Recursive summarization for the long inputs
                try:
                # Truncate input to model max (BART = 1024 tokens)
                    inputs = summarizer.tokenizer(
                        long_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024
                    )
                    input_ids = inputs["input_ids"].to(summarizer.model.device)
                    attention_mask = inputs["attention_mask"].to(summarizer.model.device)

                    summary_ids = summarizer.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=150,
                        min_length=30,
                        num_beams=4,
                        early_stopping=True
                        )

                    summary_text = summarizer.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    st.markdown(f"**Summary for {lab}**: {summary_text}")
                except Exception as e:
                    st.error(f"Summarization failed for {lab}: {e}")

            prog.progress(100)
            st.success('Done')

st.markdown("---")
#st.caption("Built from the user's Tkinter app — adapted for Streamlit. Models can be large; running locally with a GPU is recommended.")
st.caption("Built for the Supreme Court of India Judgements dataset. Models can be large; running locally with a GPU is recommended.")

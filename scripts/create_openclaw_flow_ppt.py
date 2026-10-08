from pathlib import Path
import zipfile
from xml.sax.saxutils import escape

OUT = Path('/root/JZY/agent-threat-mining/artifacts/openclaw_flow_ppt/openclaw_graph_grounded_training_flow.pptx')
PREVIEW = Path('/root/JZY/agent-threat-mining/artifacts/openclaw_flow_ppt/openclaw_graph_grounded_training_flow_preview.svg')
EMU=914400
C={'ink':'1F2937','muted':'64748B','blue':'2F6FB3','bluebg':'EEF6FF','blueline':'82AEDD','green':'4B8D43','greenbg':'F0F8ED','greenline':'9BC58F','purple':'7651A8','purplebg':'F7F1FD','purpleline':'B89DDD','red':'C94C48','orange':'D97706','yellow':'FFF7DB','white':'FFFFFF'}
def E(x): return int(round(x*EMU))
def solid(c): return '<a:solidFill><a:srgbClr val="%s"/></a:solidFill>'%c
def line(c='CBD5E1',w=9000,arrow=False,dash=None):
    z='<a:ln w="%d">%s'%(w,solid(c))
    if dash:z+='<a:prstDash val="%s"/>'%dash
    if arrow:z+='<a:tailEnd type="triangle" w="med" len="med"/>'
    return z+'</a:ln>'
def body(t,size=10,color='1F2937',bold=False,align='ctr'):
    p=[]
    for x in str(t).split('\n'):
        p.append('<a:p><a:pPr algn="%s"/><a:r><a:rPr lang="en-US" sz="%d" b="%d"><a:solidFill><a:srgbClr val="%s"/></a:solidFill><a:latin typeface="Aptos"/></a:rPr><a:t>%s</a:t></a:r><a:endParaRPr lang="en-US" sz="%d"/></a:p>'%(align,size*100,int(bold),color,escape(x),size*100))
    return '<p:txBody><a:bodyPr anchor="ctr" wrap="square" lIns="50000" rIns="50000" tIns="30000" bIns="30000"/><a:lstStyle/>%s</p:txBody>'%''.join(p)
class Slide:
    def __init__(self):self.a=[];self.i=1
    def nid(self):self.i+=1;return self.i
    def box(self,x,y,w,h,t='',fc='FFFFFF',lc='CBD5E1',size=10,bold=False,color='1F2937',geom='roundRect',lw=9000,alpha=None,align='ctr'):
        i=self.nid(); f=solid(fc) if alpha is None else '<a:solidFill><a:srgbClr val="%s"><a:alpha val="%d"/></a:srgbClr></a:solidFill>'%(fc,alpha)
        self.a.append('<p:sp><p:nvSpPr><p:cNvPr id="%d" name="Shape %d"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm><a:prstGeom prst="%s"><a:avLst/></a:prstGeom>%s%s</p:spPr>%s</p:sp>'%(i,i,E(x),E(y),E(w),E(h),geom,f,line(lc,lw),body(t,size,color,bold,align)))
    def txt(self,x,y,w,h,t,size=9,color='1F2937',bold=False,align='ctr'):self.box(x,y,w,h,t,'FFFFFF','FFFFFF',size,bold,color,'rect',0,100,align)
    def ln(self,x1,y1,x2,y2,c='64748B',w=8000,arrow=True,dash=None):
        i=self.nid();self.a.append('<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="%d" name="Connector %d"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr><p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom>%s</p:spPr></p:cxnSp>'%(i,i,E(min(x1,x2)),E(min(y1,y2)),E(abs(x2-x1) or .01),E(abs(y2-y1) or .01),line(c,w,arrow,dash)))
    def circ(self,x,y,d,t='',fc='FFFFFF',lc='CBD5E1',size=9,bold=True,color='1F2937'):return None
    def doc(self,x,y,s=1,c=None): return None
    def qwen(self,x,y,s=1,c=None): return None
    def graph(self,x,y,s=1,c=None): return None
    def target(self,x,y,s=1,c=None): return None
    def db(self,x,y,s=1,c=None): return None
    def nodes(self,x,y,s=1,c=None): return None
    def checkpoint(self,x,y,s=1,c=None): return None
    def chain(self,x,y,n,c,g=.30): return None

def slide_xml(s):return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>%s</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'%''.join(s.a)

def build():
    s = Slide()
    s.box(0, 0, 13.333, 7.5, '', 'FFFFFF', 'FFFFFF', 8, False, 'FFFFFF', 'rect', 0)
    s.txt(.25, .08, 12.8, .30, 'OpenClaw Graph-Grounded Threat Propagation Training', 20, C['ink'], True, 'left')
    s.txt(.25, .38, 12.8, .16, 'From graph-grounded base construction to failure-driven process optimization', 8, C['muted'], False, 'left')

    panels = [
        (.12, .70, 4.23, 5.52, C['bluebg'], C['blueline'], C['blue'], '1  |  Graph-Grounded State Modeling'),
        (4.55, .70, 4.23, 5.52, C['greenbg'], C['greenline'], C['green'], '2  |  Base Template + Parsing Strategy'),
        (8.98, .70, 4.23, 5.52, C['purplebg'], C['purpleline'], C['purple'], '3  |  Failure-Driven Process Optimization'),
    ]
    for x, y, w, h, bg, li, ac, title in panels:
        s.box(x, y, w, h, '', bg, li, 8)
        s.txt(x + .24, y + .18, w - .48, .28, title, 12, ac, True, 'left')

    s.ln(4.35, 3.15, 4.55, 3.15, C['blue'], 22000, True)
    s.ln(8.78, 3.15, 8.98, 3.15, C['green'], 22000, True)

    # Stage 1: graph_wo_check.md is converted into a compact state model.
    s.box(.32, 1.22, 1.25, .82, 'SKILL.md\nTask requirements', 'FFFFFF', C['blueline'], 8, True)
    s.box(.32, 2.30, 1.25, .82, 'graph_wo_check.md\nAuthoritative trajectory', 'FFFFFF', C['blueline'], 8, True)
    s.box(.32, 3.38, 1.25, .82, 'Agent + tool\ncontext', 'FFFFFF', C['blueline'], 8, True)
    s.box(1.80, 1.22, 2.25, .82, 'State trajectory\nextraction', 'FFFFFF', C['blueline'], 9, True)
    s.box(1.80, 2.30, 2.25, .82, 'Behavior, tool, and\ndecision-gate structure', 'FFFFFF', C['blueline'], 9, True)
    s.box(1.80, 3.38, 2.25, .82, 'Graph-grounded\nworkflow representation', 'FFFFFF', C['blueline'], 9, True)
    s.box(.32, 5.14, 3.73, .62, 'Output: reachable behavior nodes, decision gates, execution paths, and terminal states', 'FFFFFF', C['blueline'], 7, True, C['blue'])

    # Stage 2: construct the basic reusable framework from the base representation.
    x = 4.80
    s.box(x, 1.22, 3.77, .78, 'Base inputs', 'FFFFFF', C['greenline'], 10, True)
    s.txt(x + .30, 1.56, 3.18, .28, 'graph trajectory  |  chain schema  |  initial framework', 7, C['muted'], True)
    s.box(x, 2.20, 3.77, .88, 'Basic threat propagation template', 'FFFFFF', C['greenline'], 10, True)
    s.txt(x + .30, 2.56, 3.18, .28, 'entry  |  carrier  |  trust shift  |  effect  |  containment', 7, C['green'], True)
    s.box(x, 3.30, 3.77, .88, 'Injection threat parsing strategy', 'FFFFFF', C['greenline'], 10, True)
    s.txt(x + .30, 3.66, 3.18, .28, 'required fields  |  match signals  |  quality checks', 7, C['green'], True)
    s.box(x, 4.42, 3.77, .82, 'Initial output contract', 'FFFFFF', C['greenline'], 10, True)
    s.txt(x + .30, 4.76, 3.18, .27, 'graph-grounded targets, propagation edges, and terminal summaries', 7, C['muted'], True)
    s.box(x, 5.47, 3.77, .30, 'Base framework + injection parsing strategy', C['greenbg'], C['greenline'], 7, True, C['green'])

    # Stage 3: optimize the process using runtime failures and strict ASR.
    x = 9.23
    s.box(x, 1.22, 3.77, .82, 'Runtime feedback collection', 'FFFFFF', C['purpleline'], 10, True)
    s.txt(x + .30, 1.55, 3.15, .30, 'train/test failures  |  strict ASR  |  historical successes', 7, C['muted'], True)
    s.box(x, 2.24, 3.77, .88, 'Targeted + random process optimization', 'FFFFFF', C['purpleline'], 10, True)
    s.txt(x + .30, 2.60, 3.15, .28, '16 failure-retrieved samples + 16 random exploration', 7, C['purple'], True)
    s.box(x, 3.34, 3.77, .88, 'Multi-Qwen3 failure analysis', 'FFFFFF', C['purpleline'], 10, True)
    s.txt(x + .30, 3.70, 3.15, .28, 'failure attribution  |  graph trajectory  |  parser review', 7, C['muted'], True)
    s.box(x, 4.44, 3.77, .88, 'Candidate update + validation gate', 'FFFFFF', C['purpleline'], 10, True)
    s.txt(x + .30, 4.80, 3.15, .28, 'accept on strict-ASR improvement or non-regressing new success', 7, C['purple'], True)
    s.box(x, 5.55, 3.77, .30, 'Iterate, accumulate historical success, checkpoint, and resume', C['purplebg'], C['purpleline'], 7, True, C['purple'])

    s.box(.12, 6.52, 13.10, .54, 'Pipeline: graph base -> template + injection parsing strategy -> runtime feedback -> targeted optimization -> validated process update', 'F9FBFE', 'AFC5DE', 8, True, C['ink'])
    s.txt(.26, 7.25, 12.8, .13, 'Editable elements: rounded rectangles, text, and two main data-flow arrows only', 6, C['muted'], False, 'left')
    return s

def write_ppt(s):
    ct='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/><Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/><Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/><Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/></Types>'
    rr='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>'
    pr='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>'
    pres='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst><p:sldSz cx="12192000" cy="6858000" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>'
    master='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld name="Master"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld><p:clrMap accent1="2F6FB3" accent2="4B8D43" accent3="7651A8" accent4="C94C48" accent5="D97706" accent6="64748B" bg1="FFFFFF" bg2="FFFFFF" folHlink="954F72" hlink="0563C1" tx1="000000" tx2="666666"/><p:hf/><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
    layout='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
    th='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="OpenClaw"><a:themeElements><a:clrScheme name="Office"><a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1><a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="2F6FB3"/></a:accent1><a:accent2><a:srgbClr val="4B8D43"/></a:accent2><a:accent3><a:srgbClr val="7651A8"/></a:accent3><a:accent4><a:srgbClr val="C94C48"/></a:accent4><a:accent5><a:srgbClr val="D97706"/></a:accent5><a:accent6><a:srgbClr val="64748B"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme><a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme><a:fmtScheme name="Office"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme></a:themeElements></a:theme>'
    mr='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>'
    lr='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>'
    sr='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>'
    with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
        for n,d in {'[Content_Types].xml':ct,'_rels/.rels':rr,'ppt/presentation.xml':pres,'ppt/_rels/presentation.xml.rels':pr,'ppt/slideMasters/slideMaster1.xml':master,'ppt/slideMasters/_rels/slideMaster1.xml.rels':mr,'ppt/slideLayouts/slideLayout1.xml':layout,'ppt/slideLayouts/_rels/slideLayout1.xml.rels':lr,'ppt/slides/slide1.xml':slide_xml(s),'ppt/slides/_rels/slide1.xml.rels':sr,'ppt/theme/theme1.xml':th}.items():z.writestr(n,d)

def write_preview():
    svg = r"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1333 750"><defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#64748B"/></marker></defs><rect width="1333" height="750" fill="white"/><text x="25" y="30" font-family="Arial" font-size="22" font-weight="700" fill="#172033">OpenClaw Graph-Grounded Threat Propagation Training</text><text x="25" y="52" font-family="Arial" font-size="10" fill="#64748B">From graph-grounded base construction to failure-driven process optimization</text><rect x="12" y="70" width="423" height="552" rx="10" fill="#EEF6FF" stroke="#82AEDD"/><rect x="455" y="70" width="423" height="552" rx="10" fill="#F0F8ED" stroke="#9BC58F"/><rect x="898" y="70" width="423" height="552" rx="10" fill="#F7F1FD" stroke="#B89DDD"/><text x="36" y="105" font-family="Arial" font-size="15" font-weight="700" fill="#2F6FB3">1 | Graph-Grounded State Modeling</text><text x="479" y="105" font-family="Arial" font-size="15" font-weight="700" fill="#4B8D43">2 | Base Template + Parsing Strategy</text><text x="922" y="105" font-family="Arial" font-size="15" font-weight="700" fill="#7651A8">3 | Failure-Driven Process Optimization</text><line x1="435" y1="315" x2="455" y2="315" stroke="#2F6FB3" stroke-width="5" marker-end="url(#arrow)"/><line x1="878" y1="315" x2="898" y2="315" stroke="#4B8D43" stroke-width="5" marker-end="url(#arrow)"/><g font-family="Arial" font-size="12" font-weight="700" text-anchor="middle" fill="#1F2937"><rect x="32" y="122" width="125" height="82" rx="8" fill="white" stroke="#82AEDD"/><text x="94" y="157">SKILL.md</text><text x="94" y="176">Task requirements</text><rect x="32" y="230" width="125" height="82" rx="8" fill="white" stroke="#82AEDD"/><text x="94" y="265">graph_wo_check.md</text><text x="94" y="284">Authoritative trajectory</text><rect x="32" y="338" width="125" height="82" rx="8" fill="white" stroke="#82AEDD"/><text x="94" y="373">Agent + tool</text><text x="94" y="392">context</text><rect x="180" y="122" width="225" height="82" rx="8" fill="white" stroke="#82AEDD"/><text x="292" y="157">State trajectory</text><text x="292" y="176">extraction</text><rect x="180" y="230" width="225" height="82" rx="8" fill="white" stroke="#82AEDD"/><text x="292" y="265">Behavior, tool, and</text><text x="292" y="284">decision-gate structure</text><rect x="180" y="338" width="225" height="82" rx="8" fill="white" stroke="#82AEDD"/><text x="292" y="373">Graph-grounded</text><text x="292" y="392">workflow representation</text><rect x="32" y="514" width="373" height="62" rx="8" fill="white" stroke="#82AEDD"/><text x="218" y="545">Output: reachable behavior nodes, decision gates,</text><text x="218" y="564">execution paths, and terminal states</text><rect x="480" y="122" width="377" height="78" rx="8" fill="white" stroke="#9BC58F"/><text x="668" y="157">Base inputs</text><text x="668" y="177">graph trajectory | chain schema | initial framework</text><rect x="480" y="220" width="377" height="88" rx="8" fill="white" stroke="#9BC58F"/><text x="668" y="256">Basic threat propagation template</text><text x="668" y="277">entry | carrier | trust shift | effect | containment</text><rect x="480" y="330" width="377" height="88" rx="8" fill="white" stroke="#9BC58F"/><text x="668" y="366">Injection threat parsing strategy</text><text x="668" y="387">required fields | match signals | quality checks</text><rect x="480" y="440" width="377" height="82" rx="8" fill="white" stroke="#9BC58F"/><text x="668" y="476">Initial output contract</text><text x="668" y="497">targets | propagation edges | terminal summaries</text><rect x="480" y="547" width="377" height="30" rx="8" fill="#F0F8ED" stroke="#9BC58F"/><text x="668" y="567">Base framework + injection parsing strategy</text><rect x="923" y="122" width="377" height="82" rx="8" fill="white" stroke="#B89DDD"/><text x="1111" y="157">Runtime feedback collection</text><text x="1111" y="178">train/test failures | strict ASR | historical successes</text><rect x="923" y="224" width="377" height="88" rx="8" fill="white" stroke="#B89DDD"/><text x="1111" y="260">Targeted + random process optimization</text><text x="1111" y="281">16 failure-retrieved + 16 random exploration</text><rect x="923" y="334" width="377" height="88" rx="8" fill="white" stroke="#B89DDD"/><text x="1111" y="370">Multi-Qwen3 failure analysis</text><text x="1111" y="391">failure attribution | graph trajectory | parser review</text><rect x="923" y="444" width="377" height="88" rx="8" fill="white" stroke="#B89DDD"/><text x="1111" y="480">Candidate update + validation gate</text><text x="1111" y="501">strict-ASR improvement or new success without regression</text><rect x="923" y="554" width="377" height="30" rx="8" fill="#F7F1FD" stroke="#B89DDD"/><text x="1111" y="574">Iterate | accumulate | checkpoint | resume</text><rect x="12" y="652" width="1310" height="54" rx="8" fill="#F9FBFE" stroke="#AFC5DE"/><text x="666" y="683">Pipeline: graph base -> template + injection parsing strategy -> runtime feedback -> targeted optimization -> validated process update</text></g></svg>"""
    PREVIEW.write_text(svg)


if __name__=='__main__':
    OUT.parent.mkdir(parents=True,exist_ok=True);s=build();write_ppt(s);write_preview();print(OUT);print(PREVIEW)

# ============================================================
# widgets/mound_report.py — Mound survey AI report widget
# ============================================================
from config.settings import GROQ_API_KEY


def build_mound_report_widget(location_name: str, lat_val: float, lon_val: float,
                               savings: dict, results: list) -> str:
    loc = (location_name or "Unknown Site").replace('"', '\\"').replace("'", "\\'")
    mm  = savings["manmade"]
    nat = savings["natural"]
    unc = savings["uncertain"]
    tot = savings["total"]
    pct = savings["pct_filtered"]
    ds  = savings["days_saved"]
    cs  = savings["cost_saved"]
    ap  = savings["area_priority"]

    return f"""<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500&family=JetBrains+Mono:wght@300;400&family=Archivo+Narrow:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:'Archivo Narrow',sans-serif;color:#d4cfc8}}
.panel{{background:linear-gradient(160deg,#0e1118 0%,#111520 100%);border:1px solid #1e2535;border-top:2px solid #7a5f42;border-radius:4px;padding:16px 18px 14px;}}
.panel-label{{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.22em;text-transform:uppercase;color:#4a5268;margin-bottom:10px;display:flex;align-items:center;gap:10px;}}
.panel-label::after{{content:'';flex:1;height:1px;background:linear-gradient(90deg,#1e2535,transparent);}}
.data-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:10px;}}
.dc{{background:rgba(255,255,255,0.03);border:1px solid #1e2535;border-radius:3px;padding:7px 9px;}}
.dcl{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:0.13em;text-transform:uppercase;color:#4a5268;margin-bottom:3px;}}
.dcv{{font-family:'Cormorant Garamond',serif;font-size:15px;color:#d4cfc8;}}
.dcv.red{{color:#d46b6b;}} .dcv.green{{color:#7ec899;}} .dcv.gold{{color:#d4a84b;}}
.trigger{{display:flex;align-items:center;justify-content:space-between;cursor:pointer;padding:9px 11px;background:rgba(255,255,255,0.02);border:1px solid #1e2535;border-radius:3px;transition:all 0.2s;user-select:none;}}
.trigger:hover{{border-color:#7a5f42;background:rgba(184,150,110,0.04);}}
.trigger-text{{font-family:'Archivo Narrow',sans-serif;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#7a8399;}}
.trigger-arrow{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#7a5f42;transition:transform 0.2s;}}
#output{{margin-top:10px;display:none}} #output.show{{display:block}}
.loading-row{{display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#4a5268;letter-spacing:0.08em;padding:6px 0;}}
.bar-loader{{flex:1;height:2px;background:#1e2535;border-radius:1px;overflow:hidden;position:relative;}}
.bar-loader::after{{content:'';position:absolute;top:0;left:-40%;width:40%;height:100%;background:linear-gradient(90deg,transparent,#b8966e,transparent);animation:scan 1.4s linear infinite;}}
@keyframes scan{{to{{left:140%}}}}
#report{{background:rgba(255,255,255,0.025);border:1px solid #1e2535;border-left:3px solid #7a5f42;border-radius:0 3px 3px 0;padding:12px 14px;font-family:'JetBrains Mono',monospace;font-size:10.5px;line-height:1.95;color:#a0b0a0;white-space:pre-wrap;display:none;}}
.cursor{{display:inline-block;width:6px;height:11px;background:#b8966e;margin-left:2px;vertical-align:middle;border-radius:1px;animation:blink .5s infinite;}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0}}}}
#errmsg{{font-family:'JetBrains Mono',monospace;font-size:10px;color:#d46b6b;padding:7px 10px;background:rgba(124,58,58,0.08);border:1px solid rgba(124,58,58,0.2);border-radius:3px;display:none;}}
</style>
</head>
<body>
<div class="panel">
  <div class="panel-label">AI Survey Report — Mound Analysis</div>
  <div class="data-grid">
    <div class="dc"><div class="dcl">Total Detected</div><div class="dcv">{tot}</div></div>
    <div class="dc"><div class="dcl">Man-made</div><div class="dcv red">{mm}</div></div>
    <div class="dc"><div class="dcl">Natural</div><div class="dcv green">{nat}</div></div>
    <div class="dc"><div class="dcl">Uncertain</div><div class="dcv gold">{unc}</div></div>
    <div class="dc"><div class="dcl">Area Filtered</div><div class="dcv">{pct:.0f}%</div></div>
    <div class="dc"><div class="dcl">Days Saved</div><div class="dcv green">{ds}</div></div>
    <div class="dc" style="grid-column:span 3;background:rgba(122,95,66,0.08);border-color:rgba(122,95,66,0.3);">
      <div class="dcl">Priority Survey Area</div>
      <div class="dcv" style="font-size:13px;">{ap} sq.km — focusing on {mm} man-made candidates</div>
    </div>
  </div>
  <div class="trigger" id="triggerBtn" onclick="runReport()">
    <span class="trigger-text" id="hint">Generate AI mound survey report</span>
    <span class="trigger-arrow" id="arr">&#x25BA;</span>
  </div>
  <div id="output">
    <div class="loading-row" id="loading"><div class="bar-loader"></div><span>Generating field survey report…</span></div>
    <div id="report"></div>
    <div id="errmsg"></div>
  </div>
</div>
<script>
var busy=false;
var GROQ_KEY="{GROQ_API_KEY}";
var LOC="{loc}",LAT={lat_val:.4f},LON={lon_val:.4f};
var TOT={tot},MM={mm},NAT={nat},UNC={unc},PCT={pct:.1f},DS={ds},CS={cs},AP={ap:.1f};
function highlight(txt){{
  return txt.replace(/\\n/g,'<br>')
    .replace(/(SUMMARY|KEY FINDINGS|SITE POTENTIAL|RECOMMENDATION):/g,
      '<span style="font-size:8.5px;letter-spacing:0.2em;color:#b8966e;font-weight:600;">$1:</span>');
}}
function typewriter(el,text,speed,done){{
  el.innerHTML='<span class="cursor"></span>';
  var i=0,buf='';
  (function tick(){{
    if(i<text.length){{buf+=text[i++];el.innerHTML=highlight(buf)+'<span class="cursor"></span>';
    setTimeout(tick,(buf[buf.length-1]==='.'||buf[buf.length-1]==='?')?speed*5:speed);}}
    else{{el.innerHTML=highlight(buf);if(done)done();}}
  }})();
}}
async function runReport(){{
  if(busy)return;busy=true;
  var out=document.getElementById('output'),loading=document.getElementById('loading'),
      report=document.getElementById('report'),errmsg=document.getElementById('errmsg'),
      hint=document.getElementById('hint'),arr=document.getElementById('arr');
  report.style.display='none';errmsg.style.display='none';
  loading.style.display='flex';out.classList.add('show');
  hint.textContent='Generating report…';arr.style.transform='rotate(90deg)';
  var prompt=
"You are a senior field archaeologist writing a structured mound survey assessment.\\n\\n"+
"SURVEY DATA:\\n"+
"  Site: "+LOC+" ("+LAT.toFixed(4)+"N, "+LON.toFixed(4)+"E)\\n"+
"  Total objects detected: "+TOT+"\\n"+
"  Man-made (potential archaeological): "+MM+"\\n"+
"  Natural: "+NAT+"\\n"+
"  Uncertain: "+UNC+"\\n"+
"  Area filtered out (natural eliminated): "+PCT+"%\\n"+
"  Priority survey area: "+AP+" sq.km\\n"+
"  Estimated field days saved: "+DS+"\\n"+
"  Estimated cost saved (USD): $"+CS+"\\n\\n"+
"Write EXACTLY 4 labelled lines. No preamble.\\n\\n"+
"SUMMARY: Overall picture — "+TOT+" objects detected, "+MM+" man-made candidates, "+PCT+"% filtered.\\n"+
"KEY FINDINGS: Ratio of man-made to natural ("+MM+":"+NAT+"), filtering efficiency and implications.\\n"+
"SITE POTENTIAL: Archaeological significance of "+MM+" man-made candidates at "+LOC+".\\n"+
"RECOMMENDATION: One specific actionable next step for the "+MM+" high-priority zones.\\n\\n"+
"Rules: Each line MUST start with its label in uppercase + colon. Plain English. No markdown. Max 45 words per line.";
  try{{
    var resp=await fetch('https://api.groq.com/openai/v1/chat/completions',{{
      method:'POST',
      headers:{{'Content-Type':'application/json','Authorization':'Bearer '+GROQ_KEY}},
      body:JSON.stringify({{model:'llama-3.3-70b-versatile',messages:[{{role:'user',content:prompt}}],temperature:0.4,max_tokens:500,stream:false}})
    }});
    if(!resp.ok)throw new Error('API error '+resp.status);
    var data=await resp.json();
    var full=(data.choices&&data.choices[0]&&data.choices[0].message&&data.choices[0].message.content)||'';
    loading.style.display='none';report.style.display='block';
    typewriter(report,full.trim(),13,function(){{hint.textContent='Report complete — click to regenerate';arr.style.transform='rotate(0deg)';busy=false;}});
  }}catch(err){{
    loading.style.display='none';errmsg.style.display='block';
    errmsg.textContent='Error: '+err.message;hint.textContent='Click to retry';arr.style.transform='rotate(0deg)';busy=false;
  }}
}}
</script>
</body>
</html>"""
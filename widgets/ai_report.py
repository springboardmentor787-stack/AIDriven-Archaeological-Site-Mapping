# ============================================================
# widgets/ai_report.py — AI field report HTML widget builders
# ============================================================
from config.settings import GROQ_API_KEY


def build_ai_report_widget(location_name: str, lat_val: float, lon_val: float,
                            slope_val: float, elev_val: float, risk_score: float,
                            risk_level_str: str, vari_mean: float, coverage: dict,
                            detect_count: int, auto_conf: float) -> str:
    loc     = (location_name or "Unknown Site").replace('"', '\\"').replace("'", "\\'")
    cov_str = " | ".join(f"{l}: {p}%" for l, p in coverage.items()) if coverage else "No vegetation data"

    if risk_score < 0.33:
        rc, rb, rl = "#7ec899", "rgba(74,124,89,0.12)", "rgba(74,124,89,0.35)"
    elif risk_score < 0.66:
        rc, rb, rl = "#d4a84b", "rgba(138,110,47,0.12)", "rgba(138,110,47,0.35)"
    else:
        rc, rb, rl = "#d46b6b", "rgba(124,58,58,0.12)", "rgba(124,58,58,0.35)"

    risk_pct  = min(risk_score * 100, 100)
    why_label = "WHY SAFE" if risk_score < 0.33 else ("WHY MODERATE" if risk_score < 0.66 else "WHY HARMFUL")

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
.risk-cell{{background:{rb};border-color:{rl};grid-column:span 3;display:flex;align-items:center;justify-content:space-between;padding:9px 12px;}}
.risk-text{{font-family:'Cormorant Garamond',serif;font-size:18px;color:{rc};}}
.risk-bar-wrap{{flex:1;margin:0 12px;height:3px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;}}
.risk-bar-fill{{height:100%;width:{risk_pct:.1f}%;background:{rc};border-radius:2px;}}
.trigger{{display:flex;align-items:center;justify-content:space-between;cursor:pointer;padding:9px 11px;background:rgba(255,255,255,0.02);border:1px solid #1e2535;border-radius:3px;transition:all 0.2s;user-select:none;}}
.trigger:hover{{border-color:#7a5f42;background:rgba(184,150,110,0.04);}}
.trigger-text{{font-family:'Archivo Narrow',sans-serif;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#7a8399;}}
.trigger-arrow{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#7a5f42;transition:transform 0.2s;}}
.trigger:hover .trigger-arrow{{color:#b8966e;}}
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
  <div class="panel-label">AI Field Report</div>
  <div class="data-grid">
    <div class="dc"><div class="dcl">Slope</div><div class="dcv">{slope_val:.1f}&deg;</div></div>
    <div class="dc"><div class="dcl">Elevation</div><div class="dcv">{elev_val:.0f} m</div></div>
    <div class="dc"><div class="dcl">VARI Index</div><div class="dcv">{vari_mean:.3f}</div></div>
    <div class="dc"><div class="dcl">Artifacts</div><div class="dcv">{detect_count}</div></div>
    <div class="dc"><div class="dcl">Confidence</div><div class="dcv">{auto_conf:.0%}</div></div>
    <div class="dc"><div class="dcl">Site</div><div class="dcv" style="font-size:11px;font-family:'Archivo Narrow',sans-serif;">{loc[:16]}</div></div>
    <div class="dc risk-cell">
      <div><div class="dcl">Erosion Risk</div><div class="risk-text">{risk_level_str} &mdash; {risk_score:.1%}</div></div>
      <div class="risk-bar-wrap"><div class="risk-bar-fill"></div></div>
    </div>
  </div>
  <div class="trigger" id="triggerBtn" onclick="runReport()">
    <span class="trigger-text" id="hint">Generate AI field analysis</span>
    <span class="trigger-arrow" id="arr">&#x25BA;</span>
  </div>
  <div id="output">
    <div class="loading-row" id="loading"><div class="bar-loader"></div><span>Analysing site data…</span></div>
    <div id="report"></div>
    <div id="errmsg"></div>
  </div>
</div>
<script>
var busy=false;
var GROQ_KEY="{GROQ_API_KEY}";
var LOCATION="{loc}",LAT={lat_val:.4f},LON={lon_val:.4f};
var SLOPE={slope_val:.1f},ELEVATION={elev_val:.0f};
var RISK_SCORE={risk_score:.4f},RISK_LEVEL="{risk_level_str}";
var VARI_MEAN={vari_mean:.4f},COVERAGE="{cov_str}";
var ARTIFACTS={detect_count},DET_CONF={auto_conf:.2f};
var WHY_LABEL="{why_label}";
function highlight(txt){{
  return txt.replace(/\\n/g,'<br>')
    .replace(/(EROSION RISK|{why_label}|TERRAIN|VEGETATION|RECOMMENDATION):/g,
      '<span style="font-size:8.5px;letter-spacing:0.2em;color:#b8966e;font-weight:600;">$1:</span>');
}}
function typewriter(el,text,speed,done){{
  el.innerHTML='<span class="cursor"></span>';var i=0,buf='';
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
  hint.textContent='Generating analysis…';arr.style.transform='rotate(90deg)';
  var vaText=VARI_MEAN>=0.35?"moderate to dense":VARI_MEAN>=0.2?"sparse":"very sparse / bare soil";
  var riskPct=(RISK_SCORE*100).toFixed(1);
  var prompt=
"You are a senior field archaeologist writing a structured site assessment.\\n\\n"+
"SITE DATA:\\n  Location: "+LOCATION+" ("+LAT.toFixed(4)+"N, "+LON.toFixed(4)+"E)\\n"+
"  Slope: "+SLOPE+"deg | Elevation: "+ELEVATION+"m | Detection confidence: "+(DET_CONF*100).toFixed(0)+"%\\n"+
"  VARI: "+VARI_MEAN.toFixed(3)+" ("+vaText+")\\n  Vegetation: "+COVERAGE+"\\n"+
"  Erosion Risk: "+riskPct+"% — "+RISK_LEVEL+"\\n  Artifacts: "+ARTIFACTS+"\\n\\n"+
"Write EXACTLY 5 labelled lines. No preamble.\\n"+
"EROSION RISK: "+RISK_LEVEL+" — cite slope ("+SLOPE+"deg), elevation ("+ELEVATION+"m), score ("+riskPct+"%).\\n"+
WHY_LABEL+": Explain WHY slope+elevation+VARI drives this risk and its archaeological implications.\\n"+
"TERRAIN: Describe terrain character. What site types does this terrain preserve or destroy?\\n"+
"VEGETATION: Interpret VARI ("+vaText+") and breakdown. How does it affect preservation?\\n"+
"RECOMMENDATION: One specific actionable conservation or investigation step.\\n\\n"+
"Rules: label:colon required. Plain English. No markdown. Max 40 words per line.";
  try{{
    var resp=await fetch('https://api.groq.com/openai/v1/chat/completions',{{
      method:'POST',
      headers:{{'Content-Type':'application/json','Authorization':'Bearer '+GROQ_KEY}},
      body:JSON.stringify({{model:'llama-3.3-70b-versatile',messages:[{{role:'user',content:prompt}}],temperature:0.45,max_tokens:500,stream:false}})
    }});
    if(!resp.ok)throw new Error('API error '+resp.status);
    var data=await resp.json();
    var full=(data.choices&&data.choices[0]&&data.choices[0].message&&data.choices[0].message.content)||'';
    loading.style.display='none';report.style.display='block';
    typewriter(report,full.trim(),13,function(){{hint.textContent='Analysis complete — click to regenerate';arr.style.transform='rotate(0deg)';busy=false;}});
  }}catch(err){{
    loading.style.display='none';errmsg.style.display='block';
    errmsg.textContent='Error: '+err.message;hint.textContent='Click to retry';arr.style.transform='rotate(0deg)';busy=false;
  }}
}}
</script>
</body>
</html>"""
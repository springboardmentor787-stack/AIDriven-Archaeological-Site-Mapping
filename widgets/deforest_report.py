# ============================================================
# widgets/deforest_report.py — Deforestation AI report widget
# ============================================================
from config.settings import GROQ_API_KEY


def build_deforest_report_widget(location_name: str, lat_val: float, lon_val: float,
                                  stats: dict, vari_threshold: float,
                                  intensity: float) -> str:
    loc      = (location_name or "Unknown Site").replace('"', '\\"').replace("'", "\\'")
    hot_pct  = stats["hotspot_pct"]
    structs  = stats["struct_count"]
    veg_cov  = stats["veg_coverage"]
    mean_an  = stats["mean_anomaly"]
    peak_an  = stats["peak_anomaly"]
    ground   = round(100.0 - veg_cov, 1)

    potential = "HIGH" if hot_pct > 20 or structs > 6 else ("MODERATE" if hot_pct > 10 or structs > 3 else "LOW")
    pot_col   = "#d46b6b" if potential == "HIGH" else ("#d4a84b" if potential == "MODERATE" else "#7ec899")

    return f"""<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500&family=JetBrains+Mono:wght@300;400&family=Archivo+Narrow:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:'Archivo Narrow',sans-serif;color:#d4cfc8}}
.panel{{background:linear-gradient(160deg,#0e1118 0%,#111520 100%);border:1px solid #1e2535;border-top:2px solid #2a6e4a;border-radius:4px;padding:16px 18px 14px;}}
.panel-label{{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.22em;text-transform:uppercase;color:#4a5268;margin-bottom:10px;display:flex;align-items:center;gap:10px;}}
.panel-label::after{{content:'';flex:1;height:1px;background:linear-gradient(90deg,#1e2535,transparent);}}
.data-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:10px;}}
.dc{{background:rgba(255,255,255,0.03);border:1px solid #1e2535;border-radius:3px;padding:7px 9px;}}
.dcl{{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:0.13em;text-transform:uppercase;color:#4a5268;margin-bottom:3px;}}
.dcv{{font-family:'Cormorant Garamond',serif;font-size:15px;color:#d4cfc8;}}
.trigger{{display:flex;align-items:center;justify-content:space-between;cursor:pointer;padding:9px 11px;background:rgba(255,255,255,0.02);border:1px solid #1e2535;border-radius:3px;transition:all 0.2s;user-select:none;}}
.trigger:hover{{border-color:#2a6e4a;background:rgba(42,110,74,0.06);}}
.trigger-text{{font-family:'Archivo Narrow',sans-serif;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:#7a8399;}}
.trigger-arrow{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#2a6e4a;transition:transform 0.2s;}}
.trigger:hover .trigger-arrow{{color:#4ec98a;}}
#output{{margin-top:10px;display:none}} #output.show{{display:block}}
.loading-row{{display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#4a5268;letter-spacing:0.08em;padding:6px 0;}}
.bar-loader{{flex:1;height:2px;background:#1e2535;border-radius:1px;overflow:hidden;position:relative;}}
.bar-loader::after{{content:'';position:absolute;top:0;left:-40%;width:40%;height:100%;background:linear-gradient(90deg,transparent,#4ec98a,transparent);animation:scan 1.4s linear infinite;}}
@keyframes scan{{to{{left:140%}}}}
#report{{background:rgba(255,255,255,0.025);border:1px solid #1e2535;border-left:3px solid #2a6e4a;border-radius:0 3px 3px 0;padding:12px 14px;font-family:'JetBrains Mono',monospace;font-size:10.5px;line-height:1.95;color:#a0b8a8;white-space:pre-wrap;display:none;}}
.cursor{{display:inline-block;width:6px;height:11px;background:#4ec98a;margin-left:2px;vertical-align:middle;border-radius:1px;animation:blink .5s infinite;}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0}}}}
#errmsg{{font-family:'JetBrains Mono',monospace;font-size:10px;color:#d46b6b;padding:7px 10px;background:rgba(124,58,58,0.08);border:1px solid rgba(124,58,58,0.2);border-radius:3px;display:none;}}
</style>
</head>
<body>
<div class="panel">
  <div class="panel-label">AI Deforestation Analysis Report</div>
  <div class="data-grid">
    <div class="dc"><div class="dcl">Veg Coverage</div><div class="dcv">{veg_cov:.1f}%</div></div>
    <div class="dc"><div class="dcl">Ground Exposed</div><div class="dcv">{ground:.1f}%</div></div>
    <div class="dc"><div class="dcl">Hotspot Area</div><div class="dcv">{hot_pct:.1f}%</div></div>
    <div class="dc"><div class="dcl">Hidden Structures</div><div class="dcv" style="color:{pot_col};">{structs}</div></div>
    <div class="dc"><div class="dcl">Mean Anomaly</div><div class="dcv">{mean_an:.4f}</div></div>
    <div class="dc"><div class="dcl">Peak Anomaly</div><div class="dcv">{peak_an:.4f}</div></div>
    <div class="dc" style="grid-column:span 3;background:rgba(42,110,74,0.08);border-color:rgba(42,110,74,0.3);">
      <div class="dcl">Site Potential</div>
      <div class="dcv" style="font-size:13px;color:{pot_col};">{potential} — {structs} anomalous zones in {hot_pct:.1f}% of ground area</div>
    </div>
  </div>
  <div class="trigger" id="triggerBtn" onclick="runReport()">
    <span class="trigger-text" id="hint">Generate AI hidden-ruins analysis</span>
    <span class="trigger-arrow" id="arr">&#x25BA;</span>
  </div>
  <div id="output">
    <div class="loading-row" id="loading"><div class="bar-loader"></div><span>Analysing hidden patterns…</span></div>
    <div id="report"></div>
    <div id="errmsg"></div>
  </div>
</div>
<script>
var busy=false;
var GROQ_KEY="{GROQ_API_KEY}";
var LOC="{loc}",LAT={lat_val:.4f},LON={lon_val:.4f};
var VEG_COV={veg_cov:.1f},GROUND={ground:.1f},HOT_PCT={hot_pct:.1f};
var STRUCTS={structs},MEAN_AN={mean_an:.4f},PEAK_AN={peak_an:.4f};
var POTENTIAL="{potential}",VARI_T={vari_threshold:.2f},INTENSITY={intensity:.2f};
function highlight(txt){{
  return txt.replace(/\\n/g,'<br>')
    .replace(/(SUMMARY|HIDDEN FEATURES|VEGETATION IMPACT|RECOMMENDATION):/g,
      '<span style="font-size:8.5px;letter-spacing:0.2em;color:#4ec98a;font-weight:600;">$1:</span>');
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
  var prompt=
"You are a senior field archaeologist interpreting AI-powered digital deforestation analysis.\\n\\n"+
"DATA:\\n  Site: "+LOC+" ("+LAT.toFixed(4)+"N, "+LON.toFixed(4)+"E)\\n"+
"  Vegetation coverage: "+VEG_COV+"% | Ground exposed: "+GROUND+"%\\n"+
"  Anomaly hotspot area: "+HOT_PCT+"% | Hidden structures: "+STRUCTS+"\\n"+
"  Mean anomaly: "+MEAN_AN+" | Peak: "+PEAK_AN+"\\n"+
"  Site potential: "+POTENTIAL+"\\n\\n"+
"Write EXACTLY 4 labelled lines. No preamble.\\n"+
"SUMMARY: Overall picture — "+VEG_COV+"% vegetation, "+STRUCTS+" anomaly zones, "+POTENTIAL+" potential.\\n"+
"HIDDEN FEATURES: What buried structure types are consistent with "+STRUCTS+" zones at "+HOT_PCT+"% coverage?\\n"+
"VEGETATION IMPACT: How has "+VEG_COV+"% canopy protected or damaged buried structures at "+LOC+"?\\n"+
"RECOMMENDATION: One specific next-step calibrated to "+POTENTIAL+" potential and "+STRUCTS+" anomaly zones.\\n\\n"+
"Rules: Labels uppercase + colon. Plain English. No markdown. Max 45 words per line.";
  try{{
    var resp=await fetch('https://api.groq.com/openai/v1/chat/completions',{{
      method:'POST',
      headers:{{'Content-Type':'application/json','Authorization':'Bearer '+GROQ_KEY}},
      body:JSON.stringify({{model:'llama-3.3-70b-versatile',messages:[{{role:'user',content:prompt}}],temperature:0.4,max_tokens:520,stream:false}})
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
"""Self-contained HTML/SVG report: no server, external JS, or plotting dependency."""
from __future__ import annotations

import html
import json
from pathlib import Path
import numpy as np
from .geometry import vertices


def polygon_rings(geometry):
    if geometry.geom_type == "Polygon":
        return [[list(x) for x in geometry.exterior.coords]] + [
            [list(x) for x in r.coords] for r in geometry.interiors]
    return [ring for p in geometry.geoms for ring in polygon_rings(p)]


def write_report(path: Path, token, candidates, gt, road, actors, ego_local, centerline, trace,
                 metadata, demo=False):
    # Display positions in a translated world frame, retaining metric scale.
    origin = gt.executed[0, :2]
    def xy(a):
        return (np.asarray(a)[:, :2] - origin).tolist()
    def trajectory(c):
        s = c.executed
        return {**c.record(), "executed": xy(s), "reference": xy(c.reference),
                "generated": None if c.generated is None else xy(c.generated),
                "footprints": [(vertices(row[:3], ego_local) - origin).tolist() for row in s],
                "speed": s[:, 3].tolist(), "acceleration": s[:, 5].tolist(),
                "jerk": (np.r_[0., np.diff(s[:, 5]) / .1]).tolist(),
                "steering": s[:, 7].tolist(),
                "controls": None if c.controls is None else c.controls.tolist()}
    unique = {c.id: c for c in [gt] + list(candidates)}
    data = {"token": str(token), "demo": demo, "metadata": metadata,
            "candidates": [trajectory(c) for c in unique.values()], "gt_id": gt.id,
            "road": [xy(r) for r in polygon_rings(road)], "centerline": xy(centerline),
            "actors": [{"id": a.token, "footprints": [
                (vertices(p, a.local) - origin).tolist() for p in a.poses]} for a in actors],
            "trace": trace}
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    path.write_text(TEMPLATE.replace("__PAYLOAD__", payload), encoding="utf-8")


TEMPLATE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>DrivoR offline trajectory inspection</title>
<style>
*{box-sizing:border-box}body{font:15px system-ui,sans-serif;background:#f5f7fa;color:#18283b;margin:0;padding:24px}
h1{font-size:24px;margin:0 0 8px}p{line-height:1.6}.note{background:#fff2ce;border-left:5px solid #cf8900;padding:12px}
.layout{display:grid;grid-template-columns:minmax(500px,1.5fr) minmax(350px,1fr);gap:18px;margin-top:18px}
.card{background:white;border:1px solid #d9e0e9;border-radius:10px;padding:16px;overflow:hidden}
svg{width:100%;display:block;background:#fbfcfe}#map{height:520px}label{margin-right:12px;display:inline-block}
select{max-width:100%;width:100%;padding:9px;margin:10px 0}input[type=range]{width:65%}button{padding:7px 14px}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:7px;text-align:left;border-bottom:1px solid #ddd}
.small{color:#516278;font-size:13px}.bad{color:#b42318}.good{color:#087f5b}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}
.legend span{margin-right:15px;white-space:nowrap}.charts svg{height:135px}details{margin-top:12px}
@media(max-width:950px){.layout{grid-template-columns:1fr}body{padding:12px}}
</style><h1>DrivoR · 离线轨迹搜索检查</h1><div id="identity" class="small"></div>
<p class="note" id="warning"></p>
<div class="layout"><div class="card">
<label>候选（含 GT、零残差 round-trip、通过项和失败项）</label><select id="candidate"></select>
<div class="legend"><span style="color:#2463b3">蓝：GT 执行</span><span style="color:#13946d">绿：候选执行</span>
<span style="color:#a162cf">紫虚线：候选输入参考</span><span style="color:#d97706">橙点线：控制搜索生成参考</span></div>
<p><label><input id="all" type="checkbox">叠加报告中的候选</label>
<label><input id="ghost" type="checkbox" checked>每秒车身轮廓</label></p>
<svg id="map" role="img" aria-label="GT、候选、道路与动态障碍物"></svg>
<p><button id="play">播放</button> <input id="time" type="range" min="0" max="40" value="0"><b id="clock"></b></p>
<p class="small">米制等比例俯视图；道路洞保留；黑灰色为当前时刻障碍物；深红色表示该区间存在检查失败。轨迹线以 rear axle 为参考，车身按实际尺寸绘制。</p>
</div><div class="card"><h3>正式分数与检查状态</h3><div id="status"></div><table id="metrics"></table>
<div id="events"></div><div class="charts"><svg id="speed"></svg><svg id="acceleration"></svg><svg id="jerk"></svg></div>
<details><summary>候选控制与检查明细</summary><pre id="detail"></pre></details></div></div>
<div class="card" style="margin-top:18px"><h3>搜索过程与运行口径</h3><div id="trace"></div>
<details><summary>配置、复现信息与待完成检查</summary><pre id="meta"></pre></details></div>
<script type="application/json" id="data">__PAYLOAD__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('data').textContent), $=x=>document.getElementById(x),
 esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const G=D.candidates.find(c=>c.id===D.gt_id);let C=D.candidates[1]||G, timer=null;
$('identity').textContent='Scene: '+D.token+' · 显示 '+D.candidates.length+' 条轨迹';
$('warning').textContent=D.demo?'合成 DEMO：用于检查搜索、几何验证与报告功能。分数为演示代理值，不是 NAVSIM PDMS，也不是实验收益。':'工程检查结果：评分来自 V1 正式入口。候选尚未完成全部严格验收，不可直接用作 teacher；当前初始 ego 与图像保持不变。';
D.candidates.forEach(c=>{const o=document.createElement('option');o.value=c.id;o.textContent=c.island+' / '+c.id.slice(0,8)+' / '+(100*c.metrics.score).toFixed(3)+' / '+(c.gate.checked_nominal_pass?'已实现的名义检查通过':'失败或不可核验');$('candidate').append(o)});
$('candidate').value=C.id;
function drawMap(){
 const t=+$('time').value, points=[...G.executed,...C.executed,...C.reference,...(C.generated||[])];
 if($('all').checked)D.candidates.forEach(c=>points.push(...c.executed));
 let xmin=Math.min(...points.map(p=>p[0]))-7,xmax=Math.max(...points.map(p=>p[0]))+7,
 ymin=Math.min(...points.map(p=>p[1]))-7,ymax=Math.max(...points.map(p=>p[1]))+7;
 const W=800,H=520,scale=Math.min(W/(xmax-xmin),H/(ymax-ymin)),cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;
 const pt=p=>[(p[0]-cx)*scale+W/2,H/2-(p[1]-cy)*scale], pointsStr=a=>a.map(p=>pt(p).join(',')).join(' ');
 const line=(a,color,width=2,dash='')=>'<polyline points="'+pointsStr(a)+'" fill="none" stroke="'+color+'" stroke-width="'+width+'" stroke-dasharray="'+dash+'"/>';
 const poly=(a,color,opacity=1)=>'<polygon points="'+pointsStr(a)+'" fill="'+color+'" fill-opacity="'+opacity+'" stroke="'+color+'" stroke-width="1.5"/>';
 let out='<path d="'+D.road.map(r=>'M'+r.map(p=>pt(p).join(',')).join('L')+'Z').join(' ')+'" fill="#e8edf1" fill-rule="evenodd" stroke="#a3afbd" stroke-width="1"/>';
 out+=line(D.centerline,'#99a5b2',1.5,'8 5');
 if($('all').checked)D.candidates.forEach(c=>out+=line(c.executed,c.gate.checked_nominal_pass?'#b7d7c8':'#e9baba',1));
 out+=line(G.reference,'#83abe0',1.5,'4 4')+line(G.executed,'#2463b3',3);
 if(C.generated)out+=line(C.generated,'#d97706',2,'2 4');
 out+=line(C.reference,'#a162cf',2,'6 4')+line(C.executed,'#13946d',3);
 if($('ghost').checked)for(let k=0;k<=40;k+=10){out+=poly(G.footprints[k],'#2463b3',.05)+poly(C.footprints[k],'#13946d',.05)}
 const events=C.gate.events||[],bad=events.some(e=>Math.abs(e.t-t*.1)<.051);
 D.actors.forEach(a=>{if(a.footprints[t])out+=poly(a.footprints[t],events.some(e=>e.actor===a.id&&Math.abs(e.t-t*.1)<.051)?'#b42318':'#536273',.55)});
 out+=poly(G.footprints[t],'#2463b3',.28)+poly(C.footprints[t],bad?'#b42318':'#13946d',.42);
 out+='<line x1="25" y1="490" x2="'+(25+5*scale)+'" y2="490" stroke="#18283b" stroke-width="3"/><text x="25" y="480" font-size="13">5 m</text>';
 $('map').setAttribute('viewBox','0 0 800 520');$('map').innerHTML=out;$('clock').textContent=(t*.1).toFixed(1)+' s';
 $('events').innerHTML='<p class="'+(bad?'bad':'small')+'">当前区间：'+esc(events.filter(e=>Math.abs(e.t-t*.1)<.051).map(e=>e.kind+(e.actor?' · '+e.actor:'')).join('; ')||'没有已记录的失败事件')+'</p>';
}
function chart(key,label,thresholds){
 const arrays=[G[key],C[key]], vals=arrays.flat().concat(thresholds),lo=Math.min(...vals)-.2,hi=Math.max(...vals)+.2,
 p=(v,i)=>[45+i/40*520,110-(v-lo)/(hi-lo)*80],ln=(a,color,dash='')=>'<polyline points="'+a.map((v,i)=>p(v,i).join(',')).join(' ')+'" fill="none" stroke="'+color+'" stroke-width="2" stroke-dasharray="'+dash+'"/>';
 let out='<text x="10" y="18" font-size="13">'+label+'</text><text x="4" y="40" font-size="11">'+hi.toFixed(1)+'</text><text x="4" y="113" font-size="11">'+lo.toFixed(1)+'</text>';
 thresholds.forEach(v=>out+=ln(Array(41).fill(v),'#c99696','4 3'));
 out+=ln(G[key],'#2463b3')+ln(C[key],'#13946d')+'<text x="45" y="130" font-size="11">0 s</text><text x="545" y="130" font-size="11">4 s</text>';
 $(key).setAttribute('viewBox','0 0 580 140');$(key).innerHTML=out;
}
function refresh(){
 $('status').innerHTML='<p class="'+(C.gate.checked_nominal_pass?'good':'bad')+'">'+(C.gate.checked_nominal_pass?'已实现的名义检查通过':'失败／不可核验')+'；完整验收：未完成；训练导出：禁止</p><p>'+esc(C.gate.reasons.join(', ')||'无已实现检查失败')+'</p>';
 $('metrics').innerHTML='<tr><th>指标（0–100）</th><th>GT</th><th>候选</th><th>差值</th></tr>'+Object.keys(C.metrics).map(k=>'<tr><td>'+esc(k)+'</td><td>'+(G.metrics[k]*100).toFixed(3)+'</td><td>'+(C.metrics[k]*100).toFixed(3)+'</td><td>'+((C.metrics[k]-G.metrics[k])*100).toFixed(3)+'</td></tr>').join('');
 $('detail').textContent=JSON.stringify({gate:C.gate,theta:C.theta,controls:C.controls},null,2);
 chart('speed','速度 m/s',[]);chart('acceleration','纵向加速度 m/s²',[-3.5,2]);chart('jerk','纵向 jerk m/s³',[-3.5,3.5]);drawMap();
}
$('candidate').onchange=()=>{C=D.candidates.find(c=>c.id===$('candidate').value);refresh()};
['time','all','ghost'].forEach(x=>$(x).oninput=drawMap);
$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放'}else{$('play').textContent='暂停';timer=setInterval(()=>{$('time').value=(+$('time').value+1)%41;drawMap()},100)}};
$('trace').innerHTML='<table><tr><th>初始化岛</th><th>代</th><th>通过 / 候选</th><th>最佳分数</th><th>累计评估</th></tr>'+D.trace.filter(r=>r.generation!==undefined).map(r=>'<tr><td>'+esc(r.island)+'</td><td>'+r.generation+'</td><td>'+r.feasible+' / '+r.population+'</td><td>'+(r.best_score===null?'—':(100*r.best_score).toFixed(3))+'</td><td>'+r.evaluations+'</td></tr>').join('')+'</table>';
$('meta').textContent=JSON.stringify(D.metadata,null,2);refresh();
</script></html>'''

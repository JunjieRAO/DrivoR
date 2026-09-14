// Pure JS unit check with a small document stub. Does not launch/control a browser.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const text = fs.readFileSync(process.argv[2], 'utf8');
const data = text.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
const program = text.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const elements = {};
const get = id => elements[id] ||= {value:'0', checked:false, innerHTML:'', textContent:'',
  append(){}, setAttribute(k,v){this[k]=v}};
get('data').textContent = data;
get('ghost').checked = true;
const ctx = vm.createContext({document:{getElementById:get, createElement:()=>({})},
  setInterval:()=>1, clearInterval(){}, console});
vm.runInContext(program, ctx);
const parsed=JSON.parse(data);
for(const c of parsed.candidates){
  get('candidate').value=c.id; get('candidate').onchange();
  for(const t of [0,10,20,30,40]){
    get('time').value=String(t);get('time').oninput();
    assert.ok(get('map').innerHTML.includes('<polygon'));
    assert.ok(!/NaN|undefined|Infinity/.test(get('map').innerHTML));
  }
}
get('all').checked=true;get('all').oninput();
for(const key of ['map','speed','acceleration','jerk'])assert.ok(!/NaN|undefined|Infinity/.test(get(key).innerHTML));
console.log(`Report JS passed: ${parsed.candidates.length} candidates, five time positions each; no browser rendering performed.`);

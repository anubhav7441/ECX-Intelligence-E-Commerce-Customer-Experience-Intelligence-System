const C={blue:'#2DD4BF',orange:'#FF9F43',green:'#34d399',grey:'#8a95b0',ink:'#A4AEC6',line:'#243049',pink:'#FFB020',sky:'#7CF0DD'};
Chart.defaults.font.family='"Manrope","Segoe UI",system-ui,sans-serif';
Chart.defaults.borderColor='#212955';
Chart.defaults.color=C.ink;
function mk(id,cfg){const el=document.getElementById(id);if(!el)return null;
  cfg.options=Object.assign({responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}}},cfg.options||{});
  return new Chart(el,cfg);}
function fmt(n,d=0){return n==null?'-':Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});}
function pct(n,d=1){return n==null?'-':(Number(n)*100).toFixed(d)+'%';}
function td(text,cls){const c=document.createElement('td');c.textContent=text;if(cls)c.className=cls;return c;}
async function postJSON(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  let j={};try{j=await r.json();}catch(e){}
  if(!r.ok)throw new Error(j.error||('Request failed ('+r.status+')'));return j;}
function toggleNav(){document.getElementById('sidebar').classList.toggle('open');}
function sortTable(tableId,col,num){const t=document.getElementById(tableId);const tb=t.tBodies[0];const rows=[...tb.rows];
  const dir=t.dataset.dir==='asc'?-1:1;t.dataset.dir=dir===1?'asc':'desc';
  rows.sort((a,b)=>{let x=a.cells[col].dataset.v??a.cells[col].textContent,y=b.cells[col].dataset.v??b.cells[col].textContent;
    if(num){x=parseFloat(x);y=parseFloat(y);return (isNaN(x)?-1e18:x)-(isNaN(y)?-1e18:y)>0?dir:-dir;}return x.localeCompare(y)*dir;});
  rows.forEach(r=>tb.appendChild(r));}

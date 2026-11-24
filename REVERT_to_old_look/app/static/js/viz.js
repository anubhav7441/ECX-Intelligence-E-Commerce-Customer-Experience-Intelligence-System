// Interactive chart tiles (Chart.js). Hover or slide over a line, bar or slice to see its exact value.
(function(){
  var INK='#e8eaf8',MUTE='#9aa0c4',GRID='rgba(154,160,196,.18)',PAL=['#8b5cf6','#3b82f6','#ef4444','#22c55e','#f97316','#6b7299'];
  var st=document.createElement('style');st.textContent='.vzg{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,440px),1fr));gap:16px}.vz{background:#141a31;border:1px solid #2a3157;border-radius:14px;padding:14px;display:flex;flex-direction:column}.vz.wide{grid-column:1/-1}.vzt{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:8px;color:#e8eaf8;font-size:1rem}.vzh{font-size:.72rem;color:#9aa0c4}.vzbox{position:relative;height:270px}.vz.wide .vzbox{height:300px}.vz img{width:100%;height:auto;border-radius:8px}.vzw{margin:10px 0 2px;font-size:.84rem;color:#9aa0c4}.vze{margin:0;font-size:.88rem;line-height:1.45;color:#dfe2f6}.heatbox{height:auto;overflow-x:auto}.vzheat{border-collapse:collapse;font-size:.75rem;color:#e8eaf8}.vzheat th{padding:4px 6px;color:#9aa0c4;font-weight:500;white-space:nowrap}.vzheat td{padding:6px 8px;text-align:center;border:2px solid #141a31}';document.head.appendChild(st);
  function fmt(v){return typeof v==='number'?v.toLocaleString(undefined,{maximumFractionDigits:2}):v}
  function tick(v){var a=Math.abs(v);return a>=1e6?(v/1e6).toFixed(1)+'M':a>=1e4?(v/1e3).toFixed(0)+'k':v.toLocaleString()}
  function heat(box,s){
    var n=s.labels.length,t=document.createElement('table');t.className='vzheat';
    var h='<tr><th></th>'+s.labels.map(function(l){return '<th>'+esc(l)+'</th>'}).join('')+'</tr>';
    s.matrix.forEach(function(row,i){h+='<tr><th>'+esc(s.labels[i])+'</th>'+row.map(function(v,j){var a=Math.min(1,Math.abs(v));var c=v>=0?'239,68,68':'59,130,246';
      return '<td title="'+esc(s.labels[i])+' vs '+esc(s.labels[j])+': '+v+'" style="background:rgba('+c+','+(0.12+a*0.75).toFixed(2)+')">'+v.toFixed(2)+'</td>'}).join('')+'</tr>'});
    t.innerHTML=h;box.appendChild(t);
  }
  function esc(x){var d=document.createElement('div');d.textContent=x==null?'':x;return d.innerHTML}
  function plot(canvas,s){
    var common={responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
      plugins:{legend:{display:false},tooltip:{backgroundColor:'#0b0f22',borderColor:'#3a4275',borderWidth:1,titleColor:INK,bodyColor:INK,padding:10,displayColors:true,
        callbacks:{label:function(c){var v=c.parsed.y!=null&&s.type!=='hbar'?c.parsed.y:(c.parsed.x!=null&&s.type==='hbar'?c.parsed.x:c.parsed);if(s.type==='donut'){var tot=s.values.reduce(function(a,b){return a+b},0);return ' '+c.label+': '+fmt(c.raw)+' ('+(c.raw/tot*100).toFixed(1)+'%)'}return ' '+s.label+': '+fmt(v)}}}}};
    var scales={x:{ticks:{color:MUTE,maxRotation:0,autoSkip:true,font:{size:11}},grid:{display:false}},y:{ticks:{color:MUTE,font:{size:11},callback:tick},grid:{color:GRID},beginAtZero:true}};
    if(s.type==='line')return new Chart(canvas,{type:'line',data:{labels:s.labels,datasets:[{data:s.values,borderColor:s.color,backgroundColor:s.color+'55',fill:true,tension:.3,pointRadius:3,pointHoverRadius:7,pointBackgroundColor:'#fff',borderWidth:2}]},options:Object.assign({},common,{scales:scales})});
    if(s.type==='donut')return new Chart(canvas,{type:'doughnut',data:{labels:s.labels,datasets:[{data:s.values,backgroundColor:PAL,borderColor:'#141a31',borderWidth:3,hoverOffset:8}]},options:Object.assign({},common,{cutout:'62%',interaction:{mode:'nearest',intersect:true},plugins:Object.assign({},common.plugins,{legend:{display:true,position:'right',labels:{color:INK,boxWidth:12,font:{size:12}}}})})});
    var hb=s.type==='hbar',sc=hb?{x:Object.assign({},scales.y,{grid:{color:GRID}}),y:{ticks:{color:MUTE,font:{size:11}},grid:{display:false}}}:scales;
    return new Chart(canvas,{type:'bar',data:{labels:s.labels,datasets:[{data:s.values,backgroundColor:s.colors||s.color,borderRadius:4,categoryPercentage:s.gap?1:.8,barPercentage:s.gap?.98:.7}]},options:Object.assign({},common,{indexAxis:hb?'y':'x',scales:sc,interaction:{mode:hb?'nearest':'index',intersect:false,axis:hb?'y':'x'}})});
  }
  window.ECXViz={draw:function(host,charts,imgBase){
    var grid=document.createElement('div');grid.className='vzg';host.appendChild(grid);
    charts.forEach(function(ch){
      var b=document.createElement('div');b.className='vz'+(ch.wide?' wide':'');
      var head=document.createElement('div');head.className='vzt';head.innerHTML='<b>'+esc(ch.title)+'</b><span class="vzh">Hover over the chart to see values</span>';b.appendChild(head);
      if(ch.spec&&ch.spec.type==='heat'){var hb=document.createElement('div');hb.className='vzbox heatbox';heat(hb,ch.spec);b.appendChild(hb)}
      else if(ch.spec&&window.Chart){var cb=document.createElement('div');cb.className='vzbox';var cv=document.createElement('canvas');cb.appendChild(cv);b.appendChild(cb);try{plot(cv,ch.spec)}catch(e){cb.remove()}}
      else{var im=document.createElement('img');im.src=imgBase+ch.file;im.alt=ch.title;im.loading='lazy';b.appendChild(im)}
      var w=document.createElement('p');w.className='vzw';w.innerHTML='<b>Why this chart:</b> '+esc(ch.why);b.appendChild(w);
      var e=document.createElement('p');e.className='vze';e.innerHTML='<b>What it shows:</b> '+esc(ch.explain);b.appendChild(e);grid.appendChild(b)});
  }};
})();

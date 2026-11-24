// Exit guard: (1) Back works normally; opening the home page signs the person out (server side); (2) while work is running, asks "Do you really want to exit?".
(function(){
  var body=document.body, authed=body.dataset.user==='1', leaving=false;
  var busy={};
  function active(){for(var k in busy){if(busy[k]>0)return true;}return false;}
  window.ECX_BUSY={start:function(k){busy[k]=(busy[k]||0)+1;},end:function(k){busy[k]=Math.max(0,(busy[k]||0)-1);},active:active};

  // ---- every state-changing request counts as "work in progress" until it finishes (the logout call itself does not)
  var prev=window.fetch;
  window.fetch=function(input,init){
    var method=((init&&init.method)||(input&&input.method)||'GET').toUpperCase();
    var url=(typeof input==='string'?input:(input&&input.url))||'';
    var track=method!=='GET'&&url.indexOf('/logout')<0;
    var id='req'+Math.random();
    if(track)ECX_BUSY.start(id);
    var p=prev.apply(this,arguments);
    if(track){var done=function(){ECX_BUSY.end(id)};p.then(done,done);}
    return p;
  };

  // ---- confirmation dialog
  var css=document.createElement('style');
  css.textContent='.xg-o{position:fixed;inset:0;background:rgba(5,8,25,.62);display:flex;align-items:center;justify-content:center;z-index:99999}'+
   '.xg-b{background:#fff;color:#1b2140;border-radius:16px;max-width:420px;width:calc(100% - 32px);padding:24px;box-shadow:0 20px 60px rgba(0,0,0,.45);font-family:inherit}'+
   '.xg-b h2{font-size:1.2rem;margin:0 0 8px;color:#1b2140 !important}.xg-b p{margin:0 0 18px;color:#555d7a !important;font-size:.93rem}.xg-a{display:flex;gap:10px;justify-content:flex-end}'+
   '.xg-a button{border:0;border-radius:10px;padding:.55rem 1.1rem;font-weight:700;cursor:pointer;font-size:.95rem}.xg-y{background:#e9ebf5;color:#1b2140}.xg-n{background:#6c5ce7;color:#fff}';
  document.head.appendChild(css);
  function confirmExit(cb){
    var o=document.createElement('div');o.className='xg-o';o.setAttribute('role','dialog');o.setAttribute('aria-modal','true');o.setAttribute('aria-labelledby','xg-t');
    o.innerHTML='<div class="xg-b"><h2 id="xg-t">Do you really want to exit?</h2><p>Something is still running. If you leave now, it may not finish.</p><div class="xg-a"><button class="xg-y" type="button">Yes</button><button class="xg-n" type="button">No</button></div></div>';
    document.body.appendChild(o);
    var yes=o.querySelector('.xg-y'),no=o.querySelector('.xg-n');
    function close(v){document.removeEventListener('keydown',key,true);o.remove();cb(v);}
    function key(e){if(e.key==='Escape'){e.preventDefault();close(false);}else if(e.key==='Tab'){e.preventDefault();(document.activeElement===yes?no:yes).focus();}}
    yes.onclick=function(){close(true)};no.onclick=function(){close(false)};
    document.addEventListener('keydown',key,true);no.focus();
  }

  // ---- leave: sign out (if signed in) and go to the login page, or to the home page after a confirmed exit
  function leave(dest){
    leaving=true;
    var go=function(){location.replace(dest)};
    if(!authed){go();return;}
    var t=setTimeout(go,2500);
    fetch('/logout',{method:'POST',keepalive:true,credentials:'same-origin'}).then(function(){clearTimeout(t);go()},function(){clearTimeout(t);go()});
  }

  // ---- Back button: works like a normal browser Back (goes to the previous page). The home page signs the person out on the server.
  // If the browser restores an old copy of a page from its cache, reload it so the server checks the sign-in again.
  window.addEventListener('pageshow',function(e){
    if(e.persisted){location.reload();}
  });

  // ---- leaving by a link, the logout button, closing the tab or reloading while work is running
  document.addEventListener('click',function(e){
    if(!active()||e.defaultPrevented||e.button||e.metaKey||e.ctrlKey||e.shiftKey)return;
    var a=e.target.closest&&e.target.closest('a[href]');
    if(!a||a.target==='_blank'||a.hasAttribute('download'))return;
    var h=a.getAttribute('href');if(!h||h.charAt(0)==='#'||h.indexOf('javascript:')===0)return;
    e.preventDefault();
    confirmExit(function(yes){if(yes){leaving=true;location.href=a.href;}});
  },true);
  document.addEventListener('submit',function(e){
    var f=e.target;if(!active()||!f||!f.classList.contains('logout'))return;
    e.preventDefault();confirmExit(function(yes){if(yes){leaving=true;f.submit();}});
  },true);
  window.addEventListener('beforeunload',function(e){if(active()&&!leaving){e.preventDefault();e.returnValue='';}});
})();

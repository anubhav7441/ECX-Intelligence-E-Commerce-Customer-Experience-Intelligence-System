// Adds the CSRF token to every state-changing fetch() and sends the person to the login page if the session has expired.
(function(){
  var m=document.querySelector('meta[name="csrf-token"]');var token=m?m.content:'';
  var orig=window.fetch;
  window.fetch=function(input,init){
    init=init||{};var method=(init.method||(input&&input.method)||'GET').toUpperCase();
    if(['POST','PUT','PATCH','DELETE'].indexOf(method)>-1){
      var h=new Headers(init.headers||{});if(token&&!h.has('X-CSRF-Token'))h.set('X-CSRF-Token',token);
      h.set('X-Requested-With','fetch');init.headers=h;
    }
    return orig.call(this,input,init).then(function(r){
      var url=(typeof input==='string'?input:(input&&input.url))||'';
      if(r.status===401&&url.indexOf('/api/demo/')<0&&location.pathname.indexOf('/login')<0&&document.body.dataset.auth==='1'){
        location.href='/login?next='+encodeURIComponent(location.pathname);
      }
      return r;
    });
  };
})();

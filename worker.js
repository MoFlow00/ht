/**
 * Tele-API V2. Existing KV binding: DATA_STORE.
 * Required secret: ADMIN_API_KEY.
 * Existing keys retained: latest_posts, blacklist, curated_explorer_data.
 */
const DEFAULT_PAGE_SIZE = 20;
const MAX_PAGE_SIZE = 100;
const MAX_SEARCH_LENGTH = 120;
const MAX_USERNAME_LENGTH = 64;
const RATE_WINDOW = 60;

function norm(value) { return String(value ?? "").trim().replace(/^@+/, "").toLowerCase(); }
function response(data, status=200, headers={}) {
  return new Response(JSON.stringify(data), {status, headers:{
    "Content-Type":"application/json; charset=utf-8", "X-Content-Type-Options":"nosniff", ...headers
  }});
}
function intParam(value, fallback, min, max) {
  const n = Number.parseInt(String(value ?? ""), 10);
  return Number.isFinite(n) ? Math.min(max, Math.max(min,n)) : fallback;
}
async function getJSON(kv,key,fallback) {
  try { const v=await kv.get(key,{type:"json"}); return v == null ? fallback : v; }
  catch { throw new Error("KV_READ_FAILED"); }
}
async function putJSON(kv,key,value) {
  try { await kv.put(key,JSON.stringify(value)); }
  catch { throw new Error("KV_WRITE_FAILED"); }
}
function hasLang(text,lang) {
  const s=String(text ?? ""), ar=/[\u0600-\u06FF]/u.test(s), en=/[a-z]/i.test(s);
  return lang==="ar" ? ar : lang==="en" ? (!ar&&en) : lang==="other" ? (!ar&&!en) : true;
}
function hashtags(posts) {
  const counts=new Map(), re=/#[\w\u0621-\u064A_]+/gu;
  for (const p of posts.slice(0,1000)) for (const m of String(p?.text ?? "").matchAll(re)) {
    const t=m[0].toLowerCase(); counts.set(t,(counts.get(t)||0)+1);
  }
  return [...counts].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).slice(0,10).map(x=>x[0]);
}
function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&");
}
function matches(post, q, tag, linkType) {
  const text = String(post?.text ?? "");
  const username = String(post?.username ?? "");
  if (q && !text.toLowerCase().includes(q) && !username.toLowerCase().includes(q)) return false;
  if (tag) {
    const re = new RegExp("(^|\\s)#" + escapeRegex(tag) + "(?![\\w\\u0621-\\u064A])", "iu");
    if (!re.test(text)) return false;
  }
  if (linkType === "links" && !/https?:\/\//i.test(text)) return false;
  if (linkType === "no-links" && /https?:\/\//i.test(text)) return false;
  return true;
}
async function admin(request,env) {
  const wanted=String(env.ADMIN_API_KEY||""), got=String(request.headers.get("X-Admin-Key")||"");
  return !!wanted && !!got && wanted.length===got.length && got===wanted;
}
async function limited(request,env,scope,max=12) {
  // KV counters are best-effort only. Configure Cloudflare Rate Limiting Rules for hard limits.
  const ip=request.headers.get("CF-Connecting-IP")||"unknown", key="ratelimit:"+scope+":"+ip;
  const now=Math.floor(Date.now()/1000), current=await getJSON(env.DATA_STORE,key,{count:0,until:now+RATE_WINDOW});
  if(current.until<=now) { await putJSON(env.DATA_STORE,key,{count:1,until:now+RATE_WINDOW}); return true; }
  if(current.count>=max) return false;
  await putJSON(env.DATA_STORE,key,{count:current.count+1,until:current.until}); return true;
}
function seededShuffle(items, seed) {
  let state = (Number(seed) >>> 0) || 1;
  const random = () => {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}
async function feedLinks(url,env) {
  const page=intParam(url.searchParams.get("page"),1,1,100000);
  const size=intParam(url.searchParams.get("page_size")||"100",100,1,MAX_PAGE_SIZE);
  // Preferred path: prebuilt catalog generated from every historical Parquet partition.
  const meta=await getJSON(env.DATA_STORE,"discovery_links_meta",null);
  if(meta && Number.isFinite(Number(meta.total)) && Number.isFinite(Number(meta.pages))) {
    const catalogPageSize=intParam(meta.page_size,500,1,1000);
    const start=(page-1)*size;
    const firstCatalogPage=Math.floor(start/catalogPageSize);
    const lastCatalogPage=Math.floor((start+size-1)/catalogPageSize);
    const pageKeys=[];
    for(let i=firstCatalogPage;i<=lastCatalogPage;i++) {
      if(i>=Number(meta.pages)) break;
      pageKeys.push("discovery_links_page_"+String(i).padStart(6,"0"));
    }
    const chunks=await Promise.all(pageKeys.map(key=>getJSON(env.DATA_STORE,key,[])));
    const catalog=chunks.flat();
    const offset=start-firstCatalogPage*catalogPageSize;
    const links=catalog.slice(offset,offset+size);
    return response({page,page_size:size,total:Number(meta.total),has_more:start+links.length<Number(meta.total),
      links,source:meta.source||"historical_catalog",generated_at:meta.generated_at||null},200,{"Cache-Control":"public, max-age=60, stale-while-revalidate=300"});
  }
  // Backward-compatible fallback until the first full-history catalog is published.
  const seed=intParam(url.searchParams.get("seed"),1,1,2147483647);
  const [posts,blockedValue]=await Promise.all([
    getJSON(env.DATA_STORE,"latest_posts",[]),getJSON(env.DATA_STORE,"blacklist",[])
  ]);
  if(!Array.isArray(posts)||!Array.isArray(blockedValue)) return response({error:"Stored data format invalid"},502);
  const blocked=new Set(blockedValue.map(norm).filter(Boolean));
  const seen=new Set(), links=[];
  const urlPattern=/(?:https?:\/\/|www\.)[^\s<>"'\]]+/gi;
  for(const post of posts) {
    const username=norm(post?.username);
    if(!username || blocked.has(username)) continue;
    const text=String(post?.text??"");
    const context=text.replace(urlPattern," ").replace(/\s+/g," ").trim().slice(0,220);
    for(const match of text.matchAll(urlPattern)) {
      let raw=String(match[0]).replace(/[),.;!?]+$/,"");
      if(raw.startsWith("www.")) raw="https://"+raw;
      let normalized;
      try {
        const parsed=new URL(raw);
        if(parsed.protocol!=="http:"&&parsed.protocol!=="https:") continue;
        if(parsed.hostname.toLowerCase().startsWith("www.")) parsed.hostname=parsed.hostname.slice(4);
        normalized=parsed.href;
      } catch { continue; }
      if(seen.has(normalized)) continue;
      seen.add(normalized);
      links.push({url:normalized,username,views:Number(post?.views)||0,post_id:String(post?.post_id||""),context});
    }
  }
  const shuffled=seededShuffle(links,seed);
  const start=(page-1)*size;
  return response({page,page_size:size,seed,total:shuffled.length,has_more:start+size<shuffled.length,
    links:shuffled.slice(start,start+size)},200,{"Cache-Control":"no-store"});
}
async function feed(url,env) {
  if (url.searchParams.get("mode")==="links") return feedLinks(url,env);
  const page=intParam(url.searchParams.get("page"),1,1,100000);
  const size=intParam(url.searchParams.get("page_size")||url.searchParams.get("limit"),DEFAULT_PAGE_SIZE,1,MAX_PAGE_SIZE);
  const q=String(url.searchParams.get("q")||"").trim().slice(0,MAX_SEARCH_LENGTH).toLowerCase();
  const lang=String(url.searchParams.get("lang")||"").trim().toLowerCase();
  const tag=String(url.searchParams.get("hashtag")||"").trim().replace(/^#/,"").slice(0,64);
  const linkType=String(url.searchParams.get("link_type")||"").trim().toLowerCase();
  if(tag && !/^[\w\u0621-\u064A_]+$/u.test(tag)) return response({error:"Invalid hashtag"},400);
  if(!["","links","no-links"].includes(linkType)) return response({error:"Invalid link_type"},400);
  const [posts,blockedValue]=await Promise.all([
    getJSON(env.DATA_STORE,"latest_posts",[]),getJSON(env.DATA_STORE,"blacklist",[])
  ]);
  if(!Array.isArray(posts)||!Array.isArray(blockedValue)) return response({error:"Stored data format invalid"},502);
  const blocked=new Set(blockedValue.map(norm).filter(Boolean));
  const filtered=posts.filter(p=>{
    const u=norm(p?.username);
    return u && !blocked.has(u) && hasLang(p?.text,lang) && matches(p,q,tag,linkType);
  });
  const start=(page-1)*size;
  return response({page,page_size:size,total:filtered.length,has_more:start+size<filtered.length,
    top_hashtags:(!q&&!lang&&!tag&&!linkType)?hashtags(posts.filter(p=>!blocked.has(norm(p?.username)))):[],
    posts:filtered.slice(start,start+size)
  },200,{"Cache-Control":"public, max-age=15, stale-while-revalidate=30"});
}
async function purge(request,url,env) {
  if(request.method!=="POST") return response({error:"Method not allowed"},405,{Allow:"POST"});
  if(!await limited(request,env,"purge",8)) return response({error:"Too many requests"},429,{"Retry-After":"60"});
  if(!await admin(request,env)) return response({error:"Unauthorized"},401);
  let value;
  if(request.headers.get("Content-Type")?.includes("application/json")) {
    try { value=(await request.json())?.username; } catch { return response({error:"Invalid JSON"},400); }
  } else value=url.searchParams.get("username"); // temporary legacy support for existing frontend
  const username=norm(value);
  if(!username||username.length>MAX_USERNAME_LENGTH||! /^[a-z0-9_]+$/i.test(username))
    return response({error:"Invalid username"},400);
  const block=await getJSON(env.DATA_STORE,"blacklist",[]);
  if(!Array.isArray(block)) return response({error:"Blacklist data format invalid"},502);
  const next=[...new Set(block.map(norm).filter(Boolean))];
  if(!next.includes(username)) next.push(username);
  // Persist blacklist before feed cleanup so subsequent crawler syncs fail closed.
  await putJSON(env.DATA_STORE,"blacklist",next);
  const posts=await getJSON(env.DATA_STORE,"latest_posts",[]);
  if(!Array.isArray(posts)) return response({error:"Feed data format invalid"},502);
  const clean=posts.filter(p=>norm(p?.username)!==username), removed=posts.length-clean.length;
  await putJSON(env.DATA_STORE,"latest_posts",clean);
  const oldAudit=await getJSON(env.DATA_STORE,"purge_audit",[]);
  const audit=Array.isArray(oldAudit)?oldAudit:[];
  audit.unshift({username,purged_posts:removed,timestamp:new Date().toISOString(),action:"purge"});
  await putJSON(env.DATA_STORE,"purge_audit",audit.slice(0,200));
  return response({success:true,purged:username,purged_posts:removed,remaining_posts:clean.length,blacklisted:true});
}
async function restore(request,env) {
  if(request.method!=="POST") return response({error:"Method not allowed"},405,{Allow:"POST"});
  if(!await admin(request,env)) return response({error:"Unauthorized"},401);
  let data; try { data=await request.json(); } catch { return response({error:"Invalid JSON"},400); }
  const username=norm(data?.username);
  if(!username||username.length>MAX_USERNAME_LENGTH||! /^[a-z0-9_]+$/i.test(username))
    return response({error:"Invalid username"},400);
  const block=await getJSON(env.DATA_STORE,"blacklist",[]);
  if(!Array.isArray(block)) return response({error:"Blacklist data format invalid"},502);
  await putJSON(env.DATA_STORE,"blacklist",[...new Set(block.map(norm).filter(x=>x&&x!==username))]);
  const old=await getJSON(env.DATA_STORE,"purge_audit",[]), audit=Array.isArray(old)?old:[];
  audit.unshift({username,timestamp:new Date().toISOString(),purged_posts:0,action:"restore"});
  await putJSON(env.DATA_STORE,"purge_audit",audit.slice(0,200));
  return response({success:true,restored:username,note:"Future crawler syncs will allow this channel. Deleted historical posts are not reconstructed."});
}
async function stats(env) {
  const [posts,block,explorer]=await Promise.all([
    getJSON(env.DATA_STORE,"latest_posts",[]),getJSON(env.DATA_STORE,"blacklist",[]),
    getJSON(env.DATA_STORE,"curated_explorer_data",{})
  ]);
  const rows=Array.isArray(posts)?posts:[], channels=new Set(rows.map(p=>norm(p?.username)).filter(Boolean));
  const languages={ar:0,en:0,other:0}; let links=0;
  for(const p of rows) {
    const s=String(p?.text??""), ar=/[\u0600-\u06FF]/u.test(s), en=/[a-z]/i.test(s);
    languages[ar?"ar":en?"en":"other"]++;
    links+=(s.match(/https?:\/\//gi)||[]).length;
  }
  return response({posts:rows.length,unique_channels:channels.size,blacklisted_channels:Array.isArray(block)?block.length:0,
    links,languages,explorer:{repos:Array.isArray(explorer?.top_repos)?explorer.top_repos.length:0,
    packages:Array.isArray(explorer?.top_packages)?explorer.top_packages.length:0,
    tech_channels:Array.isArray(explorer?.top_tech_channels)?explorer.top_tech_channels.length:0},
    generated_at:new Date().toISOString()},200,{"Cache-Control":"public, max-age=30"});
}
export default {
  async fetch(request,env) {
    const url=new URL(request.url);
    const allowed=String(env.ALLOWED_ORIGIN||"*").split(",").map(x=>x.trim()).filter(Boolean);
    const origin=request.headers.get("Origin")||"";
    const cors={"Access-Control-Allow-Origin":allowed.includes("*")?"*":(allowed.includes(origin)?origin:(allowed[0]||"null")),
      "Access-Control-Allow-Methods":"GET, POST, OPTIONS","Access-Control-Allow-Headers":"Content-Type, X-Admin-Key",
      "Access-Control-Max-Age":"86400","Vary":"Origin"};
    if(request.method==="OPTIONS") return new Response(null,{status:204,headers:cors});
    if(!env.DATA_STORE) return response({error:"DATA_STORE binding is missing"},500,cors);
    try {
      let out;
      switch(url.pathname) {
        case "/api/health":
          if(request.method!=="GET") out=response({error:"Method not allowed"},405,{Allow:"GET"});
          else out=response({status:"ok",service:"Tele-API",version:"2.0.0",timestamp:new Date().toISOString()});
          break;
        case "/api/feed":
          out=request.method==="GET"?await feed(url,env):response({error:"Method not allowed"},405,{Allow:"GET"}); break;
        case "/api/explorer":
          out=request.method==="GET"?response(await getJSON(env.DATA_STORE,"curated_explorer_data",{top_repos:[],top_packages:[],top_tech_channels:[]}),200,{"Cache-Control":"public, max-age=180"}):response({error:"Method not allowed"},405,{Allow:"GET"}); break;
        case "/api/verify_admin":
          if(request.method!=="GET") out=response({error:"Method not allowed"},405,{Allow:"GET"});
          else if(!await limited(request,env,"verify",20)) out=response({is_admin:false,error:"Too many attempts"},429,{"Retry-After":"60"});
          else out=response({is_admin:await admin(request,env)});
          break;
        case "/api/purge": out=await purge(request,url,env); break;
        case "/api/restore": out=await restore(request,env); break;
        case "/api/blacklist":
        case "/api/purge_audit": {
          if(request.method!=="GET") { out=response({error:"Method not allowed"},405,{Allow:"GET"}); break; }
          if(!await admin(request,env)) { out=response({error:"Unauthorized"},401); break; }
          if(url.pathname==="/api/blacklist") {
            const b=await getJSON(env.DATA_STORE,"blacklist",[]);
            out=response({blacklist:Array.isArray(b)?[...new Set(b.map(norm).filter(Boolean))].sort():[]});
          } else {
            const a=await getJSON(env.DATA_STORE,"purge_audit",[]);
            out=response({entries:Array.isArray(a)?a.slice(0,200):[]});
          }
          break;
        }
        case "/api/stats": out=request.method==="GET"?await stats(env):response({error:"Method not allowed"},405,{Allow:"GET"}); break;
        default: out=response({error:"Not found"},404);
      }
      const h=new Headers(out.headers); for(const [k,v] of Object.entries(cors)) h.set(k,v);
      return new Response(out.body,{status:out.status,headers:h});
    } catch(error) {
      const status=/^KV_(READ|WRITE)_FAILED/.test(String(error?.message))?503:500;
      return response({error:"Internal server error",request_id:request.headers.get("cf-ray")||null},status,cors);
    }
  }
};

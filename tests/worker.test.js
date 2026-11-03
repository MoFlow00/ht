import test from "node:test";
import assert from "node:assert/strict";
import worker from "../worker.js";
function env(data={}) {
  const map=new Map(Object.entries(data).map(([k,v])=>[k,JSON.stringify(v)]));
  return {ADMIN_API_KEY:"secret",_map:map,DATA_STORE:{
    async get(k,o={}) { const v=map.get(k); return v==null?null:(o.type==="json"?JSON.parse(v):v); },
    async put(k,v) { map.set(k,v); }
  }};
}
const req=(path,opts={})=>new Request("https://api.test"+path,opts);
test("health and 404",async()=>{const e=env();assert.equal((await worker.fetch(req("/api/health"),e)).status,200);assert.equal((await worker.fetch(req("/bad"),e)).status,404);});
test("feed clamps pagination and preserves old fields",async()=>{const e=env({latest_posts:[{username:"a",text:"hello #AI https://a"},{username:"b",text:"مرحبا"}],blacklist:[]});const d=await(await worker.fetch(req("/api/feed?page=-2&page_size=1"),e)).json();assert.equal(d.page,1);assert.equal(d.page_size,1);assert.equal(d.total,2);assert.ok(Array.isArray(d.top_hashtags));});
test("hashtag and link filters",async()=>{const e=env({latest_posts:[{username:"a",text:"#AI https://a"},{username:"b",text:"#Python"}],blacklist:[]});assert.equal((await(await worker.fetch(req("/api/feed?hashtag=AI&link_type=links"),e)).json()).total,1);assert.equal((await(await worker.fetch(req("/api/feed?link_type=no-links"),e)).json()).total,1);});
test("admin verification ignores URL key",async()=>{const e=env();const d=await(await worker.fetch(req("/api/verify_admin?key=secret"),e)).json();assert.equal(d.is_admin,false);});
test("purge writes blacklist, removes matching posts, writes audit",async()=>{const e=env({latest_posts:[{username:"keep"},{username:"target"},{username:"TARGET"}],blacklist:[]});const r=await worker.fetch(req("/api/purge",{method:"POST",headers:{"Content-Type":"application/json","X-Admin-Key":"secret"},body:JSON.stringify({username:"@Target"})}),e);const d=await r.json();assert.equal(d.purged_posts,2);assert.deepEqual(JSON.parse(e._map.get("blacklist")),["target"]);assert.equal(JSON.parse(e._map.get("latest_posts")).length,1);});
test("restore unblocks future crawler syncs",async()=>{const e=env({blacklist:["target"]});const r=await worker.fetch(req("/api/restore",{method:"POST",headers:{"Content-Type":"application/json","X-Admin-Key":"secret"},body:JSON.stringify({username:"target"})}),e);assert.equal(r.status,200);assert.deepEqual(JSON.parse(e._map.get("blacklist")),[]);});
test("purge requires POST and admin",async()=>{const e=env({latest_posts:[],blacklist:[]});assert.equal((await worker.fetch(req("/api/purge"),e)).status,405);assert.equal((await worker.fetch(req("/api/purge",{method:"POST"}),e)).status,401);});

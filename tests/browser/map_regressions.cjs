const {chromium}=require(process.env.PLAYWRIGHT_PATH || 'playwright');
const fs=require('fs');
const path=require('path');
const file=process.argv[2]||'docs/index.html';
const channel=process.argv[3]||'chromium';
const baseline=process.argv.includes('--baseline');
const zlib=require('zlib');
const html=fs.readFileSync(file,'utf8');
const packed=html.match(/\{\"encoding\":\"gzip-base64\",\"data\":\"([A-Za-z0-9+/=]+)\"\}/);
const atlas=packed?JSON.parse(zlib.gunzipSync(Buffer.from(packed[1],'base64')).toString('utf8')):null;
const result={browser:channel,baseline,checks:[],errors:[]};
const add=(name,pass,details)=>result.checks.push({name,pass:Boolean(pass),details});
const view=page=>page.locator('#territory-map').evaluate(e=>({box:[e.viewBox.baseVal.x,e.viewBox.baseVal.y,e.viewBox.baseVal.width,e.viewBox.baseVal.height],scale:e.getScreenCTM().a}));
async function target(page){return page.evaluate(()=>{
  const svg=document.querySelector('#territory-map'), box=svg.getBoundingClientRect();
  const selected=document.querySelector('#territory-map .selected')?.dataset.index;
  const paths=[...svg.querySelectorAll('.map-shape')].filter(p=>p.dataset.index!==selected);
  for(const p of paths){
    const b=p.getBBox(),m=p.getScreenCTM();
    if(b.width*m.a<12||b.height*m.d<10||b.width*m.a>400)continue;
    for(const fx of [.5,.35,.65,.2,.8])for(const fy of [.5,.35,.65,.2,.8]){
      const q=new DOMPoint(b.x+b.width*fx,b.y+b.height*fy).matrixTransform(m);
      if(q.x<box.left+12||q.x>box.right-80||q.y<Math.max(box.top+20,70)||q.y>Math.min(box.bottom-12,innerHeight-12))continue;
      if(document.elementFromPoint(q.x,q.y)===p)return {x:q.x,y:q.y,index:p.dataset.index};
    }
  }
  throw new Error('No visible distinct municipal interior for real click');
});}
(async()=>{
const browser=await chromium.launch(channel==='chromium'?{headless:true}:{channel,headless:true});
try{
  const page=await browser.newPage({viewport:{width:1440,height:1080}});
  page.on('pageerror',e=>result.errors.push(String(e)));
  const loadStart=Date.now();
  await page.goto('file:///'+path.resolve(file).replace(/\\/g,'/'));
  await page.locator('#territory-map .map-shape').first().waitFor();
  await page.locator('#territory-map').scrollIntoViewIfNeeded();
  result.local_load_to_map_ms=Date.now()-loadStart;
  await page.evaluate(()=>{window.__firstMapPath=document.querySelector('#territory-map .map-shape')});
  add('no_native_map_titles',await page.locator('#territory-map title').count()===0);
  for(const zoom of [1,2,4,8]){
    await page.click('#map-zoom-reset');
    for(let z=1;z<zoom;z*=2)await page.click('#map-zoom-in');
    const point=await target(page),before=await view(page);
    await page.mouse.move(point.x,point.y);
    add('hover_disabled_by_default_zoom_'+zoom,await page.locator('#map-tooltip').isHidden());
    await page.mouse.click(point.x,point.y);
    const after=await view(page);
    add('real_pointer_click_zoom_'+zoom,await page.locator('#territory-map .selected').getAttribute('data-index')===point.index,{expected:point.index});
    add('selection_preserves_view_zoom_'+zoom,before.box.every((v,i)=>Math.abs(v-after.box[i])<1e-7),{before:before.box,after:after.box});
  }
  add('map_geometry_nodes_reused_on_selection',await page.evaluate(()=>window.__firstMapPath===document.querySelector('#territory-map .map-shape')));
  if(!baseline){
    await page.selectOption('#ego-period','1');
    add('monthly_network_has_15_edges',await page.locator('#ego-network line[data-edge-id]').count()===15);
    add('monthly_network_has_16_nodes',await page.locator('#ego-network g[data-node-index]').count()===16);
    if(atlas?.contest?.comparability){
      const rows=await page.locator('#neighbors tr').evaluateAll(nodes=>nodes.map(n=>({id:n.dataset.entityId,ratio:+n.dataset.expenseRatio,type:n.querySelector('.territory-type')?.textContent})));
      const selected=+await page.locator('#quick').inputValue(),own=atlas.contest.comparability[atlas.entities[selected].id].expense_2023;
      add('neighbor_level_ratios_match_source',rows.length===15&&rows.every(r=>Math.abs(r.ratio-atlas.contest.comparability[r.id].expense_2023/own)<1e-12));
      add('neighbor_types_match_source',rows.every(r=>r.type===atlas.contest.comparability[r.id].municipal_district_type));
      const nodes=await page.locator('#ego-network g[data-node-index]').evaluateAll(ns=>ns.map(n=>+n.dataset.nodeIndex)),peers=nodes.filter(i=>i!==selected),period=1,scale=atlas.meta.ratio_iqr;
      const top=i=>new Set(atlas.entities.map((e,j)=>({j,d:i===j?Infinity:e.ratios[period].reduce((sum,v,c)=>sum+((Math.log(v)-Math.log(atlas.entities[i].ratios[period][c]))/scale[c])**2/5,0)})).sort((a,b)=>a.d-b.d||atlas.entities[a.j].id.localeCompare(atlas.entities[b.j].id)).slice(0,15).map(v=>v.j));
      const topSets=new Map(peers.map(i=>[i,top(i)])),expected=[];
      const edgeKey=(a,b)=>[a,b].sort().join('|');
      for(let a=0;a<peers.length;a++)for(let b=a+1;b<peers.length;b++)if(topSets.get(peers[a]).has(peers[b])||topSets.get(peers[b]).has(peers[a]))expected.push(edgeKey(atlas.entities[peers[a]].id,atlas.entities[peers[b]].id));
      const actual=await page.locator('#ego-network line[data-context-edge]').evaluateAll(es=>es.map(e=>[e.dataset.contextSource,e.dataset.contextTarget].sort().join('|')));
      add('induced_context_edges_match_full_cohort_knn',JSON.stringify(actual.sort())===JSON.stringify(expected.sort()),{edges:actual.length});
      add('context_edges_do_not_capture_pointer',await page.locator('#ego-network line[data-context-edge]').evaluateAll(es=>es.every(e=>getComputedStyle(e).pointerEvents==='none')));
    }else{add('comparability_payload_available',false);}

    add('monthly_network_reports_retained_links',(await page.locator('#ego-summary').textContent()).includes('Сохранилось'));
    const neighbor=page.locator('#ego-network g[data-node-index]').nth(1),neighborId=await neighbor.getAttribute('data-node-index');
    await neighbor.click();
    add('network_node_selects_municipality',await page.locator('#quick').inputValue()===neighborId);
    await page.selectOption('#map-model','joint_road_k2');
    add('joint_map_has_two_groups',await page.locator('#map-legend .swatch').count()===2);
    add('joint_map_does_not_imply_2024_forecast',await page.locator('#map-year').isDisabled()&&(await page.locator('#map-model-note').textContent()).includes('31.12.2024'));
    await page.selectOption('#map-model','joint_road_k4');
    add('joint_map_has_four_groups',await page.locator('#map-legend .swatch').count()===4);
    if(atlas?.contest?.round2){
      for(const k of [2,4]){
        await page.selectOption('#map-model',`dmon_k${k}_seed1729`);
        add(`dmon_k${k}_legend_matches_requested_groups`,await page.locator('#map-legend .swatch').count()===k);
      }
      add('dmon_map_disables_future_year',await page.locator('#map-year').isDisabled());
      add('temporal_grid_displays_all_five_weights',await page.locator('#temporal-summary table').first().locator('tbody tr').count()===5);
      add('temporal_grid_has_three_seeds_each',atlas.contest.temporal.variants.length===5&&atlas.contest.temporal.variants.every(v=>v.seeds===3));
      add('regional_comparison_displays_three_models',await page.locator('#temporal-summary table').nth(1).locator('tbody tr').count()===3);
    }
    await page.selectOption('#map-model','frozen');
    add('frozen_map_restores_year_control',await page.locator('#map-year').isEnabled());
  }
  const beforeSearch=await view(page);
  const current=await page.locator('#quick').inputValue();
  await page.selectOption('#quick',current==='0'?'1':'0');
  const afterSearch=await view(page);
  add('dropdown_selection_preserves_zoom',beforeSearch.box.every((v,i)=>Math.abs(v-afterSearch.box[i])<1e-7),{before:beforeSearch.box,after:afterSearch.box});
  await page.click('#map-zoom-reset');await page.click('#map-zoom-in');await page.click('#map-zoom-in');
  const before=await view(page),rect=await page.locator('#territory-map').boundingBox();
  const sx=rect.x+rect.width*.55,sy=rect.y+rect.height*.55,dx=72,dy=28;
  const selected=await page.locator('#territory-map .selected').getAttribute('data-index');
  await page.mouse.move(sx,sy);await page.mouse.down();await page.mouse.move(sx+dx,sy+dy,{steps:9});await page.mouse.up();
  const after=await view(page);
  const errorX=(after.box[0]-before.box[0])*before.scale+dx,errorY=(after.box[1]-before.box[1])*before.scale+dy;
  add('pan_tracks_cursor_within_one_pixel',Math.abs(errorX)<1&&Math.abs(errorY)<1,{errorX,errorY});
  add('drag_does_not_select',await page.locator('#territory-map .selected').getAttribute('data-index')===selected);
  add('no_text_selection_after_drag',await page.evaluate(()=>!getSelection().toString()));
  const anchored=await view(page);
  await page.selectOption('#map-year','2024');await page.selectOption('#map-mode','relative');
  const redrawn=await view(page);
  add('year_and_mode_preserve_pan',anchored.box.every((v,i)=>Math.abs(v-redrawn.box[i])<1e-7));
  if(!baseline){
    const toggle=page.locator('#map-hover-tips');
    if(await toggle.count()){
      await toggle.check();const point=await target(page);await page.mouse.move(point.x,point.y);
      add('opt_in_hover_shows_description',await page.locator('#map-tooltip').isVisible());
      await toggle.uncheck();
      add('opt_out_hides_and_removes_tooltip',await page.locator('#map-tooltip').evaluate(e=>e.hidden&&getComputedStyle(e).pointerEvents==='none'));
    }else add('hover_toggle_exists',false);
  }
  const selectedPath=page.locator('#territory-map .selected');
  await selectedPath.focus();
  await page.keyboard.press('ArrowRight');
  const keyboardIndex=await page.evaluate(()=>document.activeElement.dataset.index);
  await page.keyboard.press('Enter');
  await page.waitForFunction(i=>document.querySelector('#territory-map .selected')?.dataset.index===i,keyboardIndex);
  add('keyboard_arrow_and_enter_select',await page.locator('#territory-map .selected').getAttribute('data-index')===keyboardIndex);
  await page.locator('#territory-map').scrollIntoViewIfNeeded();
  const cancelRect=await page.locator('#territory-map').boundingBox();
  const cx=cancelRect.x+cancelRect.width*.5,cy=cancelRect.y+cancelRect.height*.5;
  await page.evaluate(()=>document.querySelector('#territory-map').addEventListener('pointerdown',e=>window.__mapQAId=e.pointerId,{once:true}));
  await page.mouse.move(cx,cy);await page.mouse.down();
  const primaryBefore=await view(page);
  await page.evaluate(({x,y})=>document.querySelector('#territory-map').dispatchEvent(new PointerEvent('pointermove',{pointerId:999,isPrimary:false,clientX:x+60,clientY:y+30,bubbles:true})),{x:cx,y:cy});
  const secondaryAfter=await view(page);
  add('secondary_pointer_does_not_move_drag',primaryBefore.box.every((v,i)=>Math.abs(v-secondaryAfter.box[i])<1e-7));
  await page.mouse.move(cx+20,cy+10,{steps:3});
  await page.evaluate(()=>document.querySelector('#territory-map').dispatchEvent(new PointerEvent('pointercancel',{pointerId:window.__mapQAId,isPrimary:true,bubbles:true})));
  const canceled=await view(page);
  await page.mouse.move(cx+70,cy+30,{steps:3});await page.mouse.up();
  const afterCanceled=await view(page);
  add('pointercancel_releases_drag',canceled.box.every((v,i)=>Math.abs(v-afterCanceled.box[i])<1e-7));
  await page.setViewportSize({width:390,height:844});
  await page.locator('#territory-map').scrollIntoViewIfNeeded();
  add('mobile_no_horizontal_overflow',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  if(!baseline){await page.locator('#map-section').screenshot({path:'artifacts/map-qa/'+channel+'-map.png'});}
  await page.close();
  const touch=await browser.newPage({viewport:{width:390,height:844},hasTouch:true,isMobile:true});
  touch.on('pageerror',e=>result.errors.push(String(e)));
  await touch.goto('file:///'+path.resolve(file).replace(/\\/g,'/'));
  await touch.locator('#territory-map').scrollIntoViewIfNeeded();
  await touch.locator('#map-zoom-in').tap();await touch.locator('#map-zoom-in').tap();
  const touchPoint=await target(touch),touchBefore=await view(touch);
  await touch.touchscreen.tap(touchPoint.x,touchPoint.y);
  const touchAfter=await view(touch);
  add('mobile_real_touch_selects_zoomed_municipality',await touch.locator('#territory-map .selected').getAttribute('data-index')===touchPoint.index);
  add('mobile_touch_preserves_zoom',touchBefore.box.every((v,i)=>Math.abs(v-touchAfter.box[i])<1e-7));
  await touch.close();
  add('no_javascript_errors',result.errors.length===0,result.errors);
}finally{await browser.close();}
result.passed=result.checks.every(x=>x.pass);
fs.mkdirSync('artifacts/map-qa',{recursive:true});
fs.writeFileSync(`artifacts/map-qa/${channel}-${baseline?'before':'after'}.json`,JSON.stringify(result,null,2));
console.log(JSON.stringify(result));
if(!baseline&&!result.passed)process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1});

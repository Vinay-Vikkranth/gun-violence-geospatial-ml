const fs=require('fs'),vm=require('vm'),assert=require('assert');
const nodes=new Map();const node=id=>{if(!nodes.has(id))nodes.set(id,{innerHTML:'',textContent:'',value:'',disabled:false,hidden:false});return nodes.get(id)};
const context={document:{getElementById:node,querySelectorAll:()=>[]},console,setTimeout,clearTimeout,AbortController,URLSearchParams};vm.createContext(context);
vm.runInContext(fs.readFileSync('digital_twin/frontend/app.js','utf8').replace('act(boot)();',''),context);
(async()=>{
 await vm.runInContext(`(async()=>{
 renderControls=()=>{};renderResult=()=>{};drawMap=()=>{};populateModelPicker=()=>{};renderModelScores=()=>{};tab=()=>{};
 state.overview={features:{Schools:{label:'Schools'}},models:{random_forest:{name:'Random Forest'}}};
 state.detail={id:'TEST',score:.9,model_scores:{random_forest:.25},values:{Schools:2},quality_flags:[]};state.model='random_forest';state.result={score:.8};state.overrides={Schools:3};
 reset();if(state.result.score!==.25||state.result.baseline!==.25)throw Error('Reset uses wrong model');
 let body;api=async(path,options)=>{body=options.body;return {points:[{value:1,score:.3}],note:''}};
 $('sensitivityFeature').value='Schools';await sensitivity();if(body.model!=='random_forest')throw Error('Sensitivity lost selected model');
 let finish;api=()=>new Promise(resolve=>finish=resolve);const pending=runOptimize();state.revision++;finish({overrides:{Schools:9},result:{score:.1},changes:[]});await pending;
 if(state.overrides.Schools===9)throw Error('Stale optimization applied');
 api=async()=>({overrides:{Schools:1},result:{score:.15,values:{Schools:1},warnings:[],model:'random_forest'},optimized_score:.15,delta:-.1,changes:[],interpretation:''});
 state.overrides={Schools:3};await runOptimize();if(state.result.values.Schools!==1||state.history.at(-1).Schools!==3)throw Error('Optimization snapshot/undo incorrect');
 let reopened;state.saved=[{id:'saved',grid_id:'TEST',model:'random_forest',overrides:{Schools:1},name:'Example'}];state.model='lightgbm';showView=()=>{};selectHex=async()=>{reopened=state.model};queuePrediction=()=>{};await openSaved('saved');if(reopened!=='random_forest')throw Error('Saved model not restored');
 const chart=lineChart([{x:0,y:.1},{x:.1,y:.2}]);if(!chart.includes('362,'))throw Error('Small-range chart compressed');
 })()`,context);
 console.log('Passed: selected-model reset, sensitivity model, stale optimizer rejection, optimizer values and undo, saved model restore, fractional chart scale');
})().catch(e=>{console.error(e);process.exitCode=1});

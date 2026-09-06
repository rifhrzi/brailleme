// Execute both exact frozen TypeScript versions; no app/native/model execution.
const fs=require('node:fs');
const path=require('node:path');
const Module=require('node:module');
const assert=require('node:assert/strict');
const crypto=require('node:crypto');
const ts=require('typescript');
const out=__dirname;
const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
function loader(side){
  const root=path.join(out,side);const cache=new Map();
  function load(file){
    if(cache.has(file))return cache.get(file).exports;
    const m=new Module(file,module);cache.set(file,m);m.filename=file;m.paths=Module._nodeModulePaths(path.dirname(file));
    m.require=name=>name.startsWith('@/')?load(path.join(root,'src',name.slice(2)+'.ts')):Module.prototype.require.call(m,name);
    m._compile(ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,file);
    return m.exports;
  }
  return load(path.join(root,'src/ml/word-decoder.ts'));
}
const baseline=loader('source-original'),privateDecoder=loader('source-private');
const cases=JSON.parse(fs.readFileSync(path.join(out,'cases.json'),'utf8'));
const python=JSON.parse(fs.readFileSync(path.join(out,'python-results.json'),'utf8'));
function comparable(result){return {text:result.status==='success'?result.text:'',reason:result.status==='success'?'':result.reason,
  boxes:result.boxes.map(b=>[b.x,b.y,b.x+b.width,b.y+b.height]),masks:result.cells.map(c=>c.mask)};}
let geometryMax=0;
const rows=[];
for(let index=0;index<cases.length;index++){
  const c=cases[index],tensor=new Float32Array(48*120*11);
  for(const row of c.rows)tensor.set(row.values,(row.gy*120+row.gx)*11);
  const [x,y,right,bottom]=c.selection,selection={x,y,width:right-x,height:bottom-y};
  const row={name:c.name};
  for(const [label,decoder] of [['baseline',baseline],['private',privateDecoder]]){
    const actual=comparable(decoder.decodeWordOutput(tensor,c.thresholds,selection));const expected=python[index][label];
    assert.equal(actual.text,expected.text,c.name);assert.equal(actual.reason,expected.reason,c.name);assert.deepEqual(actual.masks,expected.masks,c.name);
    assert.equal(actual.boxes.length,expected.boxes.length,c.name);
    actual.boxes.forEach((box,i)=>box.forEach((v,j)=>{const delta=Math.abs(v-expected.boxes[i][j]);geometryMax=Math.max(geometryMax,delta);assert.ok(delta<1e-14,c.name);}));
    row[label]=actual;
  }
  rows.push(row);
}
for(const value of [NaN,-1,2]){const malformed=new Float32Array(48*120*11);malformed[0]=value;assert.throws(()=>privateDecoder.decodeWordOutput(malformed,cases[0].thresholds));}
const result={status:'passed',cases:cases.length,python_typescript_geometry_max_abs:geometryMax,
  original_and_private_decisions_exact:true,script_sha256:sha(__filename),typescript_version:ts.version,
  prototype_spec_sha256:sha(path.join(out,'prototype-spec.json')),model_execution:false,rows};
const dest=path.join(out,'typescript-test-report.json');assert.ok(!fs.existsSync(dest));fs.writeFileSync(dest,JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({status:'passed',cases:cases.length,geometryMax}));

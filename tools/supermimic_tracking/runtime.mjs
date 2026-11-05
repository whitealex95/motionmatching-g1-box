// Adapter for the pinned 2026-10-21 SuperMimic demo WebAssembly runtime.
// The downloaded engine performs inference. No weights are exported or converted.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const dir = process.argv[2];
const hashes = {
  'engine.wasm': 'a48c757becd0f2ed7c658fc18b9d765df25b63ba944ffd0d74e09b2053cf4930',
  'weights.bin': 'f1d923a3084955869c75f5bc06fc25e64d854dbeecd33d139b689eb62f73d87f',
  'engine.cfg': '45153ee8d52d5f9def4ea9fb3b1dece6ecd03b23ac4a139f043389cd69fd92d8',
  'robot.json': '2dd14d28134d547ca2596d800abf16b5ab4174a51fc52ea3dfb063449cbe8f26',
  'motion00.bin': '570ebaa69b774587c8430203056e32029568ae443b76521fd312840217d128bb',
};
const files = {};
for (const [name, hash] of Object.entries(hashes)) {
  const b = fs.readFileSync(path.join(dir, name));
  if (crypto.createHash('sha256').update(b).digest('hex') !== hash)
    throw Error(`Unsupported demo version: ${name}. Refusing an unverified memory layout.`);
  files[name] = b;
}
// Host values supplied by the demo's public worker and robot asset.
const host = [3914829593, 1407519816, 2010095901, 1826096763, 3802974929, 1132632030]
  .map((v, i) => (v ^ files['robot.json'].readUInt32LE(1424 + 52*i)) >>> 0);
const {instance} = await WebAssembly.instantiate(files['engine.wasm'], {a: {b: i => host[i]}});
const e = instance.exports;
e.a0(16);
const bytes = () => new Uint8Array(e.m.buffer);
const view = () => new DataView(e.m.buffer);
const chunk = e.a2(), input = e.a1();
for (let i=0; i<files['weights.bin'].length; i+=chunk) {
  const b=files['weights.bin'].subarray(i,i+chunk);
  bytes().set(b,input);
  if(e.a3(b.length)!==0) throw Error('Weight loading failed');
}
if(e.a4()!==0) throw Error('Weight finalization failed');
const cfg=e.a0(files['engine.cfg'].length);
bytes().set(files['engine.cfg'],cfg);
if(e.a5(cfg,files['engine.cfg'].length)!==0) throw Error('Configuration loading failed');
const motionInput=e.a15(files['motion00.bin'].length);
bytes().set(files['motion00.bin'],motionInput);
if(e.a16(0,motionInput,files['motion00.bin'].length)!==0) throw Error('Motion loading failed');
e.a7(1); // second instance: actor receiving native MuJoCo state
e.a6();
if(e.a24(36,35)!==0) throw Error('Unexpected G1 state dimensions');
const root=view().getUint32(1057640,true);
const actor=root+5920;
const motion=view().getUint32(root+11908,true);
const dims=[29,29,90,120];
const original=dims.map((d,i) => new Float32Array(e.m.buffer,
  view().getUint32(motion+12*i+4,true),view().getUint32(motion+12*i+8,true)).slice());
if(view().getUint32(motion+72,true)!==517) throw Error('Unexpected motion layout');
const statePtr=e.a25();
const initialState=() => Array.from(new Float64Array(e.m.buffer,e.a26(0),71));
const infer=state => {
  new Float64Array(e.m.buffer,statePtr,71).set(state);
  return Float64Array.from(new Float32Array(e.m.buffer,e.a28(),29));
};
// Own reference allocations, kept alive by the engine allocator. Only these
// user-supplied motion arrays and the actor's reference playhead are changed.
const refs=dims.map(d => e.a0(46*d*4));
function install() {
  for(let i=0;i<4;i++) {
    view().setUint32(motion+12*i,46*dims[i],true);
    view().setUint32(motion+12*i+4,refs[i],true);
    view().setUint32(motion+12*i+8,46*dims[i],true);
  }
  view().setUint32(motion+72,46,true);
}
function playheadZero() {
  view().setUint32(actor+5620,0,true);
  // Python already applies heading alignment to the reference quaternion.
  new Float32Array(e.m.buffer,actor+2560,4).set([1,0,0,0]);
}
if(process.argv.includes('--self-test')) {
  const state=initialState();
  const expected=[];
  for(let t=0;t<20;t++) expected.push(infer(state));
  e.a6();
  install();
  let maxError=0;
  for(let t=0;t<20;t++) {
    dims.forEach((d,i) => new Float32Array(e.m.buffer,refs[i],46*d).set(original[i].subarray(t*d,(t+46)*d)));
    playheadZero();
    const actual=infer(state);
    actual.forEach((v,i)=>maxError=Math.max(maxError,Math.abs(v-expected[t][i])));
  }
  if(!Number.isFinite(maxError) || maxError>1e-6) throw Error(`Reference-window parity failed: ${maxError}`);
  console.log(JSON.stringify({test:'native motion versus streamed reference, 20 actor steps',max_abs_target_error:maxError}));
  process.exit(0);
}
install();
// Custom references initialize from rest during the runners' settle phase.
// Remove the bundled clip's initial body velocities from reset state.
for(const i of [4,5]) {
  const p=view().getUint32(motion+12*i+4,true);
  new Float32Array(e.m.buffer,p,90).fill(0);
}
// Binary request: reset float64, 71 float64 qpos/qvel, four float32 arrays.
const nbytes=72*8+46*dims.reduce((a,b)=>a+b,0)*4;
const request=Buffer.alloc(nbytes);
fs.writeSync(1,'READY\n');
while(true) {
  let at=0;
  while(at<nbytes) {
    const n=fs.readSync(0,request,at,nbytes-at,null);
    if(n===0) {if(at) throw Error('Truncated state request'); process.exit(0);}
    at+=n;
  }
  let offset=72*8;
  dims.forEach((d,i)=> {
    bytes().set(request.subarray(offset,offset+46*d*4),refs[i]);
    offset+=46*d*4;
  });
  if(request.readDoubleLE(0)!==0) e.a6();
  playheadZero();
  const target=infer(new Float64Array(request.buffer,request.byteOffset+8,71));
  if(!Array.from(target).every(Number.isFinite)) throw Error('Non-finite policy target');
  fs.writeSync(1,Buffer.from(target.buffer));
}

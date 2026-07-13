// Interactive G1 motion-matching demo (Three.js). Loads the exported database, runs the
// JS motion matcher (mm.js, a verified 1:1 port of the Python controller) at a fixed 30 Hz,
// forward-kinematics the result, and draws the G1 as an articulated capsule skeleton.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MotionMatcher, loadDB } from './mm.js';
import { fk } from './fk.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);   // MuJoCo is z-up; render in world coords directly

// Page config, set by the host index.html (see medicine/index.html). Both pages run this same
// script off the same motion database and differ ONLY in which box skin they load -- the box
// mesh is purely visual, so the /medicine build costs one extra mesh + texture, not a copy of
// the 35 MB database.
const CFG = window.DEMO_CONFIG || {};
const DATA = CFG.data || './data';
const BOX = CFG.box || 'boxmesh';
const hud = document.getElementById('hud');
const setHud = (t) => { hud.textContent = t; };

async function loadJSON(u) { return (await fetch(u)).json(); }
async function loadBin(u) { return (await fetch(u)).arrayBuffer(); }

async function boot() {
  setHud('loading G1 model + motion database (~30 MB)...');
  const [model, meta, bin, meshMeta, meshBin, boxMeta, boxBin] = await Promise.all([
    loadJSON(`${DATA}/model.json`), loadJSON(`${DATA}/mm.json`), loadBin(`${DATA}/mm.bin`),
    loadJSON(`${DATA}/mesh.json`), loadBin(`${DATA}/mesh.bin`),
    loadJSON(`${DATA}/${BOX}.json`), loadBin(`${DATA}/${BOX}.bin`),
  ]);
  const A = loadDB(meta, bin);
  const mm = new MotionMatcher(meta, A);
  start(model.bodies, mm, meshMeta, meshBin, boxMeta, boxBin);
}

function start(bodies, mm, meshMeta, meshBuf, boxMeta, boxBuf) {
  // ---- renderer / scene / camera (z-up) ----
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.shadowMap.enabled = true;
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x9a9286);                  // warm greige (non-blue)
  scene.fog = new THREE.Fog(0x9a9286, 35, 110);                 // far fog: near ground stays solid

  const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.05, 200);
  camera.up.set(0, 0, 1);
  camera.position.set(2.6, -2.6, 1.7);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.8);
  controls.enablePan = false;

  // ---- lights + floor ----
  scene.add(new THREE.HemisphereLight(0xffffff, 0x554b40, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(4, -6, 8); sun.castShadow = true;
  sun.shadow.camera.top = 8; sun.shadow.camera.bottom = -8;
  sun.shadow.camera.left = -8; sun.shadow.camera.right = 8;
  sun.shadow.mapSize.set(2048, 2048);
  scene.add(sun);

  // Warm checker floor (procedural texture) -- clearly visible ground + motion reference.
  const cv = document.createElement('canvas'); cv.width = cv.height = 256;
  const cx = cv.getContext('2d');
  cx.fillStyle = '#7a7165'; cx.fillRect(0, 0, 256, 256);        // light square
  cx.fillStyle = '#5e564c'; cx.fillRect(0, 0, 128, 128); cx.fillRect(128, 128, 128, 128);  // dark
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(100, 100);                                     // 2 squares / tile -> 1 m squares
  tex.anisotropy = 8;
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(200, 200),
    new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95 }));
  floor.receiveShadow = true;
  scene.add(floor);

  // ---- G1 full mesh: one Three.js Group per body, holding its visual meshes (body-local).
  //      FK only moves the groups each frame; the geometry inside is static. ----
  const robot = new THREE.Group();
  scene.add(robot);
  const bodyGroups = bodies.map(() => { const g = new THREE.Group(); robot.add(g); return g; });
  for (const gm of meshMeta.geoms) {
    const pos = new Float32Array(meshBuf, gm.vstart * 12, gm.vcount * 3);             // 3*4 bytes
    const idx = new Uint16Array(meshBuf, meshMeta.idx_byte_offset + gm.istart * 2, gm.icount);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(gm.rgba[0], gm.rgba[1], gm.rgba[2]),
      metalness: 0.55, roughness: 0.45, flatShading: true });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true; mesh.receiveShadow = true;
    bodyGroups[gm.body].add(mesh);
  }
  const _yAxis = new THREE.Vector3(0, 1, 0);

  // ---- interactive box: one mesh group placed directly from the matcher's box pose
  //      (free body, NOT in the FK tree). qpos[36:43] = world position + quaternion (wxyz). ----
  const boxGroup = new THREE.Group(); scene.add(boxGroup);
  {
    const pos = new Float32Array(boxBuf, 0, boxMeta.nverts * 3);
    const idx = new Uint16Array(boxBuf, boxMeta.idx_byte_offset, boxMeta.nidx);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    // rgba is the material tint (it multiplies the texture, exactly as in the MuJoCo scene).
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(boxMeta.rgba[0], boxMeta.rgba[1], boxMeta.rgba[2]),
      metalness: 0.1, roughness: 0.8, flatShading: true });
    if (boxMeta.uv_byte_offset >= 0 && boxMeta.texture) {   // printed carton (see boxmesh.png)
      const uv = new Float32Array(boxBuf, boxMeta.uv_byte_offset, boxMeta.nverts * 2);
      geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
      const tex = new THREE.TextureLoader().load(`${DATA}/${boxMeta.texture}`);
      tex.colorSpace = THREE.SRGBColorSpace;                // else the kraft reads washed out
      tex.anisotropy = 8;                                   // keep the print legible at grazing angles
      mat.map = tex;
      mat.needsUpdate = true;
    }
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true; mesh.receiveShadow = true;
    boxGroup.add(mesh);
  }

  // ---- command-trajectory gizmo (red spheres + facing sticks) ----
  const gizmo = new THREE.Group(); scene.add(gizmo);
  const red = new THREE.MeshBasicMaterial({ color: 0xe21818 });
  const gizSph = [0, 1, 2].map(() => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 10), red); gizmo.add(m); return m; });
  const gizStk = [0, 1, 2].map(() => { const m = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1, 6), red); gizmo.add(m); return m; });

  // ---- keyboard ----
  const held = new Set();
  let shift = false;
  addEventListener('keydown', (e) => {
    shift = e.shiftKey;
    const k = e.code;
    if (k === 'Space') { mm.reset(); e.preventDefault(); }
    else if (k === 'KeyJ') mm.triggerJump();
    else if (k === 'KeyB') mm.triggerBox();
    else if (k === 'KeyT') gizmo.visible = !gizmo.visible;
    else held.add(k);
    if (k.startsWith('Arrow')) e.preventDefault();
  });
  addEventListener('keyup', (e) => { shift = e.shiftKey; held.delete(e.code); });
  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  function command() {
    // camera-relative ground frame
    const d = new THREE.Vector3(); camera.getWorldDirection(d);
    let fx = d.x, fy = d.y; const fn = Math.hypot(fx, fy) || 1; fx /= fn; fy /= fn;
    const rx = fy, ry = -fx;                                     // right = forward rotated -90deg
    const fwd = [fx, fy, 0], right = [rx, ry, 0];
    const acc = (v, s) => [v[0] + s[0], v[1] + s[1], 0];
    let move = [0, 0, 0], face = [0, 0, 0];
    if (held.has('KeyW')) move = acc(move, fwd);
    if (held.has('KeyS')) move = acc(move, [-fwd[0], -fwd[1], 0]);
    if (held.has('KeyD')) move = acc(move, right);
    if (held.has('KeyA')) move = acc(move, [-right[0], -right[1], 0]);
    if (held.has('ArrowUp')) face = acc(face, fwd);
    if (held.has('ArrowDown')) face = acc(face, [-fwd[0], -fwd[1], 0]);
    if (held.has('ArrowRight')) face = acc(face, right);
    if (held.has('ArrowLeft')) face = acc(face, [-right[0], -right[1], 0]);
    // Full stick = MAX_SPEED, except while CARRYing the box (the carry clips are slow, so we
    // cap the command at the data's pace); Shift scales either to a walk -- matches viewer.py.
    const top = (mm.state === mm.CARRY ? mm.CARRY_MAX_SPEED : mm.MAX_SPEED) * (shift ? mm.WALK_SCALE : 1);
    const mN = Math.hypot(move[0], move[1]);
    if (mN > 1e-6) { const s = top / mN; move = [move[0] * s, move[1] * s, 0]; }
    else move = [0, 0, 0];
    const fN = Math.hypot(face[0], face[1]);
    face = fN > 1e-6 ? [face[0] / fN, face[1] / fN, 0] : [0, 0, 0];
    return { move, face, speed: mN > 1e-6 ? top : 0 };
  }

  // ---- place the body mesh-groups from FK (wxyz quat -> three xyzw) ----
  const vP = new THREE.Vector3(), vC = new THREE.Vector3(), vMid = new THREE.Vector3(), vDir = new THREE.Vector3();
  function place(qpos) {
    const { wp, wq } = fk(bodies, qpos);
    for (let i = 0; i < bodies.length; i++) {
      bodyGroups[i].position.set(wp[i][0], wp[i][1], wp[i][2]);
      bodyGroups[i].quaternion.set(wq[i][1], wq[i][2], wq[i][3], wq[i][0]);
    }
  }
  // place the interactive box from a box qpos (pos[0:3] + quat wxyz[3:7])
  function placeBox(bq) {
    boxGroup.position.set(bq[0], bq[1], bq[2]);
    boxGroup.quaternion.set(bq[4], bq[5], bq[6], bq[3]);
  }

  function drawGizmo() {
    for (let k = 0; k < 3; k++) {
      const p = mm.Tpos[k], dir = mm.Tdir[k];
      gizSph[k].position.set(p[0], p[1], 0.05);
      const tip = [p[0] + 0.3 * dir[0], p[1] + 0.3 * dir[1], 0.05];
      vP.set(p[0], p[1], 0.05); vC.set(tip[0], tip[1], 0.05);
      vMid.addVectors(vP, vC).multiplyScalar(0.5); gizStk[k].position.copy(vMid);
      vDir.subVectors(vC, vP).normalize(); gizStk[k].quaternion.setFromUnitVectors(_yAxis, vDir);
      gizStk[k].scale.set(1, vP.distanceTo(vC), 1);
    }
  }

  // ---- fixed-timestep loop with render interpolation ----
  // The matcher steps at a fixed 30 Hz but the display refreshes at 60-144 Hz; rendering
  // the latest 30 Hz pose every frame makes the 30 fps motion judder against the smooth
  // follow-camera (reads as motion blur). So we interpolate the rendered pose between the
  // two most recent 30 Hz steps by the leftover accumulator fraction -> smooth at any fps.
  const DT = mm.DT;
  const _q0 = new THREE.Quaternion(), _q1 = new THREE.Quaternion(), _qi = new THREE.Quaternion();
  const _rq = new Float64Array(36);
  function interp(a, b, t) {                       // a,b: qpos(36); pos lerp, quat slerp, joints lerp
    for (let i = 0; i < 3; i++) _rq[i] = a[i] + (b[i] - a[i]) * t;
    _q0.set(a[4], a[5], a[6], a[3]); _q1.set(b[4], b[5], b[6], b[3]);   // wxyz -> three xyzw
    _qi.slerpQuaternions(_q0, _q1, t);
    _rq[3] = _qi.w; _rq[4] = _qi.x; _rq[5] = _qi.y; _rq[6] = _qi.z;
    for (let i = 7; i < 36; i++) _rq[i] = a[i] + (b[i] - a[i]) * t;
    return _rq;
  }
  const _bq = new Float64Array(7);
  function interpBox(a, b, t) {                    // box qpos(7): pos lerp, quat slerp
    for (let i = 0; i < 3; i++) _bq[i] = a[i] + (b[i] - a[i]) * t;
    _q0.set(a[4], a[5], a[6], a[3]); _q1.set(b[4], b[5], b[6], b[3]);
    _qi.slerpQuaternions(_q0, _q1, t);
    _bq[3] = _qi.w; _bq[4] = _qi.x; _bq[5] = _qi.y; _bq[6] = _qi.z;
    return _bq;
  }

  let acc = 0, last = performance.now() / 1000, lastSpeed = 0;
  let curQ = mm.step([0, 0, 0], [0, 0, 0]), prevQ = curQ;
  let curB = mm.boxQpos(), prevB = curB;
  let fps = 0, fpsN = 0, fpsT = last;
  function frame() {
    const now = performance.now() / 1000;
    acc += Math.min(now - last, 0.1); last = now;
    while (acc >= DT) {
      const c = command(); lastSpeed = c.speed;
      prevQ = curQ; curQ = mm.step(c.move, c.face);
      prevB = curB; curB = mm.boxQpos();
      acc -= DT;
    }
    const f = acc / DT;
    const rq = interp(prevQ, curQ, f);              // render one step behind, smoothly
    place(rq);
    placeBox(interpBox(prevB, curB, f));
    drawGizmo();

    // follow camera (keep the pelvis centred; user can still orbit/zoom)
    controls.target.lerp(new THREE.Vector3(rq[0], rq[1], 0.8), 0.2);
    controls.update();

    fpsN++;
    if (now - fpsT >= 0.5) { fps = fpsN / (now - fpsT); fpsN = 0; fpsT = now; }

    // Box state machine takes precedence in the HUD; otherwise show the loco gait.
    const state = mm.stateName();
    let head;
    if (state === 'LOCOMOTION' && !mm.jumping) {
      head = lastSpeed > mm.MAX_SPEED * (1 + mm.WALK_SCALE) / 2 ? 'RUN' : (lastSpeed > 1e-3 ? 'WALK' : 'IDLE');
      head += mm.nearBox ? '  [B: pick up]' : '  (walk to the box, then B)';
    } else {
      head = mm.jumping ? 'JUMP' : state;
      if (state === 'CARRY') head += '  [B: set down]';
    }
    const cid = mm._clipOf(mm.cur);
    const fic = mm.cur - mm.starts[cid];
    setHud(`${head}  ${lastSpeed.toFixed(1)} m/s\nclip [${cid}]: ${mm.clipNames[cid]}\nframe ${fic} (global ${mm.cur})\n` +
      `box: ${mm.boxHeldNow ? 'held' : 'resting'}\n` +
      `\nrender ${fps.toFixed(0)} fps · sim ${(1 / DT).toFixed(0)} Hz (${(DT * 1000).toFixed(1)} ms)\n` +
      `search every ${(mm.SEARCH_TIME * 1000).toFixed(0)} ms\n` +
      `\nWASD move · arrows face · Shift walk\nB box · J jump · Space reset · T gizmo · drag/scroll camera`);

    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  setHud('');
  frame();
}

boot().catch((e) => setHud('error: ' + e.message));

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { BOX } from './sim.js';

export { BOX };

const toThreeQuat = (w) => new THREE.Quaternion(w[1], w[2], w[3], w[0]);

function groundTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#d9dce0';
  g.fillRect(0, 0, 512, 512);
  g.fillStyle = '#ccd0d5';
  g.fillRect(0, 0, 256, 256);
  g.fillRect(256, 256, 256, 256);
  g.strokeStyle = '#b8bdc4';
  g.lineWidth = 3;
  g.strokeRect(0, 0, 512, 512);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(30, 30);
  t.anisotropy = 8;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function boxTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = '#c99a62';
  g.fillRect(0, 0, 256, 256);
  for (let i = 0; i < 40; i++) {
    g.strokeStyle = `rgba(120, 80, 40, ${0.05 + Math.random() * 0.08})`;
    g.lineWidth = 1 + Math.random() * 2;
    const y = Math.random() * 256;
    g.beginPath(); g.moveTo(0, y); g.bezierCurveTo(80, y + 6, 170, y - 6, 256, y + 2); g.stroke();
  }
  g.strokeStyle = 'rgba(90, 60, 30, 0.6)';
  g.lineWidth = 8;
  g.strokeRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export class Scene {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#eef1f5');
    this.scene.fog = new THREE.Fog('#eef1f5', 12, 30);

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 100);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-2.4, -2.9, 1.75);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0, 0.7);
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 14;
    this.follow = true;

    this.scene.add(new THREE.HemisphereLight('#ffffff', '#9aa3ad', 1.1));
    const sun = new THREE.DirectionalLight('#ffffff', 2.2);
    sun.position.set(-3, -4, 8);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = -6; sc.right = 6; sc.top = 6; sc.bottom = -6; sc.near = 0.5; sc.far = 25;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    this.sun = sun;
    this.scene.add(sun, sun.target);

    const ground = new THREE.Mesh(new THREE.PlaneGeometry(120, 120),
      new THREE.MeshStandardMaterial({ map: groundTexture(), roughness: 0.95 }));
    ground.receiveShadow = true;
    this.scene.add(ground);
    this.ground = ground;

    this.boxMeshes = new Map();
    this.boxGeom = new THREE.BoxGeometry(...BOX.size);
    this.boxMat = new THREE.MeshStandardMaterial({ map: boxTexture(), roughness: 0.8 });
    this.planMat = new THREE.MeshBasicMaterial({ color: '#c99a62', transparent: true, opacity: 0.12, depthWrite: false });
    this.edgeGeom = new THREE.EdgesGeometry(this.boxGeom);
    this.physBoxMeshes = new Map();
    this.ghostMat = new THREE.MeshStandardMaterial({ color: '#4dabf7', transparent: true, opacity: 0.3, depthWrite: false, roughness: 0.6 });
    this.ghostBox = new THREE.Mesh(this.boxGeom, this.ghostMat);
    this.ghostBox.renderOrder = 2;
    this.ghostBox.visible = false;
    this.scene.add(this.ghostBox);
    this.placeMarker = new THREE.Group();
    const ghost = new THREE.Mesh(this.boxGeom, new THREE.MeshBasicMaterial({ color: '#1971c2', transparent: true, opacity: 0.18, depthWrite: false }));
    const ghostEdges = new THREE.LineSegments(this.edgeGeom, new THREE.LineBasicMaterial({ color: '#1971c2' }));
    const arrow = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.03, 0.005), new THREE.MeshBasicMaterial({ color: '#1971c2' }));
    arrow.position.set(BOX.size[0] / 2 + 0.11, 0, -BOX.size[2] / 2 + 0.004);
    this.placeArrow = arrow;
    ghost.userData.placeMarker = true;
    this.placeMarker.add(ghost, ghostEdges, arrow);
    this.placeMarker.visible = false;
    this.scene.add(this.placeMarker);
    this.placeGhost = ghost;

    this.gizmos = new THREE.Group();
    this.scene.add(this.gizmos);
    this.makeGizmos();
    this.showGizmos = true;
    this.makeFootprints();
    this.showFootprints = true;

    this.raycaster = new THREE.Raycaster();
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  async loadRobot(base) {
    const meta = await (await fetch(`${base}/robot.json`)).json();
    this.meta = meta;
    const blob = await (await fetch(`${base}/robot.bin`)).arrayBuffer();
    const geoms = meta.meshes.map((m) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(blob, m.vert_offset, m.nvert * 3), 3));
      g.setIndex(new THREE.BufferAttribute(new Uint32Array(blob, m.index_offset, m.nindex), 1));
      g.computeVertexNormals();
      return g;
    });
    const mats = new Map();
    const material = (rgba) => {
      const k = rgba.join(',');
      if (!mats.has(k)) {
        const dark = rgba[0] < 0.3;
        mats.set(k, new THREE.MeshStandardMaterial({
          color: new THREE.Color(rgba[0], rgba[1], rgba[2]).convertSRGBToLinear(),
          roughness: dark ? 0.55 : 0.35, metalness: dark ? 0.2 : 0.05,
        }));
      }
      return mats.get(k);
    };
    this.robotParts = { meta, geoms, material };
    const main = this.addRobot(null);
    this.robot = main.root;
    this.bodies = main.bodies;
    this.pelvis = this.bodies[0];
    // The reference a tracker follows in the physical mode.
    this.ghost = this.addRobot(this.ghostMat);
    this.ghost.root.visible = false;
  }

  // Another robot with its own bodies, drawn with the robot's materials, or with mat over all of it.
  addRobot(mat = null) {
    const { meta, geoms, material } = this.robotParts;
    const bodies = meta.bodies.map((b) => {
      const body = new THREE.Group();
      body.position.set(...b.pos);
      body.quaternion.copy(toThreeQuat(b.quat));
      const joint = new THREE.Group();
      body.add(joint);
      return { meta: b, body, joint, axis: b.axis ? new THREE.Vector3(...b.axis) : null };
    });
    const root = new THREE.Group();
    bodies.forEach((b) => (b.meta.parent < 0 ? root : bodies[b.meta.parent].joint).add(b.body));
    for (const g of meta.geoms) {
      const mesh = new THREE.Mesh(geoms[g.mesh], mat || material(g.rgba));
      mesh.position.set(...g.pos);
      mesh.quaternion.copy(toThreeQuat(g.quat));
      mesh.castShadow = !mat;
      mesh.receiveShadow = !mat;
      if (mat) mesh.renderOrder = 2;
      bodies[g.body].joint.add(mesh);
    }
    this.scene.add(root);
    return { root, bodies };
  }

  static pose(bodies, state) {
    bodies[0].body.position.set(...state.hipPos);
    bodies[0].body.quaternion.copy(toThreeQuat(state.hipRot));
    for (const b of bodies) {
      if (b.axis) b.joint.quaternion.setFromAxisAngle(b.axis, state.joints[b.meta.dof]);
    }
  }

  setPose(state) { Scene.pose(this.bodies, state); }

  // state: a pose as setPose takes it, or null to hide the reference. box: the pose of the box in the
  // reference's hands, [x, y, z, qw, qx, qy, qz], or null.
  setGhost(state, box = null) {
    this.ghost.root.visible = !!state;
    if (state) Scene.pose(this.ghost.bodies, state);
    this.ghostBox.visible = !!state && !!box;
    if (this.ghostBox.visible) {
      this.ghostBox.position.set(box[0], box[1], box[2]);
      this.ghostBox.quaternion.copy(toThreeQuat(box.slice(3)));
    }
  }

  // Draws every box at BOX.size, after it changed.
  setBoxSize() {
    const old = [this.boxGeom, this.edgeGeom];
    this.boxGeom = new THREE.BoxGeometry(...BOX.size);
    this.edgeGeom = new THREE.EdgesGeometry(this.boxGeom);
    const swap = (m) => {
      if (m.geometry === old[0]) m.geometry = this.boxGeom;
      else if (m.geometry === old[1]) m.geometry = this.edgeGeom;
    };
    this.scene.traverse(swap);
    for (const m of this.boxMeshes.values()) m.label.position.set(0, 0, BOX.size[2] / 2 + 0.25);
    this.placeArrow.position.set(BOX.size[0] / 2 + 0.11, 0, -BOX.size[2] / 2 + 0.004);
    old.forEach((g) => g.dispose());
  }

  // One entry per physics box slot: [x, y, z, qw, qx, qy, qz] or null.
  syncPhysBoxes(slots) {
    slots.forEach((p, i) => {
      let m = this.physBoxMeshes.get(i);
      if (!p) { if (m) m.visible = false; return; }
      if (!m) {
        m = new THREE.Mesh(this.boxGeom, this.boxMat);
        m.castShadow = true;
        m.receiveShadow = true;
        m.add(new THREE.LineSegments(this.edgeGeom, new THREE.LineBasicMaterial({ color: '#7a5530' })));
        this.scene.add(m);
        this.physBoxMeshes.set(i, m);
      }
      m.visible = true;
      m.position.set(p[0], p[1], p[2]);
      m.quaternion.copy(toThreeQuat(p.slice(3)));
    });
  }

  hidePhysBoxes() { for (const m of this.physBoxMeshes.values()) m.visible = false; }

  // held: { id, pos, rot } world pose of the box in the robot's hands, or null.
  // plan: the boxes are where the commands put them, drawn faint, while syncPhysBoxes draws where the physics
  // has them. The held box is then left to the reference (setGhost).
  syncBoxes(boxes, selectedId, targetId, held, placeTarget, plan = false) {
    const alive = new Set();
    for (const box of boxes) {
      alive.add(box.id);
      let m = this.boxMeshes.get(box.id);
      if (!m) {
        const mesh = new THREE.Mesh(this.boxGeom, this.boxMat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData.boxId = box.id;
        const edges = new THREE.LineSegments(this.edgeGeom, new THREE.LineBasicMaterial({ color: '#7a5530' }));
        mesh.add(edges);
        const label = this.makeLabel(box.label);
        label.position.set(0, 0, BOX.size[2] / 2 + 0.25);
        mesh.add(label);
        this.scene.add(mesh);
        m = { mesh, edges, label };
        this.boxMeshes.set(box.id, m);
      }
      const isHeld = !!held && held.id === box.id;
      m.mesh.material = plan ? this.planMat : this.boxMat;
      m.mesh.castShadow = m.mesh.receiveShadow = !plan;
      m.mesh.visible = !(plan && isHeld);
      if (isHeld) {
        m.mesh.position.set(...held.pos);
        m.mesh.quaternion.copy(toThreeQuat(held.rot));
      } else {
        m.mesh.position.set(box.pos[0], box.pos[1], BOX.size[2] / 2);
        m.mesh.rotation.set(0, 0, box.yaw);
      }
      m.label.visible = !isHeld;
      const isTarget = box.id === targetId, isSel = box.id === selectedId;
      m.edges.material.color.set(isTarget ? '#e8590c' : isSel ? '#1971c2' : plan ? '#a07a4f' : '#7a5530');
      m.edges.material.linewidth = 2;
      m.label.material.opacity = isTarget || isSel ? 1 : 0.75;
    }
    for (const [id, m] of this.boxMeshes) {
      if (!alive.has(id)) {
        this.scene.remove(m.mesh);
        this.boxMeshes.delete(id);
      }
    }
    this.placeMarker.visible = !!placeTarget;
    if (placeTarget) {
      this.placeMarker.position.set(placeTarget.pos[0], placeTarget.pos[1], BOX.size[2] / 2);
      this.placeMarker.rotation.set(0, 0, placeTarget.yaw);
    }
  }

  makeLabel(text) {
    const c = document.createElement('canvas');
    c.width = 128; c.height = 128;
    const g = c.getContext('2d');
    g.fillStyle = 'rgba(255,255,255,0.92)';
    g.beginPath(); g.arc(64, 64, 52, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#343a40';
    g.font = 'bold 64px system-ui, sans-serif';
    g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText(text, 64, 68);
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthTest: false }));
    s.scale.set(0.22, 0.22, 1);
    s.renderOrder = 10;
    return s;
  }

  makeGizmos() {
    const sphere = new THREE.SphereGeometry(0.035, 12, 8);
    const red = new THREE.MeshBasicMaterial({ color: '#e03131' });
    this.trajPts = [0, 1, 2, 3].map(() => {
      const g = new THREE.Group();
      g.add(new THREE.Mesh(sphere, red));
      const stick = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.012, 0.012), red);
      stick.position.x = 0.09;
      g.add(stick);
      this.gizmos.add(g);
      return g;
    });
    const ring = (color) => {
      const g = new THREE.Group();
      const r = new THREE.Mesh(new THREE.RingGeometry(0.07, 0.095, 32), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthTest: false }));
      const arrow = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.02, 0.005), new THREE.MeshBasicMaterial({ color, depthTest: false }));
      arrow.position.x = 0.15;
      r.renderOrder = arrow.renderOrder = 11;
      g.add(r, arrow);
      this.gizmos.add(g);
      return g;
    };
    this.targetMarker = ring('#e8590c');
    this.animMarker = ring('#2f9e44');
    this.riseMarker = ring('#7048e8');
  }

  updateGizmos(ctrl) {
    const show = this.showGizmos && !!ctrl;
    if (!ctrl) {
      for (const g of [...this.trajPts, this.targetMarker, this.animMarker, this.riseMarker]) g.visible = false;
      return;
    }
    const loco = ctrl.mode === 'loco';
    this.trajPts.forEach((g, i) => {
      g.visible = show && loco;
      const p = ctrl.trajectory.pos[i], r = ctrl.trajectory.rot[i];
      g.position.set(p[0], p[1], 0.02);
      g.quaternion.copy(toThreeQuat(r));
    });
    const place = (m, pos, yaw, z, visible) => {
      m.visible = show && visible;
      if (!m.visible) return;
      m.position.set(pos[0], pos[1], z);
      m.rotation.set(0, 0, yaw);
    };
    const phase = ctrl.boxPhase();
    const pre = phase === 'pick-pre' || phase === 'place-pre';
    const t = ctrl.target;
    place(this.targetMarker, t ? t.pos : [0, 0], t ? t.yaw : 0, 0.012, !!t && (pre || ctrl.mode === 'loco'));
    const a = pre ? ctrl.animTarget() : null;
    place(this.animMarker, a ? a.pos : [0, 0], a ? Math.atan2(a.dir[1], a.dir[0]) : 0, 0.018, !!a);
    const c = !pre && phase ? ctrl.commit : null;
    place(this.riseMarker, c ? c.pos : [0, 0], c ? c.yaw : 0, 0.015, !!c);
  }

  // A G1 sole: its contact spheres are 0.05 m behind to 0.12 m in front of the ankle roll link, 0.03 m to each side.
  makeFootprints() {
    // 3 cm wider than the sole on each side, so the print shows around a foot that stands on it.
    const shape = new THREE.Shape(), x0 = -0.095, x1 = 0.165, w = 0.07, r = 0.05;
    shape.moveTo(x0 + r, -w);
    shape.lineTo(x1 - r, -w);
    shape.quadraticCurveTo(x1, -w, x1, 0);
    shape.quadraticCurveTo(x1, w, x1 - r, w);
    shape.lineTo(x0 + r, w);
    shape.quadraticCurveTo(x0, w, x0, 0);
    shape.quadraticCurveTo(x0, -w, x0 + r, -w);
    const geom = new THREE.ShapeGeometry(shape, 6);
    this.footMats = { pick: '#eb6834', place: '#6250d6' };
    for (const k in this.footMats) this.footMats[k] = new THREE.MeshBasicMaterial({ color: this.footMats[k], transparent: true, opacity: 0.55, depthWrite: false });
    this.footprints = [0, 1].map(() => {
      const m = new THREE.Mesh(geom, this.footMats.pick);
      m.renderOrder = 1;
      m.visible = false;
      this.scene.add(m);
      return m;
    });
  }

  // feet: [{ pos, yaw }] per foot (FootLock.feetOf), or null. kind: 'pick' or 'place'.
  setFootprints(feet, kind) {
    this.footprints.forEach((m, s) => {
      m.visible = this.showFootprints && !!feet;
      if (!m.visible) return;
      m.material = this.footMats[kind] || this.footMats.pick;
      m.position.set(feet[s].pos[0], feet[s].pos[1], 0.003);
      m.rotation.set(0, 0, feet[s].yaw);
    });
  }

  followCharacter(hipPos, dt) {
    if (!this.follow) return;
    const t = this.controls.target;
    const goal = new THREE.Vector3(hipPos[0], hipPos[1], 0.65);
    const k = 1 - Math.exp(-dt * 4);
    const delta = goal.clone().sub(t).multiplyScalar(k);
    t.add(delta);
    this.camera.position.add(delta);
    this.sun.position.set(t.x - 3, t.y - 4, 8);
    this.sun.target.position.set(t.x, t.y, 0);
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  // Pointer helpers: the box under a screen point, and the ground point under it.
  pick(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const targets = [...this.boxMeshes.values()].map((m) => m.mesh);
    if (this.placeMarker.visible) targets.push(this.placeGhost);
    const hits = this.raycaster.intersectObjects(targets, false);
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    const g = new THREE.Vector3();
    const onGround = this.raycaster.ray.intersectPlane(plane, g);
    const top = hits.length ? hits[0].object.userData : {};
    return { boxId: top.boxId ?? null, marker: !!top.placeMarker, ground: onGround ? [g.x, g.y] : null };
  }

  cameraYaw() {
    const d = new THREE.Vector3();
    this.camera.getWorldDirection(d);
    return Math.atan2(d.y, d.x);
  }
}

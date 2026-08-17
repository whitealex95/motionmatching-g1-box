"""Port of the SceneBot web demo's motion-graph reference player.

Plays clip segments from motion_graph.json, stitching them with planar
(yaw + translation) alignment so the reference root path is continuous, and
emits one 50 Hz stream packet per tick: lower-body command, VR 3-point
targets, contact label, and the world anchor pose the policy tracks.

PickUpBox is a non-recurring segment (clip 11, frames 0..120) played forward
at half speed; PutDownBox is the same segment played backward. After the
pickup completes the demo freezes the upper-body targets (UpperBodyFreeze)
so the robot can walk while carrying the box.
"""
import numpy as np

from .rotations import (wxyz_to_xyzw, xyzw_to_wxyz, quat_mul_xyzw,
                        quat_conj_xyzw, quat_rotate_xyzw, yaw_quat_xyzw,
                        yaw_from_wxyz)

FORWARD = 'Forward'
BACKWARD = 'Backward'
TURN_LEFT = 'ForwardTurnLeft'
TURN_RIGHT = 'ForwardTurnRight'
CLIMB_STAIR = 'ClimbStair'
STEP_ON_BOX = 'StepOnBox'
COME_DOWN_BOX = 'ComeDownBox'
PICK_UP_BOX = 'PickUpBox'
KICK = 'Kick'
SIT_DOWN = 'SitDown'
STOP = 'Stop'
PUT_DOWN_BOX = 'PutDownBox'
STAND_UP = 'StandUp'

PICKUP_EDGE = f'{STOP}->{PICK_UP_BOX}'
SIT_EDGE = f'{STOP}->{SIT_DOWN}'
_NON_RECURRING_SRC = {PICK_UP_BOX, SIT_DOWN}
CONTACT_KEEP_FEET = {CLIMB_STAIR, STEP_ON_BOX, COME_DOWN_BOX, PICK_UP_BOX,
                     PUT_DOWN_BOX, SIT_DOWN, STAND_UP}

# Isaac joint indices of the 12 leg joints (left leg pitch..ankle, right leg)
LOWER_IDX = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18], int)
# Isaac BFS body indices of the VR 3-point bodies: L wrist, R wrist, torso
VR_BODIES = [28, 29, 9]
VR_OFFSETS = np.array([[0.18, -0.025, 0.0],
                       [0.18, 0.025, 0.0],
                       [0.0, 0.0, 0.35]])
# joint indices the demo freezes for the reference-ghost pose while carrying
FROZEN_JOINTS = [12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]


class GraphState:
    __slots__ = ('edge_key', 'clip_idx', 'frame_idx', 'segment_end_frame',
                 'rot', 'trans', 'command', 'playback_direction',
                 'frame_step_accum')

    def __init__(self, edge_key='', clip_idx=0, frame_idx=0,
                 segment_end_frame=0, rot=None, trans=None, command=STOP,
                 playback_direction=1, frame_step_accum=0.0):
        self.edge_key = edge_key
        self.clip_idx = int(clip_idx)
        self.frame_idx = int(frame_idx)
        self.segment_end_frame = int(segment_end_frame)
        self.rot = np.array([0.0, 0.0, 0.0, 1.0]) if rot is None else np.asarray(rot, float)
        self.trans = np.zeros(3) if trans is None else np.asarray(trans, float)
        self.command = command
        self.playback_direction = int(playback_direction)
        self.frame_step_accum = float(frame_step_accum)

    def snapshot(self):
        return GraphState(self.edge_key, self.clip_idx, self.frame_idx,
                          self.segment_end_frame, self.rot.copy(),
                          self.trans.copy(), self.command,
                          self.playback_direction, self.frame_step_accum)


def _clamp(v, lo, hi):
    v = int(np.floor(v)) if np.isfinite(v) else lo
    return lo if v < lo else (hi if v > hi else v)


def segment_finished(st):
    if st.playback_direction >= 0:
        return st.frame_idx >= st.segment_end_frame
    return st.frame_idx <= st.segment_end_frame


def _advance(st, pickup_scale=1.0):
    step = float(pickup_scale) if (st.command == PICK_UP_BOX
                                   and st.playback_direction >= 0) else 1.0
    st.frame_step_accum += step
    while st.frame_step_accum >= 1.0:
        st.frame_idx += 1 if st.playback_direction >= 0 else -1
        st.frame_step_accum -= 1.0


def _finished_reverse(st, key):
    return (st.edge_key == key and st.playback_direction < 0
            and st.frame_idx <= st.segment_end_frame)


def _at_reverse_start(st, edge_map, key):
    e = edge_map.get(key)
    if not e:
        return False
    start = max(0, int(e['start_frame']))
    return (st.command == STOP and st.clip_idx == int(e['clip_idx'])
            and st.frame_idx == start and st.segment_end_frame == start
            and st.playback_direction >= 0)


def _clamp_to_clip(st, n_frames):
    hi = max(0, n_frames - 1)
    st.frame_idx = _clamp(st.frame_idx, 0, hi)
    if st.playback_direction >= 0:
        st.segment_end_frame = _clamp(st.segment_end_frame, st.frame_idx, hi)
    else:
        st.segment_end_frame = _clamp(st.segment_end_frame, 0, st.frame_idx)


def _rotate_about(st, pivot, yaw):
    if abs(yaw) == 0.0:
        return
    g = yaw_quat_xyzw(yaw)
    st.rot = quat_mul_xyzw(g, st.rot)
    rotated_trans = quat_rotate_xyzw(g, st.trans)
    rotated_pivot = quat_rotate_xyzw(g, pivot)
    st.trans = rotated_trans + (np.asarray(pivot, float) - rotated_pivot)


def plan_alignment(pos, quat_wxyz, target_pos, target_quat_wxyz):
    dyaw = yaw_from_wxyz(target_quat_wxyz) - yaw_from_wxyz(quat_wxyz)
    r_delta = yaw_quat_xyzw(dyaw)
    t = np.asarray(target_pos, float) - quat_rotate_xyzw(r_delta, pos)
    return r_delta, t


def apply_alignment(rot, trans, clip_pos, clip_quat_wxyz):
    pos = quat_rotate_xyzw(rot, clip_pos) + trans
    quat = quat_mul_xyzw(rot, wxyz_to_xyzw(clip_quat_wxyz))
    return pos, xyzw_to_wxyz(quat)


def aligned_root(clip, frame, rot, trans):
    f = clip.frame(frame)
    pos, quat = apply_alignment(rot, trans, clip.body_pos[f, 0],
                                clip.body_quat[f, 0])
    return {'joint_pos_isaac': clip.joint_pos[f],
            'root_pos_w': pos, 'root_quat_wxyz': quat}


def _dst_command(seg, direction):
    if direction < 0 and seg['dst_command'] == PICK_UP_BOX:
        return PUT_DOWN_BOX
    if direction < 0 and seg['dst_command'] == SIT_DOWN:
        return STAND_UP
    return str(seg['dst_command'])


def _direction_for(cmd, edge_key):
    if (cmd == PUT_DOWN_BOX and edge_key == PICKUP_EDGE) or \
       (cmd == STAND_UP and edge_key == SIT_EDGE):
        return -1
    return 1


def _state_from_segment(seg, rot, trans, direction=1):
    start = max(0, int(seg['start_frame']))
    end = max(start, int(seg['end_frame']))
    frm, to = (start, end) if direction >= 0 else (end, start)
    return GraphState(edge_key=str(seg['edge_key']),
                      clip_idx=int(seg['clip_idx']), frame_idx=frm,
                      segment_end_frame=to, rot=rot, trans=trans,
                      command=_dst_command(seg, direction),
                      playback_direction=direction)


def _enter_segment(bundle, target_pos, target_quat, seg, direction=1):
    st = _state_from_segment(seg, np.array([0.0, 0.0, 0.0, 1.0]),
                             np.zeros(3), direction)
    clip = bundle.clip(st.clip_idx)
    f = _clamp(st.frame_idx, 0, clip.n_frames - 1)
    r_delta, t = plan_alignment(clip.body_pos[f, 0], clip.body_quat[f, 0],
                                target_pos, target_quat)
    st.frame_idx = f
    st.rot, st.trans = r_delta, t
    return st


def _jump_to_stop_start(edge_map, bundle, target_pos, target_quat, key):
    e = edge_map.get(key)
    if not e:
        return None
    start = max(0, int(e['start_frame']))
    pseudo = {'edge_key': f'{STOP}->{STOP}', 'dst_command': STOP,
              'clip_idx': int(e['clip_idx']), 'start_frame': start,
              'end_frame': start}
    st = _enter_segment(bundle, target_pos, target_quat, pseudo, 1)
    st.edge_key = pseudo['edge_key']
    st.command = STOP
    st.playback_direction = 1
    st.frame_idx = start
    st.segment_end_frame = start
    return st


def _seg_sort_key(seg):
    return (float(seg.get('score', 0.0)) or 0.0, int(seg['clip_idx']),
            int(seg['start_frame']), str(seg['edge_key']))


def _best(segs):
    return min(segs, key=_seg_sort_key)


def _initial_edge_key(edge_map, want):
    segs = list(edge_map.values())
    if not segs:
        raise ValueError('edge_map is empty')
    cands = [s for s in segs if s['dst_command'] == want]
    if not cands:
        cands = [s for s in segs if s['dst_command'] == STOP]
    filtered = [s for s in cands
                if str(s['src_command']) not in _NON_RECURRING_SRC]
    if filtered:
        cands = filtered
    from_stop = [s for s in cands if s['src_command'] == STOP]
    if from_stop:
        cands = from_stop
    if not cands:
        cands = segs
    return str(_best(cands)['edge_key'])


def _next_edge_key(edge_map, cur_edge_key, want):
    if want == PUT_DOWN_BOX and PICKUP_EDGE in edge_map:
        return PICKUP_EDGE
    if want == STAND_UP and SIT_EDGE in edge_map:
        return SIT_EDGE
    e = edge_map.get(cur_edge_key)
    if e is None:
        if '->' not in cur_edge_key:
            return None
        cur_dst = cur_edge_key.split('->', 1)[1]
    else:
        cur_dst = str(e['dst_command'])
    outgoing = [s for s in edge_map.values() if s['src_command'] == cur_dst]
    if not outgoing:
        return None
    cands = [s for s in outgoing if s['dst_command'] == want]
    if not cands and want != STOP:
        cands = [s for s in outgoing if s['dst_command'] == cur_dst]
    if not cands:
        cands = [s for s in outgoing if s['dst_command'] == STOP]
    if not cands:
        cands = outgoing
    return str(_best(cands)['edge_key'])


def _transition(st, edge_map, bundle, cmd, root_pos, root_quat,
                pickup_scale=1.0):
    if not segment_finished(st):
        _advance(st, pickup_scale)
        return st
    if _finished_reverse(st, PICKUP_EDGE):
        return _jump_to_stop_start(edge_map, bundle, root_pos, root_quat,
                                   PICKUP_EDGE) or st
    if _finished_reverse(st, SIT_EDGE):
        return _jump_to_stop_start(edge_map, bundle, root_pos, root_quat,
                                   SIT_EDGE) or st
    if (cmd == PUT_DOWN_BOX and _at_reverse_start(st, edge_map, PICKUP_EDGE)) \
            or (cmd == STAND_UP and _at_reverse_start(st, edge_map, SIT_EDGE)) \
            or (cmd == STOP and st.command in (STOP, SIT_DOWN)):
        return st
    key = _next_edge_key(edge_map, st.edge_key, cmd)
    if key is None:
        return st
    seg = edge_map[key]
    direction = _direction_for(cmd, key)
    nxt = _enter_segment(bundle, root_pos, root_quat, seg, direction)
    nxt.edge_key = key
    nxt.command = _dst_command(seg, direction)
    return nxt


def _skip_to_stop(st, edge_map, bundle, from_cmd, root_pos, root_quat):
    key = f'{from_cmd}->{STOP}'
    if key not in edge_map:
        key = _next_edge_key(edge_map, st.edge_key, STOP)
        if key is None:
            return st
    e = edge_map[key]
    end = max(int(e['start_frame']), int(e['end_frame']))
    clip = bundle.clip(int(e['clip_idx']))
    f = _clamp(end, 0, clip.n_frames - 1)
    r_delta, t = plan_alignment(clip.body_pos[f, 0], clip.body_quat[f, 0],
                                root_pos, root_quat)
    return GraphState(edge_key=key, clip_idx=int(e['clip_idx']), frame_idx=f,
                      segment_end_frame=f, rot=r_delta, trans=t, command=STOP,
                      playback_direction=1)


def build_packet(clip, frame, contact_mask, anchor_pos, anchor_quat_wxyz,
                 direction=1):
    f = clip.frame(frame)
    jp = clip.joint_pos[f]              # (29,) isaac order
    jv = clip.joint_vel[f]
    if direction < 0:
        jv = -jv
    lower_cmd = np.concatenate([jp[LOWER_IDX], jv[LOWER_IDX]]).astype(np.float32)

    body_pos = clip.body_pos[f]         # (30, 3) world
    body_quat = clip.body_quat[f]       # (30, 4) wxyz
    root_p = body_pos[0].astype(float)
    root_inv = quat_conj_xyzw(wxyz_to_xyzw(body_quat[0]))
    vr_pos = np.zeros(9, np.float32)
    vr_orn = np.zeros(12, np.float32)
    for i, k in enumerate(VR_BODIES):
        body_q = wxyz_to_xyzw(body_quat[k])
        off_w = quat_rotate_xyzw(body_q, VR_OFFSETS[i])
        rel = body_pos[k] + off_w - root_p
        vr_pos[i * 3:i * 3 + 3] = quat_rotate_xyzw(root_inv, rel)
        q_local = quat_mul_xyzw(root_inv, body_q)
        vr_orn[i * 4:i * 4 + 4] = xyzw_to_wxyz(q_local)

    if anchor_pos is None:
        anchor_pos = root_p
    anchor_q = wxyz_to_xyzw(anchor_quat_wxyz if anchor_quat_wxyz is not None
                            else body_quat[0])
    return {
        'lower_cmd': lower_cmd,
        'vr_3point_pos_l': vr_pos,
        'vr_3point_orn_l': vr_orn,
        'contact_mask': np.asarray(contact_mask, np.float32),
        'motion_anchor_pos_w': np.asarray(anchor_pos, np.float32),
        'motion_anchor_orn_w': np.asarray(anchor_q, np.float32),  # xyzw
    }


def ref_qpos36(joint_pos_isaac, root_pos_w, root_quat_wxyz, isaac_to_mujoco):
    q = np.zeros(36)
    q[0:3] = root_pos_w
    q[3:7] = root_quat_wxyz
    q[7:36] = np.asarray(joint_pos_isaac, float)[isaac_to_mujoco]
    return q


class MotionGraphPlayer:
    def __init__(self, graph, bundle, contacts, meta,
                 pickup_forward_step_scale=0.5, contact_labels_mn_only=True):
        self.edge_map = graph['edge_segments']
        self.bundle = bundle
        self.contacts = contacts
        self.fps = float(graph.get('fps', 1.0 / meta['control_dt']))
        self.stream_contact_dim = contacts.max_stream_dim()
        self.default_contact = np.zeros(self.stream_contact_dim, np.float32)
        src = np.asarray(meta.get('default_contact_label', []), np.float32)
        n = min(len(src), self.stream_contact_dim)
        self.default_contact[:n] = src[:n]
        self.contact_labels_mn_only = contact_labels_mn_only
        self.pickup_scale = float(pickup_forward_step_scale)
        self.reset()

    def reset(self):
        key = _initial_edge_key(self.edge_map, STOP)
        seg = self.edge_map[key]
        st = _state_from_segment(seg, np.array([0.0, 0.0, 0.0, 1.0]),
                                 np.zeros(3), _direction_for(STOP, key))
        st.frame_idx = st.segment_end_frame
        clip = self.bundle.clip(st.clip_idx)
        f = _clamp(st.frame_idx, 0, clip.n_frames - 1)
        root_p = clip.body_pos[f, 0]
        st.rot, st.trans = plan_alignment(
            root_p, clip.body_quat[f, 0],
            np.array([0.0, 0.0, root_p[2]]), np.array([1.0, 0.0, 0.0, 0.0]))
        self.state = st

    def _pad_contact(self, label):
        out = np.zeros(self.stream_contact_dim, np.float32)
        n = min(len(label), self.stream_contact_dim)
        out[:n] = np.asarray(label, np.float32)[:n]
        return out

    def _maybe_zero_feet(self, label, from_clip):
        if not self.contact_labels_mn_only:
            return label
        if self.state.command in CONTACT_KEEP_FEET:
            return label
        label = label.copy()
        label[0] = 0.0
        label[1] = 0.0
        return label

    def step(self, command, stop_command=None, yaw_adjust=0.0):
        st = self.state
        clip = self.bundle.clip(st.clip_idx)
        _clamp_to_clip(st, clip.n_frames)
        cur = aligned_root(clip, st.frame_idx, st.rot, st.trans)
        if yaw_adjust and abs(yaw_adjust) > 0:
            _rotate_about(st, cur['root_pos_w'], yaw_adjust)
        cur = aligned_root(clip, st.frame_idx, st.rot, st.trans)
        publish = st.snapshot()

        label = self.default_contact
        from_clip = False
        if self.contacts.has_clip(st.clip_idx):
            raw = self.contacts.at_frame(st.clip_idx, st.frame_idx)
            if raw is not None:
                label = self._pad_contact(raw)
                from_clip = True
        label = self._maybe_zero_feet(label, from_clip)

        pkt = build_packet(clip, st.frame_idx, label, cur['root_pos_w'],
                           cur['root_quat_wxyz'], st.playback_direction)
        completed = (publish.command == PICK_UP_BOX
                     and publish.playback_direction >= 0
                     and segment_finished(publish))

        if stop_command:
            self.state = _skip_to_stop(st, self.edge_map, self.bundle,
                                       stop_command, cur['root_pos_w'],
                                       cur['root_quat_wxyz'])
        else:
            self.state = _transition(st, self.edge_map, self.bundle, command,
                                     cur['root_pos_w'],
                                     cur['root_quat_wxyz'],
                                     self.pickup_scale)

        pkt.update(joint_pos_isaac=np.array(cur['joint_pos_isaac'], float),
                   root_pos_w=np.array(cur['root_pos_w'], float),
                   root_quat_wxyz=np.array(cur['root_quat_wxyz'], float),
                   command=publish.command, edge_key=publish.edge_key,
                   clip_idx=publish.clip_idx, frame_idx=publish.frame_idx,
                   pickup_forward_completed=completed)
        return pkt

    def is_idle(self):
        non_loco = {FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, STEP_ON_BOX,
                    COME_DOWN_BOX, PICK_UP_BOX, PUT_DOWN_BOX, KICK, STAND_UP}
        return (segment_finished(self.state)
                and self.state.command not in non_loco)


class UpperBodyFreeze:
    def __init__(self):
        self.active = False
        self._joints = None
        self._wrists = None
        self._vr_pos = None
        self._vr_orn = None

    def snapshot(self, pkt):
        self.active = True
        self._joints = np.array(pkt['joint_pos_isaac'], float)
        cm = pkt['contact_mask']
        self._wrists = np.array([cm[2] if len(cm) > 2 else 0.0,
                                 cm[3] if len(cm) > 3 else 0.0], np.float32)
        self._vr_pos = np.array(pkt['vr_3point_pos_l'], np.float32)
        self._vr_orn = np.array(pkt['vr_3point_orn_l'], np.float32)

    def clear(self):
        self.active = False
        self._joints = self._wrists = self._vr_pos = self._vr_orn = None

    def apply_to_packet(self, pkt):
        if not self.active or self._joints is None:
            return
        pkt['vr_3point_pos_l'] = self._vr_pos.copy()
        pkt['vr_3point_orn_l'] = self._vr_orn.copy()
        cm = np.array(pkt['contact_mask'], np.float32)
        if len(cm) >= 4:
            cm[2] = self._wrists[0]
            cm[3] = self._wrists[1]
        pkt['contact_mask'] = cm

    def blend_ghost_joints(self, joint_pos_isaac):
        q = np.array(joint_pos_isaac, float)
        if self.active and self._joints is not None:
            q[FROZEN_JOINTS] = self._joints[FROZEN_JOINTS]
        return q

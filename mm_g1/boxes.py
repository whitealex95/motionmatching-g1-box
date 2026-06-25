"""Pick / carry / place segmentation + entry indexing for the box skill (B trigger).

Each OmniRetarget robot-object clip is one continuous pick -> carry -> place sequence.
`segment_phases` slices it into the three phases from the box's height trajectory and marks
the contiguous interval over which the box is *attached* to the robot (lifted clear of the
floor / being handled) -- before pick contact and after place release the box rests in the
world, in between it rides the robot's base frame.

`box_entries` (used by the controller, mirroring jumps.jump_entries) returns the candidate
ENTRY frames -- the first few frames of each pick / place phase, the reach-down / set-down
approach -- so a skill is entered from its start (nearest-neighbour matched to the live pose
+ box pose) and then ridden to the phase end.
"""
import numpy as np

from . import config as C


def _speed(pos, fps=C.FPS):
    """Per-frame speed (m/s) of a world position track via forward differences."""
    if len(pos) < 2:
        return np.zeros(len(pos))
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    return np.concatenate([[d[0]], d])


def segment_phases(box_pos):
    """Segment one clip's box-height trajectory into pick / carry / place.

    box_pos: (T, 3) world box positions.
    Returns (skill, attach, info):
      skill  (T,) int   per-frame phase code (SKILL_PICK / SKILL_CARRY / SKILL_PLACE)
      attach (T,) bool  box rides the robot (True over [contact .. release])
      info   dict       pick/carry/place (start, stop) half-open spans + contact/release
    """
    T = len(box_pos)
    bz = box_pos[:, 2]
    z_rest = float(np.median(bz[:min(C.BOX_REST_FRAMES, T)]))
    z_peak = float(bz.max())
    hi = z_rest + C.BOX_CARRY_FRAC * (z_peak - z_rest)

    # CARRY = the lifted plateau; PICK / PLACE are the rise / fall on either side.
    above = np.where(bz > hi)[0]
    if len(above) == 0:                      # degenerate (box never really lifts): all carry
        carry_s, carry_e = 0, T - 1
    else:
        carry_s, carry_e = int(above[0]), int(above[-1])

    skill = np.empty(T, np.int32)
    skill[:carry_s] = C.SKILL_PICK
    skill[carry_s:carry_e + 1] = C.SKILL_CARRY
    skill[carry_e + 1:] = C.SKILL_PLACE

    # Attached = lifted clear of the floor OR being handled; collapse to one contiguous
    # [contact .. release] interval so a flicker mid-carry can't detach the box.
    held = (bz > z_rest + C.BOX_HOLD_DZ) | (_speed(box_pos) > C.BOX_HOLD_SPEED)
    hidx = np.where(held)[0]
    attach = np.zeros(T, bool)
    if len(hidx):
        contact, release = int(hidx[0]), int(hidx[-1])
        attach[contact:release + 1] = True
    else:                                    # never lifts: treat the carry plateau as held
        contact, release = carry_s, carry_e
        attach[carry_s:carry_e + 1] = True

    info = dict(pick=(0, carry_s), carry=(carry_s, carry_e + 1), place=(carry_e + 1, T),
                contact=contact, release=release, z_rest=z_rest, z_peak=z_peak)
    return skill, attach, info


def box_entries(lib):
    """Candidate pick / place ENTRY frames + their phase-end frames (global indices).

    Mirrors jumps.jump_entries: returns (pick_enter, pick_end_of, place_enter, place_end_of)
    where *_enter are arrays of global frame indices at the start of each phase and *_end_of
    maps each entry frame to the last frame of its phase (where the ride finishes).
    """
    skill = lib["skill"]
    fic = lib["frame_in_clip"]
    starts = np.where(fic == 0)[0]
    stops = np.append(starts[1:], len(skill))

    pick_enter, pick_end_of = [], {}
    place_enter, place_end_of = [], {}
    for rs, re in zip(starts, stops):
        for code, enter, end_of, n in (
                (C.SKILL_PICK, pick_enter, pick_end_of, C.PICK_ENTRY),
                (C.SKILL_PLACE, place_enter, place_end_of, C.PLACE_ENTRY)):
            idx = rs + np.where(skill[rs:re] == code)[0]
            if len(idx) == 0:
                continue
            phase_end = int(idx[-1])
            for f in idx[:n]:                # first n frames of the phase are entry points
                enter.append(int(f))
                end_of[int(f)] = phase_end
    return (np.array(pick_enter, np.int32), pick_end_of,
            np.array(place_enter, np.int32), place_end_of)


def carry_segments(lib):
    """Contiguous CARRY runs as (range_start, range_stop) half-open global spans."""
    skill = lib["skill"]
    fic = lib["frame_in_clip"]
    starts = np.where(fic == 0)[0]
    stops = np.append(starts[1:], len(skill))
    segs = []
    for rs, re in zip(starts, stops):
        m = skill[rs:re] == C.SKILL_CARRY
        if not m.any():
            continue
        idx = np.where(m)[0]
        segs.append((int(rs + idx[0]), int(rs + idx[-1] + 1)))
    return segs

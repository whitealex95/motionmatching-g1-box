"""UI 1's desired trajectory, walked with T. A copy of the loco driver's RouteFollow (motionmatching-g1-loco
sonic_g1_loco/mm_driver.py, main 9fbd029: no SLAM pose, the robot stands) and of the WaypointFollower it steers with
(emm/config.py), kept the same so the two windows can be merged."""
import json

import numpy as np

ROUTE_NAMES = ('halab', 'halab_gap')          # the map server's named routes (its live/route.py)


def moving(sticks):
    """Held keys take over from the trajectory only when they move or face (Shift alone does not)."""
    return any(np.linalg.norm(v) > 1e-6 for v in sticks)


def parse_points(spec):
    """'x,y;x,y;...' (map frame) or a .json from UI 1 -> [[x, y], ...]. The named routes live in the map server."""
    if spec.endswith('.json'):
        with open(spec) as fh:
            data = json.load(fh)
        pts = data['points'] if isinstance(data, dict) else data
        return [[float(x), float(y)] for x, y in pts]
    return [[float(v) for v in p.split(',')] for p in spec.split(';') if p.strip()]


class WaypointFollower:
    def __init__(self, waypoints, reach=1.0, loop=True):
        self.wp = np.array(waypoints, float)        # (n, 2)
        self.reach = reach
        self.loop = loop
        self.idx = 0
        self.done = False

    def stick(self, pos_xy, mag):
        """(left, right) sticks at world position pos_xy: move (3,) and face (3,) or None at the end."""
        if self.done:
            return np.zeros(3), None
        tgt = self.wp[self.idx]
        d = tgt - np.asarray(pos_xy, float)[:2]
        if np.linalg.norm(d) < self.reach:
            self.idx += 1
            if self.idx >= len(self.wp):
                if self.loop:
                    self.idx = 0
                else:
                    self.done = True
                    return np.zeros(3), None
            tgt = self.wp[self.idx]
            d = tgt - np.asarray(pos_xy, float)[:2]
        d = d / (np.linalg.norm(d) + 1e-9)
        v = np.array([d[0], d[1], 0.0])
        return mag * v, v


class RouteFollow:
    """The map server's copy of the route (UI 1 shows it, edits there take effect while walking). `tour` seeds the
    server with a named route, points or a .json and switches its Drive flag on. Without a map server the points are
    followed locally. The robot's map pose is judged, not the reference's. On a one-way route a waypoint also counts
    as passed once the robot is past its plane (the line through it across the segment that leads to it), and at the
    last one the robot stands and Drive goes off (the disc alone made the robot circle a 0.3 m goal). Without a SLAM
    pose the robot stands until it is back (paused); the bridge's drive_sticks runs the follower also while keys take
    over, so its waypoint progress follows the robot."""

    def __init__(self, br, tour, speed, tag='[box]'):
        self.br, self.speed, self.tag = br, speed, tag
        self.follower = None
        self.version = -1
        self.drive = False
        self.laps = 0
        self.done = False
        self.t_poll = 0.0
        self.local = False                # --tour followed without a map server: no SLAM, the reference's own xy
        self.paused = False               # Drive on but no SLAM pose: the robot stands until the pose is back
        if tour and tour != 'server':
            if br.map.request({'op': 'command', 'cmd': {'op': 'set_route', 'spec': tour, 'drive': True}}) is None:
                if tour in ROUTE_NAMES:
                    raise SystemExit(f'{tag} no map server: the named route {tour!r} lives there')
                print(f'{tag} no map server: following the local route', flush=True)
                self._set(parse_points(tour), True)
                self.drive = self.local = True

    def _set(self, points, loop, reach=0.6):
        f = self.follower
        keep = f is not None and (loop or (not f.done and len(points) == len(f.wp)))   # an edit mid-walk keeps the target
        self.follower = WaypointFollower(points, reach=reach, loop=loop)
        self.follower.idx = min(f.idx, len(points) - 1) if keep else 0
        self.done = False

    def robot_xy(self):
        """The robot's map xy, or None without a SLAM pose (the bridge's slam: none, or older than 0.5 s). Only the
        local route (no map server, so no SLAM) walks on the reference's xy. Before 2026-11-09 the reference's xy
        stood in for a lost pose and Drive walked on blind while the reference drifted off the robot."""
        return self.br.robot_xy() if self.br.slam is not None or self.local else None

    def _pause(self, on):
        if on != self.paused:
            self.paused = on
            self.br.say('trajectory paused: no SLAM pose, the robot stands (the keys still drive it)' if on
                        else 'trajectory resumed: the SLAM pose is back')

    def poll(self, now):
        if now - self.t_poll < 0.5:
            return
        self.t_poll = now
        r = self.br.map.request({'op': 'route'})
        if r is None:
            return
        drive, f = bool(r['drive']), self.follower
        if drive and not self.drive and f is not None and f.done:    # Drive after the end: again from the first waypoint
            f.idx, f.done, self.done = 0, False, False
        self.drive = drive
        if r['version'] != self.version:
            self.version = r['version']
            if r['points']:
                self._set(r['points'], r['loop'], float(r.get('reach', 0.6)))
            else:
                self.follower = None
                self.done = False

    def set_drive(self, on):
        self.drive = on
        self.br.map.request({'op': 'command', 'cmd': {'op': 'set_route', 'drive': on}})

    def stick(self, pos_xy, max_speed):
        """(left, right) for the controller, or None: nothing to follow, Drive off, the end of a one-way route, or
        pos_xy None (no SLAM pose: paused)."""
        f = self.follower
        if f is None or not self.drive:
            self.paused = False
            return None
        if pos_xy is None:
            self._pause(True)
            return None
        self._pause(False)
        p = np.asarray(pos_xy, float)[:2]
        i, n = f.idx, len(f.wp)
        if not f.loop and not f.done and n > 1:
            along = f.wp[i] - f.wp[i - 1] if i > 0 else f.wp[1] - f.wp[0]
            if (p - f.wp[i]) @ along >= 0:                 # past the waypoint's plane
                if i == n - 1:
                    f.done = True
                else:
                    f.idx += 1
        before = f.idx
        sl, sr = f.stick(p, self.speed / max_speed)
        if f.idx < before:
            self.laps += 1
            print(f'{self.tag} lap {self.laps} done', flush=True)
        if f.done and not self.done:
            self.done = True
            print(f'{self.tag} route done: standing', flush=True)
            self.set_drive(False)
        return None if sr is None else (sl, sr)

    @property
    def idx(self):
        return None if self.follower is None else int(self.follower.idx)

    def state(self):
        return {'idx': self.idx, 'version': self.version, 'laps': self.laps,
                'driving': self.drive and self.follower is not None, 'done': self.done, 'paused': self.paused}

    def text(self, auto):
        """The window's trajectory line."""
        f = self.follower
        if f is None:
            return 'none (draw it in UI 1, or --tour)'
        if f.done:
            return 'done: standing at its end (T, or Drive in UI 1, walks it again from waypoint 1)'
        if not self.drive:
            return f'off: {len(f.wp)} waypoints (T, or Drive in UI 1, walks them)'
        xy = self.robot_xy()
        if xy is None:
            return f'paused at waypoint {f.idx + 1}/{len(f.wp)}: no SLAM pose, standing (the keys still drive)'
        d = float(np.linalg.norm(f.wp[f.idx] - xy))
        return (f'to waypoint {f.idx + 1}/{len(f.wp)}, {d:.2f} m away, at {self.speed:.2f} m/s'
                + ('' if auto else '  (the keys override it)'))

"""UI 3's link to this driver, through the map server (motionmatching-g1-deploy live/operator.py): the page's held
keys and its commands, which the driver asks for with the `operator` REQ op, the driver's flags as the page lists
them, and a restart of the driver with changed flags. The same file is in motionmatching-g1-loco
(sonic_g1_loco/operator_link.py) and motionmatching-g1-box (sonic_g1_box/operator_link.py): the map server's op is
the contract, nothing is imported across the repos."""

import argparse
import collections
import math
import os
import sys
import time

import numpy as np

POLL_S = 0.05
NOTICES = 12


class Operator:
    """The page's keys and commands. poll() returns the commands queued since the last poll; keys() the held keys
    (a dict, or None when the page sends none: a closed tab, the keyboard on the map)."""

    def __init__(self, maplink, driver):
        self.map, self.driver = maplink, driver
        self.after = None
        self.held, self.cam_az_deg = None, 0.0
        self.t_poll = -1.0

    def poll(self, now):
        if now - self.t_poll < POLL_S:
            return []
        self.t_poll = now
        r = self.map.request({'op': 'operator', 'driver': self.driver, 'after': self.after})
        if r is None:                      # no map server, or one without UI 3: ask again in a second, not every step
            self.held = None
            self.t_poll = now + 1.0
            return []
        first = self.after is None
        self.after = int(r['seq'])         # also after a map server restart, whose numbers start again at 0
        self.held = r.get('keys')
        self.cam_az_deg = float(r.get('cam_az_deg') or 0.0)
        return [] if first else [c for _, c in r.get('cmds') or []]

    def keys(self):
        return self.held if self.held and any(self.held.values()) else None


def keys_stick(keys, which, run_scale, cam_azimuth=None):
    """genoview_g1.keyboard_stick on the page's held keys: a world-plane (3,) vector, 'left' = W A S D move,
    'right' = the arrows face, Shift scales by run_scale, cam_azimuth (deg) turns W into that heading."""
    k = keys or {}
    if which == 'left':
        x, y = float(bool(k.get('w'))) - float(bool(k.get('s'))), float(bool(k.get('a'))) - float(bool(k.get('d')))
    else:
        x = float(bool(k.get('up'))) - float(bool(k.get('down')))
        y = float(bool(k.get('left'))) - float(bool(k.get('right')))
    mag = math.hypot(x, y)
    if mag > 0.0:
        s = run_scale if k.get('shift') else 1.0
        x, y = x / mag * s, y / mag * s
    if cam_azimuth is None:
        return np.array([x, y, 0.0])
    a = math.radians(cam_azimuth)
    return np.array([x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a), 0.0])


class Flags:
    """The driver's flags for the page: `live` ones are set at once (`apply(name, value)` does it), `restart` ones
    take a restart of the driver, which runs its own command again with the new values. argv() is that command
    line, with the live changes so far in it, so a restart keeps them."""

    def __init__(self, parser, args, live, restart):
        self.args = args
        self.actions = {a.dest: a for a in parser._actions if a.option_strings and a.dest in set(live) | set(restart)}
        self.live = set(live)
        self.order = [n for n in list(live) + list(restart) if n in self.actions]
        self.argv0 = list(getattr(sys, 'orig_argv', None) or [sys.executable] + sys.argv)
        self.changed = {}
        self.notices = collections.deque(maxlen=NOTICES)

    def say(self, tag, msg):
        print(f'{tag} {msg}', flush=True)
        self.notices.append((round(time.time(), 2), msg))

    def flag(self, name):
        return max(self.actions[name].option_strings, key=len)

    def value(self, name, v):
        """The flag's value from the page, as argparse would read it; ValueError when it is not one."""
        a = self.actions[name]
        if v is None or (isinstance(v, str) and v.strip() == '' and a.type in (int, float)):
            if a.default is None:
                return None
            raise ValueError(f'{self.flag(name)} needs a value')
        try:
            if a.type in (int, float):
                v = a.type(v)
                if not math.isfinite(v):
                    raise ValueError
            elif a.type is not None:
                v = a.type(str(v))
        except (ValueError, TypeError, argparse.ArgumentTypeError) as e:
            raise ValueError(f'{self.flag(name)}: {v!r} is not a valid value'
                             + (f' ({e})' if isinstance(e, argparse.ArgumentTypeError) else '')) from None
        else:
            v = str(v)
            if v == '' and a.default is None:
                v = None
        if a.choices is not None and v not in a.choices:
            raise ValueError(f'{self.flag(name)}: one of {", ".join(map(str, a.choices))}')
        return v

    def spec(self):
        out = []
        for n in self.order:
            a = self.actions[n]
            v = getattr(self.args, n)
            kind = 'float' if a.type is float else 'int' if a.type is int else 'str'
            out.append({'name': n, 'flag': self.flag(n), 'kind': kind, 'live': n in self.live,
                        'value': v if v is None or isinstance(v, (int, float, str)) else ','.join(map(str, v)),
                        'choices': None if a.choices is None else [str(c) for c in a.choices],
                        'help': (a.help or '').replace('%%', '%')})
        return out

    def text(self, v):
        return ','.join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)

    def note_live(self, name, v):
        self.changed[name] = v

    def argv(self, extra=None):
        changes = dict(self.changed, **(extra or {}))
        out, skip = [], False
        flags = {self.flag(n): n for n in changes}
        alias = {o: self.flag(n) for n in changes for o in self.actions[n].option_strings}
        for a in self.argv0:
            if skip:
                skip = False
                continue
            head = a.split('=', 1)[0]
            if head in alias:
                skip = '=' not in a
                continue
            out.append(a)
        for f, n in flags.items():
            v = changes[n]
            if v is not None:
                out += [f, self.text(v)]
        return out

    def restart(self, extra, close):
        """Runs the driver's command again with `extra` ({name: value}) on top: close() first frees its ports."""
        argv = self.argv(extra)
        close()
        try:
            import zmq
            zmq.Context.instance().destroy(linger=0)
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, argv)    # argv[0] as typed: the launchers' kill patterns match `^python ...`

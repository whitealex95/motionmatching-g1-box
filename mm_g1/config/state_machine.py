"""The B-driven pick/carry/place behavior: command speeds, the move-to-pick
approach, and the interactive box spawn. (The states themselves are
mm_g1.states.State; transitions live in mm_g1.controller.)"""

# Desired locomotion speed (m/s) fed to the trajectory springs. Full stick = MAX_SPEED,
# a walk; holding Shift scales it by RUN_SCALE, a run (1.2 m/s). Same as the loco
# repo's live.mm_driver.
MAX_SPEED = 0.8
RUN_SCALE = 1.5

# Full-stick command speed while CARRYING the box. The carry clips are near-stationary -- the
# root translates at ~0.5 m/s on average (p95 ~0.75, max ~0.97), so commanding the locomotion
# speed here would just ask for motion the carry data cannot supply. Capping the command at
# the data's ~p95 lets WASD nudge the carry as fast as the clips actually move, no faster.
CARRY_MAX_SPEED = 0.75

# --- Approach heuristics (move-to-pick, ported from motionmatching-g1-shelf) -----
# B plans a walking route to the pick stance computed from the LIVE box pose (the
# stance-to-box offset recorded in the baked pick clip, inverted): straight to a way-in
# point behind the stance, a rounded corner, then in along the stance heading, aiming
# past it so the walk never decays into the slow-walk dead zone. On the final leg the
# root is pinned to the rail; the pick entry fires at the stance-plane crossing.
MOVE_WAYIN = 0.6         # way-in point this far behind the stance (m)
MOVE_OVERSHOOT = 0.35    # route/tap target past the stance (m), keeps the walk alive
# Unlike the shelf demo (which cuts into its clip at the stance crossing, still
# walking), the pick here starts only after the reference has STOPPED at the
# stance: the tracking policy cannot stop instantly, and the SceneBot demo's own
# sequence also settles before the squat. The walk decelerates toward the
# stance, holds inside the arrive radius, and the entry waits for the root to
# settle below the arrive speed.
MOVE_ARRIVE_NEAR = 0.12  # hold-still radius around the stance (m)
MOVE_ARRIVE_LOOSE = 0.30 # a settled stop this near the stance also counts, after MOVE_ARRIVE_LOOSE_S
MOVE_ARRIVE_LOOSE_S = 1.0
# Approach speeds: the route walk is clipped to (MOVE_ROUTE_SPEED_MIN, MOVE_ROUTE_SPEED_MAX);
# inside 0.8 m the endgame servo runs between MOVE_END_SPEED_MIN and _MAX and stops
# commanding MOVE_STOP_DIST before the stance. A tracking policy lags decelerations, so a
# real robot overshoots a fast reference stop into the box: scale these down for it.
MOVE_ROUTE_SPEED_MIN = 0.25
MOVE_ROUTE_SPEED_MAX = 1.2
MOVE_END_SPEED_MIN = 0.35
MOVE_END_SPEED_MAX = 0.55
MOVE_STOP_DIST = 0.18
# The pick fires only with the root within MOVE_FWD_TOL of the stance ALONG the approach
# rail (a lateral miss is absorbed by the arms, a forward miss is not); short of that the
# endgame servo keeps commanding, even inside MOVE_STOP_DIST. None: no such gate.
MOVE_FWD_TOL = None
MOVE_ARRIVE_YAW = 0.6    # yaw tolerance at the stance (rad)
MOVE_ARRIVE_SPEED = 0.25  # root speed below this counts as settled (m/s)
MOVE_TIMEOUT = 8.0       # per-leg give-up (s)
SNAP_RADIUS = 4.0        # rail pin active inside this radius
SNAP_HALFLIFE = 1.0      # rail pin half-life (s)

# Where the box spawns, expressed in the robot's start frame (so it is always a reachable
# distance in front of wherever the character begins / resets), plus its resting height. The
# resting orientation is taken from the data so the pick entry lines up (see controller).
BOX_SPAWN_FWD = 1.6      # metres in front of the robot's start facing
BOX_SPAWN_LAT = 0.0      # metres to the robot's left (+) / right (-)

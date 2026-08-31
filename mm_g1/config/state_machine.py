"""The B-driven pick/carry/place behavior: command speeds, the move-to-pick
approach, and the interactive box spawn. (The states themselves are
mm_g1.states.State; transitions live in mm_g1.controller.)"""

# Desired locomotion speed (m/s) fed to the trajectory springs. Full stick = MAX_SPEED;
# holding Shift scales it to a walk (GenoView's 0.4 scale).
MAX_SPEED = 2.5
WALK_SCALE = 0.4

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

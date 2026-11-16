// The predefined command sequence. Each step waits for a condition on the robot, then acts on the
// world. Conditions use simulated time, so a run is the same every time for a given controller.

const after = (s) => (ctx) => ctx.stepTime >= s;

// Box C stands at the -x edge of table T1 (World.boxOnTable), which faces east. The table is west of where the robot
// walks with boxes A and B, so the way to it does not pass between them.
export const INITIAL_TABLES = [{ label: 'T1', pos: [-4.6, -5.0], yaw: Math.PI + 0.3, size: [0.6, 0.9, 0.5] }];

export const INITIAL_BOXES = [
  { label: 'A', pos: [2.0, 0.4], yaw: 0.3 },
  { label: 'B', pos: [-1.8, 2.2], yaw: -0.6 },
  { label: 'C', table: 'T1', edge: 2 },
];

export const SCENARIO = [
  { title: 'Pick up box A', when: after(1.0), act: (w) => w.setPick('A') },
  { title: 'Box A jumps 1.4 m while the robot bends down to it',
    when: (ctx) => (ctx.phase === 'pick-pre' && ctx.hipz < 0.71) || ctx.holding, act: (w) => w.moveBoxBy('A', -0.3, -1.4, 1.0) },
  { title: 'Carry box A with the keyboard (W, then W+D)',
    when: (ctx) => ctx.holding && ctx.uprightTime > 1.0, act: (w) => w.drive([[2.2, 'W'], [1.8, 'WD']]) },
  { title: 'Put box A down at the blue marker',
    when: (ctx) => ctx.driveDone && ctx.stepTime > 0.5, act: (w) => w.setPlaceAhead(2.0, 0.8, 0.5) },
  { title: 'The marker jumps while the robot carries the box to it',
    when: (ctx) => ctx.holding && ctx.stepTime > 1.8, act: (w) => w.setPlaceAhead(1.6, -1.3, -0.4) },
  { title: 'Pick up box B', when: (ctx) => !ctx.holding && ctx.uprightTime > 1.0, act: (w) => w.setPick('B') },
  { title: 'Box B jumps during the approach', when: after(1.6), act: (w) => w.moveBoxBy('B', 1.0, -0.7, 0.8) },
  { title: 'Put box B down next to box A',
    when: (ctx) => ctx.holding && ctx.uprightTime > 1.0, act: (w) => w.setPlaceNear('A', 0.0, 1.0, 0.0) },
  { title: 'The marker moves to the other side of box A while box B is being lowered',
    when: (ctx) => (ctx.phase === 'place-pre' && ctx.hipz < 0.71) || !ctx.holding, act: (w) => w.movePlaceBy(0.0, -2.0) },
  { title: 'Pick up box C from the table', when: (ctx) => !ctx.holding && ctx.uprightTime > 1.0, act: (w) => w.setPick('C') },
  { title: 'Put box C down on the floor',
    when: (ctx) => ctx.holding && ctx.uprightTime > 1.0, act: (w) => w.setPlaceAhead(0.0, 1.8, 1.2) },
  { title: 'End', when: (ctx) => !ctx.holding && ctx.uprightTime > 1.5, act: (w) => w.finish() },
];

"""Combine PD and virtual-force torque with an explicit clipping order."""


def compose_torque(pd_torque, virtual_torque, clip=None, order='after-force'):
    if order not in ('after-force', 'before-force'):
        raise ValueError(f'Unknown torque clipping order: {order}')
    torque = pd_torque
    if clip is not None and order == 'before-force':
        torque = clip(torque)
    if virtual_torque is not None:
        torque = torque + virtual_torque
    if clip is not None and order == 'after-force':
        torque = clip(torque)
    return torque


def add_torque_clip_argument(parser):
    parser.add_argument('--torque-clip', choices=['after-force', 'before-force'],
                        default='after-force',
                        help='SuperMimic: clip combined torque (default), or clip PD '
                             'before adding virtual force with no final clipping. '
                             'The before-force variant bypasses the scene motor clip. '
                             'SONIC keeps its existing scene motor limits.')


def configure_torque_clipping(model, policy, tracker, order):
    """Move the scene motor clip before virtual force for the SuperMimic variant."""
    if tracker != 'supermimic':
        if order != 'after-force':
            raise ValueError('--torque-clip before-force currently requires SuperMimic')
        return None
    if order == 'before-force':
        import numpy as np
        # Preserve both existing limits on PD torque, then remove the final
        # MuJoCo motor clip so it cannot suppress the added virtual-force torque.
        limits = policy.torque_limits.copy()
        active = model.actuator_ctrllimited[:29].astype(bool)
        limits[active, 0] = np.maximum(limits[active, 0], model.actuator_ctrlrange[:29][active, 0])
        limits[active, 1] = np.minimum(limits[active, 1], model.actuator_ctrlrange[:29][active, 1])
        joints = model.actuator_trnid[:29, 0]
        if not np.all(model.actuator_gear[:29, 0] == 1):
            raise ValueError('Expected direct torque motors with unit gear')
        for flags, ranges in ((model.actuator_forcelimited[:29], model.actuator_forcerange[:29]),
                              (model.jnt_actfrclimited[joints], model.jnt_actfrcrange[joints])):
            active = flags.astype(bool)
            limits[active, 0] = np.maximum(limits[active, 0], ranges[active, 0])
            limits[active, 1] = np.minimum(limits[active, 1], ranges[active, 1])
        model.actuator_ctrllimited[:29] = False
        model.actuator_forcelimited[:29] = False
        model.jnt_actfrclimited[joints] = False
        return lambda torque: np.clip(torque, limits[:, 0], limits[:, 1])
    return policy.clip_torque

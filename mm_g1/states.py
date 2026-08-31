"""Phase labels baked into the motion library + the controller's runtime states."""
from enum import Enum, IntEnum, auto


class Phase(IntEnum):
    """Per-frame label stored as an int array in lib['phase'] (the .npz cache).
    Any non-LOCO code keeps that frame out of the locomotion search; DISABLED
    frames belong to no database at all."""
    LOCO = 0
    PICK = 1
    CARRY = 2
    PLACE = 3
    DISABLED = 4

    @property
    def db(self):
        """Key of this phase's feature database in features.build_db."""
        return self.name.lower()


class State(Enum):
    """MotionMatcher runtime state. MOVE_TO_PICK is controller-only (the
    walk to the pick stance); frames themselves are labeled with Phase.

        LOCOMOTION --B--> MOVE_TO_PICK --> PICK (ride) --> CARRY (search)
                   --B--> PLACE (ride) --> LOCOMOTION
    """
    LOCOMOTION = auto()
    MOVE_TO_PICK = auto()
    PICK = auto()
    CARRY = auto()
    PLACE = auto()

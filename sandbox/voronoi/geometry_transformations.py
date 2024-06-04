import numpy as np


def get_tras_rot_2d(pos, azm=0, orig=None):
    # pos is a numpy array (N, 2)
    # azm is an angle in rads
    # orig is None or a vector of a position

    # Previous checks
    if len(pos.shape) == 2:
        assert pos.shape[1] == 2
    else:
        assert len(pos) == 2
        pos = np.array([pos])

    if orig is None:
        orig = np.array([[0.0, 0.0]])
    else:
        if len(orig.shape) == 2:
            assert orig.shape == (1, 2)
        else:
            assert len(orig) == 2
            orig = np.array([orig])

    tras_pos = pos - orig
    rot_mat = np.array([[np.cos(azm), np.sin(azm)], [-np.sin(azm), np.cos(azm)]])
    new_pos = np.dot(tras_pos, rot_mat)
    return new_pos


def get_tras_2d(pos, orig=None):
    # pos is a numpy array (N, 2)
    # orig is None or a vector of a position
    
    # Previous checks
    if len(pos.shape) == 2:
        assert pos.shape[1] == 2
    else:
        assert len(pos) == 2
        pos = np.array([pos])

    if orig is None:
        orig = np.array([[0.0, 0.0]])
    else:
        if len(orig.shape) == 2:
            assert orig.shape == (1, 2)
        else:
            assert len(orig) == 2
            orig = np.array([orig])

    tras_pos = pos - orig
    return tras_pos
    
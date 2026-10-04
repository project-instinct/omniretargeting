"""Optional exact native kinematics and sparse assembly for solver guards.

The guards retain their original residuals, budgets and constraint order.
Only their common point-Jacobian and block-matrix primitives change. The
full constraint/LM comparison passed; the separate switch retains the reference.
"""

import os


def enabled():
    return os.environ.get('OMNIRETARGETING_FAST_GUARDS', '1') == '1'

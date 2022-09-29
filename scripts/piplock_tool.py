# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------

import sys
file = sys.argv[1]
with open(file, 'r') as f1:
    text = f1.read()
package_list = text.split('Would install ')[1].split(' ')
content = '\n'.join(['=='.join(i.rsplit('-', 1)) for i in package_list])
with open(file, 'w') as f2:
    f2.write(content)

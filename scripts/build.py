# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------
import os
import shutil
import subprocess

if __name__ == "__main__":
    src_pth = "config"
    target_path = "corl/config"

    if os.path.exists(src_pth):
        if os.path.exists(target_path) and os.path.isdir(target_path):
            shutil.rmtree(target_path)
        shutil.copytree(src_pth, target_path)

    #
    # Some final post install setup items if not CICD
    #
    # Using environment variables:
    #
    # github CI/CD provides several predefined environment variables that indicate
    # the pipeline context. You can check for these variables within your code to
    # determine the execution environment.
    #
    # $CI_PIPELINE_SOURCE: This variable indicates the source that triggered the
    # pipeline (e.g., push, merge request event). If the variable is not set then
    # we are not running in CICD in github!
    if os.environ.get("CI_PIPELINE_SOURCE") is None:
        result = subprocess.run(['pre-commit', 'install'], capture_output=True, text=True)
        print(result.stdout)
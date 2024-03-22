---
title: CoRL
---

<!-- {{ include_top_banner() }} -->

<!-- {{ include_github_badges() }} -->

<!-- {{ include_repo_badges() }} -->

# 1. Core ACT3 Reinforcement Learning Library

> **This repository and corresponding documentation site are currently under construction. We are still porting items and updating instructions for GitHub.**

## 1.1. Summary

The Core ACT3 Reinforcement Learning library (CoRL) is created and maintained by the Air Force Research Laboratory’s (AFRL) [Autonomy Capability Team (ACT3)](https://www.afrl.af.mil/ACT3/). CoRL is intended to enable scalable deep reinforcement learning (RL) experimentation in a manner extensible to new simulations and new ways for the learning agents to interact with them. The objective is to make RL research easier by removing lock-in to particular simulations.

### 1.1.1. Benefits

- Makes RL environment development significantly easier
- Provides hyper configurable environments, agents and experiments
- Record observations by adding a few lines of config (instead of creating a new file for each observation)
- Reuse glues/dones/rewards between different tasks if they are general
- Uses an episode parameter provider (EPP) to randomize both domain and curriculum learning
- Has an integration first focus, which means that integrating agents to the real world or different simulators is significantly easier

### 1.1.2. Related Publications

- [CoRL: Environment Creation and Management Focused on System Integration](https://arxiv.org/abs/2303.02182)
- [Inside the special F-16 the Air Force is using to test out ML](https://breakingdefense.com/2023/01/inside-the-special-f-16-the-air-force-is-using-to-test-out-ML/)
- [AFRL, AFTC collaborate on future technology via weeklong autonomy summit](https://www.wpafb.af.mil/News/Article-Display/Article/3244878/afrl-aftc-collaborate-on-future-technology-via-weeklong-autonomy-summit/)
- [Demonstrating and testing machine learning applications in aerospace](https://aerospaceamerica.aiaa.org/year-in-review/demonstrating-and-testing-artificial-intelligence-applications-in-aerospace/)

## 1.2. Documentation

Documentation for the CoRL repository can be accessed directly as files in this repository, as a public documentation site, or can be built locally as an MkDocs site.

### 1.2.1. Guides

- [Quick Start Guide](guides/quick-start-guide.md)

### 1.2.2. Documentation Web Site

The [full public documentation site](https://act3-ace.github.io/CoRL/) is available on GitHub pages.

### 1.2.3. Local Documentation

A local version of the documentation site can be built using [MkDocs](https://www.mkdocs.org/).

Build the documentation:

```sh
mkdocs build
```

> Follow  CLI prompts, as needed, to install all required plugins.

Serve the documentation:

```sh
mkdocs serve
```

## 1.3. Notices and Warnings

### 1.3.1. Initial Contributors

Initial contributors include scientists and engineers associated with the [Air Force Research Laboratory (AFRL)](https://www.afrl.af.mil/), [Autonomy Capability Team 3 (ACT3)](https://www.afrl.af.mil/ACT3/), and the [Aerospace Systems Directorate (RQ)](https://www.afrl.af.mil/RQ/).

### 1.3.2. Citing CoRL

If you use CoRL in your work, please use the following BibTeX to cite the CoRL white paper:

```bibtex
@inproceedings{
  title={CoRL: Environment Creation and Management Focused on System Integration},
  author={Justin D. Merrick, Benjamin K. Heiner, Cameron Long, Brian Stieber, Steve Fierro, Vardaan Gangal, Madison Blake, Joshua Blackburn},
  year={2023},
  url={https://arxiv.org/abs/2303.02182}
}
```

To cite the source code, use the **Cite this repository** option on GitHub to access the reference.

### 1.3.3. Distribution Statement

Approved for public release: distribution unlimited.

#### 1.3.3.1. Case Number

|    Date    |     Release Number     | Description      |
| :--------: | :--------------------: | :--------------: |
| 2022-05-20 |     AFRL-2022-2455     | Initial release  |
| 2023-03-02 | APRS-RYZ-2023-01-00006 | Second release   |
| 2024-21-03 | AFRL-2024-1562         | Thirst release   |

#### 1.3.3.2. Designation Indicator

- Controlled by: Air Force Research Laboratory (AFRL)
- Controlled by: AFRL Autonomy Capability Team (ACT3)

#### 1.3.3.3. Points of Contact

- [Terry Wilson](mailto:terry.wilson.11@us.af.mil)
- [Benjamin Heiner](mailto:benjamin.heiner@us.af.mil)
- [Kerianne Hobbs](mailto:kerianne.hobbs@us.af.mil)

##### 1.3.3.3.1. Repository Contributors

{{ get_authors() }}

##### 1.3.3.3.2. Documentation Contributors

{{ git_site_authors }}

{{ include_glossary_abbreviations() }}

{{ include_bottom_banner() }}

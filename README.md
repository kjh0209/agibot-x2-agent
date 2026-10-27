# AgiBot X2 Home Agent

A language-driven agent for the AgiBot X2 Ultra humanoid that walks into a home it has never seen, builds its own map, learns what each room and object is, and carries out chat commands such as "go to the bed" or "pick up the ball and put it on the sofa".

Everything runs in a self-built MuJoCo simulator that publishes the robot's official ROS 2 topics and enforces its real control protocol, so the agent code is written against the same interfaces it would use on hardware.

Built by Jihyeok Kim (KAIST) in August 2026 as an independent project. Simulation only: no physical robot was used.

## Why

Household robots will enter homes that nobody has mapped for them. On the first day, the robot has to learn the layout and the meaning of each place within minutes, keep that knowledge correct, and let a person talk to it naturally. I built the whole loop to find out which part of this is actually hard.

## How it works

```
 MuJoCo world (hand-built houses, photogrammetry scans of real rooms)
        │  official X2 topics, motion-control protocol, head camera
        ▼
 Geometry layer      simulated LiDAR (4 heights x 96 bearings, 5 Hz)
                     -> log-odds occupancy grid with per-cell height (slam_map.py)
        ▼
 Semantic layer      frontier exploration; 4-frame scans -> VLM labels
                     {room, furniture, objects}; fast model proposes,
                     stronger model confirms on site before anything enters the map
        ▼
 Language layer      tool-calling LLM sees compact state (places, rooms, objects, pose),
                     not raw frames, and acts through tools
        ▼
 Chat app            FastAPI + WebSocket backend, browser frontend with voice output
```

Tools available to the LLM: `explore`, `survey_map`, `remember_place`, `forget_map`, `go_to_place`, `go_to_object`, `go_to_point`, `move`, `turn`, `turn_head`, `face`, `look`, `pick_up`, `put_down`, `open_door`, `close_door`, `sit_down`, `lie_down`, `stand_up`, `stop`, `speak`, `play_motion`, `show_emoji`, `get_status`.

Safety: a geometric motion gate (base footprint vs. obstacles with clearance), a reflex layer that stops before collisions, a hard-wired stop path that bypasses the LLM, and an audit log of every tool call.

## Results (simulation)

| Measure | Value | Note |
|---|---|---|
| Occupancy map accuracy | 91.3% | mapped cells within 0.15 m of true geometry |
| Full house from a blank map | 14.4 min | 13 places, each verified on site; audit found 0 places inside furniture |
| Fetch (find, pick up, carry) | 156 s -> 42 s | after cost-map A* with clearance preference |
| "Look at X" | minutes -> 15 s | direct `face` tool instead of visual search |
| Errand smoke test | 9 / 9 | `scripts/smoke_test.sh`, run after every change |

## What failed and what it taught me

- **Geometry was easy; meaning was the bottleneck.** A room was labeled "bedroom" because four frames from one spot counted as four votes. The fix: one vote per scan, two independent votes from positions 0.8 to 3.5 m apart, and a re-check of every room with an evidence prompt ("bedroom only if a bed is visible").
- **Fixed thresholds do not transfer between homes.** On a scanned real bed, blanket folds reached 0.7 to 1.0 m, so every fixed "tall obstacle" height failed. A relative rule (more than 0.45 m above the surface is a wall or headboard) worked in every world.
- **A ground-truth map hides failures.** After removing the simulator's privileged map, A* wandered into unobserved space and the robot's own cell was marked occupied. Neither showed up before.
- **Exploration is slow.** One round covers about 1 to 1.5 rooms; a full house needs repeated rounds.

Limits: the VLM and LLM are cloud models (Claude Haiku / Sonnet through the API), not on the robot; low-resolution renders (320x240) still cause naming errors.

## Repository layout

```
ros2_ws/src/x2_sim/        MuJoCo simulator node: worlds, head camera, LiDAR, motion-control mock, MJPEG viewer
ros2_ws/src/robot_agent/   LLM agent: tools, occupancy mapping, VLM labeling, safety, audit log, chat CLI
ros2_ws/tools/             photogrammetry scan (GLB) -> MuJoCo world converter
ros2_ws/assets/tex/        CC0 textures
chat/                      chat app (FastAPI + WebSocket backend, static frontend)
docker/                    Ubuntu 22.04 + ROS 2 Humble dev container
scripts/                   environment, setup, supervisor, smoke test, chaos test
deploy/                    systemd units for running on the robot's host PC
docs/                      hardware-transition checklist, network setup, demo script (Korean)
```

The design log (`진행상황_정리.txt`) and the roadmap are in Korean and record every decision and bug in order.

## Running it

Requirements: Docker (Colima on Apple Silicon), an Anthropic API key.

```bash
cp .env.example .env            # set ANTHROPIC_API_KEY
cd docker && docker compose build && docker compose up -d
docker exec -it agibot-dev bash

# inside the container
cd /ws && colcon build && source install/setup.bash
bash /scripts/supervise_sim.sh &          # simulator (auto-restarts on crash)
ros2 run x2_sim mjpeg_server &            # live camera viewer
ros2 run robot_agent agent_node &         # LLM agent
python3 /chat/backend/main.py             # chat app on http://localhost:8000
```

Optional environment variables: `AGENT_MODEL` (default `claude-haiku-4-5`), `VLM_MODEL`, `VLM_VERIFY_MODEL`.

The AgiBot SDK and URDF are not redistributed here; download them from the official developer site (https://x2-aimdk.agibot.com) into `sdk/downloads/`. The photogrammetry scan worlds need your own GLB scans, converted with `ros2_ws/tools/convert_scan.py`.

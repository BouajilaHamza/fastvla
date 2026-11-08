# Unitree G1 Humanoid Embodiment: MuJoCo Physics & Motion Evaluation Report

## 1. Diagnostic: Why the Previous Kick Motion Looked Awkward

When viewing the previous rollout video, the motion appeared floating, disjointed, or "not good". A thorough diagnostic identified the root cause:
- **Simplified Wireframe Renderer**: The previous script ([`render_g1_rollout.py`](file:///home/hamza/Documents/BouajilaHamza/fastvla/scripts/visualization/render_g1_rollout.py)) relied on a hand-crafted geometric stick-figure skeleton drawn in Matplotlib.
- **Missing Embodiment Geometry**: It had no robot chassis, no motorized joint shells, no foot sole surfaces, no ground contact planes, and did not utilize the pelvis orientation quaternion from the dataset.
- **Kinematic Distortion**: Without the physical robot body, high angular momentum kicks look unnatural because the human eye expects to see the actual mechanical limbs, pelvis tilt, and counter-balancing arm guards of the robot.

---

## 2. Solution: Official Unitree G1 3D Robot Embodiment in MuJoCo

We integrated the official **Unitree G1 23-DoF MuJoCo robot model** (`g1_23dof.xml`) from Unitree Robotics:
1. **Authentic 3D Robot Meshes**: Loaded all official `.STL` meshes for the robot head, camera visor, torso chassis, waist yaw gear, hip pitch/roll/yaw motor housings, knee linkages, ankle pitch/roll motors, and rubber foot soles.
2. **Exact Kinematics & Physics Engine**: Forward kinematics computed directly by MuJoCo (`mujoco.mj_forward()`), respecting joint constraints and real physical links.
3. **Quaternion Orientation Mapping**: Corrected the dataset quaternion convention $(x, y, z, w) \to (w, x, y, z)$ to ensure the robot stands naturally upright on the floor with dynamic pelvis counter-tilt.
4. **Photorealistic Floor & Lighting**: Rendered on a textured checkerboard floor with directional sunlight, specular highlights, and contact shadows.

---

## 3. Evaluated Kick Regimes: Roundhouse Kick vs. Crisp Side Kick

To ensure the captain has the clearest visual verification, we rendered **two distinct kick motions** plus the fluid Capoeira dance:

### A. Roundhouse Kick (ركلة دائرية ديناميكية مع الدوران)
- **Prompt**: `"نفذ ركلة دائرية سريعة وقوية بالساق مع الارتكاز والدوران"`
- **Kinematic Nature**: Ballistic circular strike. The robot pivots on its left support foot, leans its pelvis backwards for counter-weight, and whips the right leg in an arc.
- **Rollout Assets**:
  - MP4 Video (60 FPS): [`assets/unitree_g1_roundhouse_kick_embodiment.mp4`](file:///home/hamza/Documents/BouajilaHamza/fastvla/assets/unitree_g1_roundhouse_kick_embodiment.mp4)
  - Animated GIF: [`assets/unitree_g1_roundhouse_kick_embodiment.gif`](file:///home/hamza/Documents/BouajilaHamza/fastvla/assets/unitree_g1_roundhouse_kick_embodiment.gif)

### B. Crisp Lateral Side Kick (ركلة جانبية مباشرة مع فرد الساق)
- **Prompt**: `"سدد ركلة جانبية مباشرة وقوية مع فرد الساق بالكامل"`
- **Kinematic Nature**: Direct, linear extension kick (Karate/Taekwondo style). The robot chambers its right knee, drives the heel directly sideways with full knee extension, and holds a rigid counter-balancing arm guard. This motion is exceptionally clean and easy to judge visually.
- **Rollout Assets**:
  - MP4 Video (60 FPS): [`assets/unitree_g1_side_kick_embodiment.mp4`](file:///home/hamza/Documents/BouajilaHamza/fastvla/assets/unitree_g1_side_kick_embodiment.mp4)
  - Animated GIF: [`assets/unitree_g1_side_kick_embodiment.gif`](file:///home/hamza/Documents/BouajilaHamza/fastvla/assets/unitree_g1_side_kick_embodiment.gif)

### C. Capoeira Ginga (رقصة الكابويرا الإيقاعية المتدفقة)
- **Prompt**: `"أدِّ حركات رقصة الكابويرا البرازيلية القتالية مع التمايل المستمر (الجينغا)"`
- **Kinematic Nature**: Periodic low center-of-mass sway with alternating dual-foot contact and continuous arm guard coordination.
- **Rollout Assets**:
  - MP4 Video (60 FPS): [`assets/unitree_g1_capoeira_dance_embodiment.mp4`](file:///home/hamza/Documents/BouajilaHamza/fastvla/assets/unitree_g1_capoeira_dance_embodiment.mp4)
  - Animated GIF: [`assets/unitree_g1_capoeira_dance_embodiment.gif`](file:///home/hamza/Documents/BouajilaHamza/fastvla/assets/unitree_g1_capoeira_dance_embodiment.gif)

---

## 4. Live Repository Pushes

1. **GitHub Repository**:
   - Pushed commit [`8b1e7eb`](https://github.com/BouajilaHamza/fastvla/commit/8b1e7ebb459220fe44365b2633ff800d9818817a) on `origin/main`.
   - Updated [`README.md`](file:///home/hamza/Documents/BouajilaHamza/fastvla/README.md) on GitHub with the 3-column MuJoCo embodiment showcase.
   - Added the rendering pipeline script: [`scripts/visualization/render_g1_mujoco.py`](file:///home/hamza/Documents/BouajilaHamza/fastvla/scripts/visualization/render_g1_mujoco.py).
2. **Hugging Face Hub**:
   - Model Card: [hamzabouajila/fastvla-unitree-g1-ar](https://huggingface.co/hamzabouajila/fastvla-unitree-g1-ar)
   - Dataset Card: [hamzabouajila/ar-unitree-g1-motions](https://huggingface.co/datasets/hamzabouajila/ar-unitree-g1-motions)

# movements

<img width="1160" height="428" alt="Screenshot 2026-06-22 090311" src="https://github.com/user-attachments/assets/38f03302-b4d0-4aa6-9e6b-d1e15f8f48e8" />

This is a motion capture website to capture human joints, movements, and positions that can be directly mapped to the MuJoCo Humanoid 5 in ManiSkill and Unitree H1 robot in Isaac Sim. 

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Running the Motion Capture App Locally (Terminal 1)

Install dependencies in `my-app`
```bash
npm install
```

Run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

Select a camera source and begin recording. The browser will generate MediaPipe landmarks and stream them to the WebSocket server.ct

## Websockets (Terminal 2)
The WebSocket server receives pose frames from the browser and writes live motion files.

Start the server:
```bash
python websocket_pose_server.py
```

Expected output:
```bash
WebSocket server listening on ws://0.0.0.0:8765
Client connected
Received frame ...
```

Generated files:
```bash
received_frames/
├── latest_frame.json
├── latest_h1_frame.json
├── latest_mujoco_frame.json
└── frames.jsonl
```

## ManiSkill MuJoCo Humanoid-v5(Terminal 3)
Launch the MuJoCo Humanoid-v5 viewer:
```bash
python play_latest_mujoco.py
```

The viewer continuously reads and updates the humanoid poses in real time
```bash
received_frames/latest_mujoco_frame.json
```

## ManiSkill UniTree H1 Robot (Terminal 4)
In ManiSkill, you must download all UniTree H1 asset. If using Isaac Sim, launch and load the Unitree H1 environment. 

The retargeting pipeline generates which can be used by Isaac Sim, Isaac Lab, or ManiSkill
```bash
received_frames/latest_h1_frame.json
```



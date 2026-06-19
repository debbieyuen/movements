import asyncio
import json
import time
from pathlib import Path

import websockets

from live_h1_remapper import pose_to_h1

HOST = "0.0.0.0"
PORT = 8765

OUT_DIR = Path("received_frames")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_FILE = OUT_DIR / "latest_frame.json"
JSONL_FILE = OUT_DIR / "frames.jsonl"
LATEST_H1_FILE = OUT_DIR / "latest_h1_frame.json"


async def handle_client(websocket):
    print("Client connected")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"ok": False, "error": "invalid json"}))
                continue

            server_frame = {
                "serverUnixMs": int(time.time() * 1000),
                **data,
            }

            # Save browser pose frame
            LATEST_FILE.write_text(json.dumps(server_frame, indent=2), encoding="utf-8")

            with JSONL_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(server_frame) + "\n")

            # Convert to H1 live frame
            h1_frame = pose_to_h1(server_frame)

            LATEST_H1_FILE.write_text(
                json.dumps(h1_frame, indent=2),
                encoding="utf-8",
            )

            frame_index = server_frame.get("frameIndex", "?")
            role = server_frame.get("role", "?")
            time_ms = server_frame.get("timeMs", "?")
            print(f"Received frame {frame_index} from {role} at {time_ms} ms")

            await websocket.send(
                json.dumps(
                    {
                        "ok": True,
                        "saved": True,
                        "frameIndex": frame_index,
                    }
                )
            )

    except websockets.ConnectionClosed:
        print("Client disconnected")


async def main():
    print(f"WebSocket server listening on ws://{HOST}:{PORT}")
    async with websockets.serve(handle_client, HOST, PORT, max_size=20 * 1024 * 1024):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
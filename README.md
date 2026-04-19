# Freehand
Using head and face gestures/movements to control keyboard and mouse controls.

## Setup

**Requires Python 3.11** (mediapipe does not yet support 3.13).

If you don't have Python 3.11, install it via [python.org](https://www.python.org/downloads/release/python-3119/) or:
```bash
# macOS (Homebrew)
brew install python@3.11

# Windows (winget)
winget install Python.Python.3.11
```

Then set up the project:

### Option A: conda (recommended)

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd Freehand

# 2. Create a conda environment with Python 3.11
conda create -n freehand python=3.11 -y
conda activate freehand

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py --run
```

### Option B: venv

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd Freehand

# 2. Create a virtual environment with Python 3.11
python3.11 -m venv venv

# 3. Activate it
# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run
python main.py --run
```

### macOS permissions

Go to **System Settings > Privacy & Security** and grant your terminal app access to:
- **Camera** (for face tracking)
- **Microphone** (for voice commands)
- **Accessibility** (for mouse/keyboard control)

### Windows notes

- `sounddevice` requires [PortAudio](http://www.portaudio.com/). If you get an error, install it or run `pip install sounddevice` which bundles it on most systems.
- You may need to run your terminal as Administrator for `pynput` to control the mouse/keyboard.

## Usage

```bash
python main.py --run           # start cursor control
python main.py --run --debug   # start with camera preview window
python main.py --debug-pose    # print raw head angles (for testing)
```

### Controls
- **Head movement** — moves cursor
- **Blink** — left click
- **Mouth open (hold)** — scroll mode (pitch controls scroll direction)
- **"command drag"** — start dragging
- **"command drop"** — stop dragging
- **"command [key]"** — voice commands (enter, backspace, tab, escape, undo, redo, copy, paste, cut, select all)

# Remote Development Guide: VS Code + Google Colab

This guide explains how to connect your local VS Code to a Google Colab Runtime (Free GPU) to run the `finwise_core` heavy workloads without freezing your machine.

## Why this method?
While there is an official "Google Colab" extension for editing notebooks, the **SSH Tunnel** method turns Colab into a full Remote Server. This allows you to:
- Behave as if the Colab VM is your local machine.
- Run terminal commands (`python core/orchestrator.py`).
- Use the full VS Code Debugger.
- Edit multiple files across the project directory easily.

---

## Prerequisites (Local Machine)

1.  **VS Code Extension**: Install **Remote - SSH** (ms-vscode-remote.remote-ssh).
2.  **Cloudflared** (Optional but recommended): The script uses Cloudflare tunnels. VS Code usually prompts to download the binary automatically, but having it installed helps.

---

## Step 1: Start the Colab Bridge

1.  Log in to [Google Colab](https://colab.research.google.com/).
2.  Upload `finwise_colab_bridge.ipynb` from this repository.
3.  **Runtime > Change runtime type**: Select **T4 GPU** (or preferred GPU).
4.  Run the first cell in the notebook.
    - It installs `colab_ssh`.
    - It launches a Cloudflare tunnel.
    - It prints a **VS Code Configuration** block in the output.

## Step 2: Connect from VS Code

1.  Copy the configuration block from the Colab output. It looks like this:
    ```text
    Host google_colab_ssh
        HostName molecules-expect-...trycloudflare.com
        User root
        IdentityFile ...
    ```
2.  In VS Code, press `F1` (or `Ctrl+Shift+P`) and type **Remote-SSH: Open Configuration File**.
3.  Select your user config file (usually `C:\Users\YourName\.ssh\config`).
4.  Paste the configuration at the bottom and save.
5.  Press `F1` -> **Remote-SSH: Connect to Host...**
6.  Select `google_colab_ssh`.
7.  New window opens. It will ask for the password you defined in the notebook (default in script: `finwise_dev`).

## Step 3: Syncing Code

Once connected, you are in an empty Ubuntu VM (`/root/`).

**Option A: Git (Recommended)**
1.  Open Terminal in the remote VS Code window (`Ctrl+~`).
2.  Clone your repo:
    ```bash
    git clone https://github.com/your-repo/finwise_core.git
    cd finwise_core
    pip install -r requirements.txt
    ```

**Option B: Drag & Drop**
1.  In the Remote VS Code file explorer, drag your local `finwise_core` folder into the remote `/content/` directory.

## Step 4: Running on GPU

Now you can run the heavy scripts on the Colab GPU:

```bash
# In the remote terminal
cd /content/finwise_core
python app/gradio_app.py
```

The Gradio app will launch and provide a public URL (gradio.live) which you can open on your local browser.

---

## Important Notes

-   **Persistence**: Colab runtimes recycle after 12 hours (or earlier if idle). **You must push your code to Git** to save changes. Files on the VM are ephemeral.
-   **Reconnection**: Every time you restart the Colab notebook, the `HostName` url changes. You must update your `.ssh/config` file each time.

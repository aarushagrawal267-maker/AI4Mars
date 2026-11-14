# Deploying the full app for free

The full app (React website + FastAPI + PyTorch) needs about 650 MB of memory and a Docker-capable host. Free plans of
Render (512 MB) and Hugging Face (Docker Spaces need PRO) cannot run it. Free ways that can:

- **GitHub Codespaces** (no card): a free cloud machine with 8 GB RAM and a public link while it runs — see below.
- **Oracle Cloud Always Free** (card for verification): a permanent 24/7 server.
- **Your own computer + Cloudflare Tunnel**: a public link while your computer is on.

## GitHub Codespaces (free, no card)

Personal GitHub accounts include 120 core-hours of Codespaces per month (about 60 hours on the 2-core machine).

1. Open https://codespaces.new/aarushagrawal267-maker/AI4Mars?quickstart=1 (or repository page -> **Code** ->
   **Codespaces** -> **Create codespace on main**). Choose the **2-core / 8 GB** machine if asked.
2. Wait for the first setup (about 10 minutes: it installs PyTorch and the website). The configuration in
   `.devcontainer/` then starts the full app automatically (`deploy/codespace-start.sh`).
3. Open the **PORTS** tab: port **8000** must show **Public** visibility (the script tries to set it; otherwise
   right-click the port -> **Port Visibility** -> **Public**). The **Forwarded Address**
   (`https://<codespace-name>-8000.app.github.dev`) is the public link anyone can open.
4. Admin password for Settings -> Admin token: run `cat .mti_admin_token` in the terminal.

The link works while the codespace runs. It stops after a period without editor activity (default 30 minutes;
up to 4 hours in GitHub -> Settings -> Codespaces -> Default idle timeout). Reopen it from github.com/codespaces —
the app starts again automatically and keeps its data; the link stays the same.

Comparison of the other two options:

| | Oracle Cloud "Always Free" server | Your own computer + Cloudflare Tunnel |
|---|---|---|
| Cost | Free forever (a card is needed once for identity checks; Always Free resources are not charged) | Free |
| Online | 24/7 | Only while your computer is on and the commands are running |
| Address | `http://<server IP>` (permanent) | `https://<random>.trycloudflare.com` (new each start) |
| Features | Everything (training on the server CPU) | Everything (uses your computer) |
| Setup | ~30 minutes, one command on the server | ~5 minutes |

In both cases the app runs in **public mode**: visitors can browse, analyse images and plan routes; training,
evaluation, dataset changes and deletion need the admin password, entered under **Settings -> Admin token**.

## Option 1: Oracle Cloud Always Free (24/7)

1. Create an account at https://www.oracle.com/cloud/free/ (choose a home region close to you; it cannot be changed).
2. Console -> **Compute -> Instances -> Create instance**:
   - Image: **Canonical Ubuntu 22.04** (or 24.04)
   - Shape: **Ampere VM.Standard.A1.Flex**, 2 OCPUs and 12 GB memory (within the Always Free limits).
     If it says "out of capacity", try another availability domain or try again later. (The other free shape,
     VM.Standard.E2.1.Micro, has only 1 GB of memory and is not enough.)
   - Networking: keep "Assign a public IPv4 address" on.
   - SSH keys: **Generate a key pair** and **download the private key**.
   - **Create**, wait until it is *Running*, copy the **Public IP address**.
3. Allow web traffic: instance page -> **Subnet** link -> **Security Lists** -> default list -> **Add Ingress Rules**:
   source `0.0.0.0/0`, IP protocol TCP, destination port `80` -> **Add**.
4. Connect (Windows PowerShell, macOS/Linux terminal):
   ```
   ssh -i path\to\downloaded-key.key ubuntu@YOUR_PUBLIC_IP
   ```
   (Or use **Cloud Shell** / the instance's **Console connection** in the Oracle web console.)
5. On the server, run the one-command setup:
   ```
   curl -fsSL https://raw.githubusercontent.com/aarushagrawal267-maker/AI4Mars/main/deploy/server-setup.sh | bash
   ```
   It installs Docker, builds the app with the trained models, starts it on port 80 with automatic restart, and prints
   the public URL and where the admin password was saved (`cat ~/.mti_admin_token`).
6. Open `http://YOUR_PUBLIC_IP`. Run the same command again later to update to the latest code; data, trained models
   and experiments are kept in Docker volumes.

## Option 2: Your computer + Cloudflare Tunnel (free, while it is on)

After `setup.ps1` (Windows) has been run once:

1. Build the website once:
   ```
   cd frontend
   npm run build
   cd ..
   ```
2. Start the app in public mode (PowerShell; choose your own password):
   ```
   $env:MTI_PUBLIC_MODE="true"; $env:MTI_ADMIN_TOKEN="choose-a-long-password"; $env:MTI_FRONTEND_DIST="$PWD\frontend\dist"
   .venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
   ```
3. In a second PowerShell window, install and start the tunnel:
   ```
   winget install --id Cloudflare.cloudflared
   cloudflared tunnel --url http://localhost:8000
   ```
4. It prints an address like `https://something-random.trycloudflare.com` — anyone can open it while both windows
   stay open. Close them to take the site offline.

# Deploying on Render

This publishes the app at a public `https://<name>.onrender.com` address in **public demo mode**: visitors can
browse the results, analyse their own images with the trained models and plan routes; training, dataset changes,
evaluation, deletion and settings are locked behind an admin token.

The trained models, results database and the 322 AI4Mars test images are committed under `deploy/bundle/`, and
`deploy/render.Dockerfile` copies them into the image, so no extra download link is needed (this also works for a
private GitHub repository).

## Requirements

- A [Render](https://render.com) account (sign in with GitHub) with access to this repository.
- A plan with **at least 2 GB of memory (Standard)**. Measured peak memory is ~550 MB (PyTorch alone uses ~370 MB),
  above the 512 MB of the Free and Starter plans, where the app would be killed.

## Steps

1. Render dashboard -> **New +** -> **Web Service** -> select this repository.
2. Settings:

   | Field | Value |
   |---|---|
   | Branch | `main` |
   | Language / Runtime | Docker |
   | Dockerfile Path | `./deploy/render.Dockerfile` |
   | Docker Build Context | `.` |
   | Instance Type | Standard |
   | Health Check Path (Advanced) | `/api/health` |

3. Environment variables:

   | Key | Value |
   |---|---|
   | `MTI_PUBLIC_MODE` | `true` |
   | `MTI_ADMIN_TOKEN` | a long password of your choice |
   | `MTI_MODEL_CACHE_SIZE` | `1` |
   | `MTI_MAX_OUTPUT_SIDE` | `1024` |

4. **Create Web Service**. The first build takes 10-15 minutes (PyTorch install + website build). It is ready when
   the log shows `[bundle] trained models present in the image` and `Uvicorn running on http://0.0.0.0:10000`.
5. Open the URL shown at the top of the service page.

Alternative: **New +** -> **Blueprint** reads `render.yaml` and fills in the same settings (it generates the admin
token for you: service -> Environment -> reveal `MTI_ADMIN_TOKEN`).

## Admin access

On the website open **Settings -> Admin token**, paste `MTI_ADMIN_TOKEN` and save; locked actions then work from that
browser. The bundle contains no training images, so train locally and commit a new bundle
(`python scripts/deploy_space.py --stage`, then copy `deploy/stage/{data,checkpoints,experiments}` to `deploy/bundle/`).

## Notes

- Analyses made by visitors are lost when the service restarts or redeploys (the image's bundle is restored). Add a
  Render persistent disk at `/app/data` to keep them.
- Every push to `main` redeploys automatically.
- Optional: `MTI_DEMO_BUNDLE_URL` (+ `MTI_DEMO_BUNDLE_SHA256`) lets an image without `deploy/bundle/` download a bundle
  zip at start-up (`python scripts/deploy_space.py --bundle` creates one).
- Free alternative with 16 GB RAM: Hugging Face Spaces, `scripts/deploy_hf_space.py`.

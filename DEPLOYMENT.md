# DeepScan Deployment

This project is a single FastAPI app that serves both the API and the UI.

## Recommended Host

Use a Docker-based host with at least 2 GB RAM. The app downloads Hugging Face model files on first boot if `models/best_model.pt` is not present, so the first deploy can take a few minutes.

Good options:

- Render Web Service, easiest Docker deploy
- Railway, simple Docker deploy
- Fly.io, good if you are comfortable with CLI deploys
- Any VPS with Docker Compose

Avoid static-only hosts like Netlify or Vercel for this exact project because it needs a Python ML backend.

## Environment Variables

Set these in your hosting dashboard:

```env
DEBUG=false
DEVICE=cpu
MAX_FILE_SIZE_MB=200
CORS_ORIGINS=*
```

For CPU-only cloud plans, keep `DEVICE=cpu`. Use `DEVICE=cuda` only on a GPU machine with a CUDA-capable image/runtime.

## Deploy on Render

1. Push this folder to GitHub.
2. In Render, create a new Web Service.
3. Choose your GitHub repository.
4. Select Docker as the runtime.
5. Set environment variables from the section above.
6. Choose a plan with at least 2 GB RAM.
7. Deploy.

Render provides `PORT` automatically. The Dockerfile reads it, so no extra start command is needed.

## Deploy on Railway

1. Push this folder to GitHub.
2. In Railway, create a new project from the GitHub repository.
3. Railway should detect the Dockerfile.
4. Add the environment variables above.
5. Deploy.

Railway also provides `PORT` automatically.

## Deploy on a VPS

Install Docker and Docker Compose, then run:

```bash
docker compose up --build -d
```

Open:

```text
http://YOUR_SERVER_IP:8000
```

For production, put Nginx or Caddy in front of it and point your domain to the server.

## After Deploy

Check these URLs:

```text
https://YOUR_DOMAIN/health
https://YOUR_DOMAIN/docs
https://YOUR_DOMAIN/
```

`/health` should return `model_loaded: true`. If it is false or the app crashes, check the deployment logs. The most common cause is not enough memory during model download/load.

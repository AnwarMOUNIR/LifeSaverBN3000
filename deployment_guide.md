# 🚀 LifeSaverBN3000 Deployment Guide

This guide provides step-by-step instructions for deploying the **LifeSaverBN3000** Medical Decision Support app to various platforms.

## 1. Streamlit Community Cloud (Recommended)
**The "Vercel for Streamlit" experience.** It is completely free, connects directly to your GitHub repo, and handles SSL/deployment automatically.
1. Push your latest code to a public GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Sign in with GitHub and select your repo.
4. Set the **Main file path** to `app/app.py`.
5. Click **Deploy**.

## 2. Docker (Containerized Deployment)
Run the app on any cloud provider (AWS, GC, Azure) or local server.
1. **Build the image**:
   ```bash
   docker build -t lifesaver-app .
   ```
2. **Run the container**:
   ```bash
   docker run -p 8501:8501 lifesaver-app
   ```
3. Access the app at `http://localhost:8501`.

## 3. Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Select **Streamlit** as the SDK.
3. Upload your project files or connect your GitHub repo.
4. Ensure `requirements.txt` is in the root directory.

## 4. Troubleshooting
- **Missing Models**: Ensure `models/best_model.pkl` and `models/label_encoder.pkl` are committed/uploaded.
- **Port Context**: If deploying to a VPS, ensure port `8501` is open in your firewall settings.

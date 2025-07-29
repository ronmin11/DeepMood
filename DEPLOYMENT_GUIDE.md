# DeepMood Deployment Guide

## Backend Deployment on Render

### Step 1: Deploy Backend to Render

1. **Go to Render Dashboard**: Visit [render.com](https://render.com) and sign in
2. **Create New Web Service**: Click "New" → "Web Service"
3. **Connect Repository**: Connect your GitHub repository
4. **Configure Service**:
   - **Name**: `deepmood-backend`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && python app.py`
   - **Plan**: Free

### Step 2: Set Environment Variables

In the Render dashboard for your backend service, add these environment variables:

- **FLASK_ENV**: `production`
- **TOGETHER_API_KEY**: Your Together AI API key (get from [together.ai](https://together.ai))

### Step 3: Deploy

Click "Create Web Service" and wait for deployment to complete.

### Step 4: Update Frontend URLs

The frontend code has been updated to use the correct backend URL:
- Backend URL: `https://deepmood-backend.onrender.com`

## Testing the Deployment

After deployment, test the backend endpoints:

```bash
# Test health endpoint
curl https://deepmood-backend.onrender.com/health

# Test chatbot endpoint
curl -X POST https://deepmood-backend.onrender.com/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "emotion": "neutral"}'
```

## Troubleshooting

### Common Issues:

1. **CORS Errors**: Make sure the frontend URL is in the CORS origins list
2. **API Key Missing**: Ensure TOGETHER_API_KEY is set in Render environment variables
3. **Port Issues**: The app now uses the PORT environment variable from Render

### Debug Commands:

```bash
# Test backend locally
cd backend
python app.py

# Test with curl
curl http://localhost:5000/health
```

## File Structure

```
DeepMood/
├── backend/
│   ├── app.py              # Main Flask application
│   ├── requirements.txt    # Python dependencies
│   └── Procfile           # Deployment configuration
├── frontend/              # React frontend
├── render.yaml            # Render deployment config
└── DEPLOYMENT_GUIDE.md   # This file
```

## Environment Variables

### Backend (Render):
- `FLASK_ENV`: `production`
- `TOGETHER_API_KEY`: Your Together AI API key
- `PORT`: Automatically set by Render

### Frontend:
- `NODE_ENV`: `production` (for production builds) 
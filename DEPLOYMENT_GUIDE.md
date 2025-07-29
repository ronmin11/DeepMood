# DeepMood Full-Stack Deployment Guide

## Full-Stack Deployment on Render

### Step 1: Deploy Full Application to Render

1. **Go to Render Dashboard**: Visit [render.com](https://render.com) and sign in
2. **Create New Web Service**: Click "New" → "Web Service"
3. **Connect Repository**: Connect your GitHub repository
4. **Configure Service**:
   - **Name**: `deepmood-fullstack`
   - **Environment**: `Python`
   - **Build Command**: 
     ```bash
     npm install
     npm run build
     pip install -r backend/requirements.txt
     ```
   - **Start Command**: `cd backend && python app.py`
   - **Plan**: Free

### Step 2: Set Environment Variables

In the Render dashboard for your service, add these environment variables:

- **FLASK_ENV**: `production`
- **TOGETHER_API_KEY**: Your Together AI API key (get from [together.ai](https://together.ai))

### Step 3: Deploy

Click "Create Web Service" and wait for deployment to complete.

## How It Works

This deployment setup:

1. **Builds the Frontend**: The build command installs Node.js dependencies and builds the React app
2. **Installs Python Dependencies**: Installs all required Python packages
3. **Serves Everything from Python**: The Flask app serves both the frontend static files and API endpoints
4. **Single URL**: Everything runs on one URL (e.g., `https://deepmood-fullstack.onrender.com`)

## Testing the Deployment

After deployment, test the application:

```bash
# Test the main application
curl https://deepmood-fullstack.onrender.com

# Test API endpoints
curl https://deepmood-fullstack.onrender.com/api/health
curl -X POST https://deepmood-fullstack.onrender.com/api/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "emotion": "neutral"}'
```

## Local Development

```bash
# Install dependencies
npm run install-deps

# Start development servers
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Troubleshooting

### Common Issues:

1. **Build Failures**: Make sure all dependencies are properly listed
2. **API Key Missing**: Ensure TOGETHER_API_KEY is set in Render environment variables
3. **Static Files Not Found**: The Flask app serves frontend from `../frontend/dist`

### Debug Commands:

```bash
# Test locally
npm run dev

# Test production build
npm run build
npm start
```

## File Structure

```
DeepMood/
├── backend/
│   ├── app.py              # Main Flask application (serves frontend + API)
│   ├── requirements.txt    # Python dependencies
│   └── Procfile           # Deployment configuration
├── frontend/              # React frontend
│   ├── dist/              # Built static files (served by Flask)
│   └── src/               # React source code
├── package.json           # Root package.json for build scripts
├── render.yaml            # Render deployment config
└── DEPLOYMENT_GUIDE.md   # This file
```

## Environment Variables

### Production (Render):
- `FLASK_ENV`: `production`
- `TOGETHER_API_KEY`: Your Together AI API key
- `PORT`: Automatically set by Render

### Development:
- `NODE_ENV`: `development` (for development builds) 
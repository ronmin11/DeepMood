# DeepMood - Emotion-Aware AI Conversations

DeepMood is an advanced application that combines real-time facial emotion recognition with AI-powered therapeutic conversations. The system detects emotions through webcam feed and provides contextually appropriate responses based on the user's emotional state.

## Project Structure

```
DeepMood/
├── frontend/          # React TypeScript frontend with shadcn/ui
├── backend/           # Python Flask backend with AI models
├── README.md         # This file
└── .gitignore        # Git ignore file
```

## Features

- **Real-time Emotion Detection**: Uses computer vision to analyze facial expressions
- **AI-Powered Chatbot**: Context-aware therapeutic conversations using Together AI
- **Modern UI**: Beautiful React interface with shadcn/ui components
- **Webcam Integration**: Live video feed with emotion analysis
- **Responsive Design**: Works on desktop and mobile devices

## Technology Stack

### Frontend
- React 18 with TypeScript
- Vite for build tooling
- shadcn/ui for components
- Tailwind CSS for styling
- React Router for navigation

### Backend
- Python Flask
- Together AI for LLM conversations
- Transformers for emotion classification
- OpenCV for image processing
- CORS enabled for frontend communication

## Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+
- Together AI API key

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Together AI API key:
```bash
# Copy the template file
cp env_template.txt .env

# Edit the .env file and replace 'your_api_key_here' with your actual API key
# The .env file is already in .gitignore, so it won't be committed to git
```

4. Run the Flask backend:
```bash
python app.py
```

The backend will be available at `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## API Endpoints

### Backend API (`http://localhost:5000`)

- `POST /predict` - Analyze emotion from uploaded image
- `POST /chatbot` - Get AI response based on message and emotion
- `GET /health` - Health check endpoint

## Usage

1. Start both the backend and frontend servers
2. Open the frontend in your browser
3. Allow camera access for emotion detection
4. Start a conversation with the AI therapist
5. The system will automatically detect your emotions and provide appropriate responses

## Development

### Backend Development
- The Flask app is in `backend/app.py`
- Add new endpoints in the Flask app
- Update `requirements.txt` for new dependencies

### Frontend Development
- Main app component: `frontend/src/App.tsx`
- Pages: `frontend/src/pages/`
- Components: `frontend/src/components/`
- UI components: `frontend/src/components/ui/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test both frontend and backend
5. Submit a pull request

## License

This project is licensed under the MIT License.

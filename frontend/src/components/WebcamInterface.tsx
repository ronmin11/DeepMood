import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Camera, CameraOff, User } from 'lucide-react';
import { useRef, useCallback } from 'react';

interface EmotionData {
  emotion: string;
  confidence: number;
  color: string;
  allEmotions?: { [key: string]: number };
}

interface WebcamInterfaceProps {
  onEmotionDetected?: (emotion: string) => void;
}

export const WebcamInterface = ({ onEmotionDetected }: WebcamInterfaceProps) => {
  const [isActive, setIsActive] = useState(false);
  const [detectedEmotion, setDetectedEmotion] = useState<EmotionData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const getEmotionColor = (emotion: string) => {
    const emotionColors: { [key: string]: string } = {
      'happy': 'text-green-500',
      'sad': 'text-blue-500',
      'angry': 'text-red-500',
      'surprised': 'text-yellow-500',
      'neutral': 'text-gray-500',
      'fearful': 'text-purple-500',
      'disgusted': 'text-orange-500'
    };
    return emotionColors[emotion.toLowerCase()] || 'text-primary';
  };

  const captureFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) return;

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to blob and send to backend
    canvas.toBlob(async (blob) => {
      if (!blob) return;

      const formData = new FormData();
      formData.append('image', blob, 'frame.jpg');

      try {
        setIsAnalyzing(true);
        // Use localhost for development, deployed URL for production
        const apiUrl = process.env.NODE_ENV === 'production'
          ? '/api/predict'
          : 'http://localhost:5000/api/predict';
          
        const response = await fetch(apiUrl, {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          const emotionData: EmotionData = {
            emotion: result.emotion,
            confidence: result.confidence,
            color: getEmotionColor(result.emotion),
            allEmotions: result.all_predictions || {}
          };
          
          setDetectedEmotion(emotionData);
          onEmotionDetected?.(result.emotion);
        }
      } catch (error) {
        console.error('Error analyzing emotion:', error);
      } finally {
        setIsAnalyzing(false);
      }
    }, 'image/jpeg', 0.8);
  }, [onEmotionDetected]);

  const startCamera = async () => {
    try {
      console.log('Requesting camera access...');
      
      // Try different camera constraints in order of preference
      const constraints = [
        { 
          video: { 
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          } 
        },
        { video: { facingMode: 'user' } },
        { video: true }
      ];
      
      let stream = null;
      let lastError = null;
      
      for (const constraint of constraints) {
        try {
          console.log('Trying constraint:', constraint);
          stream = await navigator.mediaDevices.getUserMedia(constraint);
          console.log('Success with constraint:', constraint);
          break;
        } catch (err) {
          console.log('Failed with constraint:', constraint, 'Error:', err);
          lastError = err;
        }
      }
      
      if (!stream) {
        throw lastError || new Error('All camera constraints failed');
      }
      
      console.log('Camera stream obtained:', stream);
      console.log('Checking video element reference...');
      console.log('videoRef.current:', videoRef.current);
      
      // Set active first so React renders the video element
      setIsActive(true);
      console.log('Set isActive to true, video element should render now');
      
      // If video element doesn't exist, wait and retry
      const setupVideo = () => {
        if (videoRef.current) {
          console.log('Video element found, setting up...');
          videoRef.current.srcObject = stream;
          streamRef.current = stream;
          
          videoRef.current.play().then(() => {
            console.log('Video playing successfully');
            // Start emotion detection every 1 second for more frequent updates
            const interval = setInterval(() => {
              captureFrame();
            }, 1000);
            (videoRef.current as any).emotionInterval = interval;
          }).catch(console.error);
        } else {
          console.log('Video element still not found, retrying in 100ms...');
          setTimeout(setupVideo, 100);
        }
      };
      
      setupVideo();
    } catch (error: any) {
      console.error('Error accessing camera:', error);
      console.error('Error name:', error.name);
      console.error('Error message:', error.message);
      
      let errorMessage = 'Unable to access camera. ';
      
      if (error.name === 'NotAllowedError') {
        errorMessage += 'Please allow camera permissions when prompted.';
      } else if (error.name === 'NotFoundError') {
        errorMessage += 'No camera found on this device.';
      } else if (error.name === 'NotReadableError') {
        errorMessage += 'Camera is already in use by another application.';
      } else if (error.name === 'OverconstrainedError') {
        errorMessage += 'Camera does not support the requested settings.';
      } else if (error.name === 'SecurityError') {
        errorMessage += 'Camera access blocked due to security restrictions. Try using HTTPS.';
      } else {
        errorMessage += `Error: ${error.message || 'Unknown error'}`;
      }
      
      alert(errorMessage);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      if ((videoRef.current as any).emotionInterval) {
        clearInterval((videoRef.current as any).emotionInterval);
      }
    }
    
    setIsActive(false);
    setDetectedEmotion(null);
  };

  const handleToggleCamera = () => {
    if (isActive) {
      stopCamera();
    } else {
      startCamera();
    }
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Webcam Feed */}
      <Card className="relative overflow-hidden bg-gradient-primary border-primary/20 shadow-glow">
        <div className="aspect-video bg-muted/50 flex items-center justify-center relative">
          {isActive ? (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
                style={{ minHeight: '300px', backgroundColor: '#000', border: '2px solid red' }}
                onCanPlay={() => console.log('Video can play')}
                onPlay={() => console.log('Video started playing')}
                onLoadedData={() => console.log('Video loaded data')}
                onLoadedMetadata={() => console.log('Video loaded metadata')}
                onError={(e) => console.error('Video error:', e)}
              />
              <canvas
                ref={canvasRef}
                className="hidden"
              />
              {isAnalyzing && (
                <div className="absolute top-4 left-4 bg-card/90 backdrop-blur-sm rounded-lg p-2 border border-border">
                  <p className="text-xs text-muted-foreground">Analyzing...</p>
                </div>
              )}
            </>
          ) : (
            <div className="text-center">
              <CameraOff className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Camera Inactive</p>
            </div>
          )}
          
          {/* Emotion Overlay */}
          {detectedEmotion && (
            <div className="absolute top-4 right-4 bg-card/90 backdrop-blur-sm rounded-lg p-3 border border-border shadow-card">
              <div className="text-center">
                <p className="text-xs text-muted-foreground mb-1">Detected Emotion</p>
                <p className={`font-semibold ${detectedEmotion.color}`}>
                  {detectedEmotion.emotion}
                </p>
                <p className="text-xs text-muted-foreground">
                  {Math.round(detectedEmotion.confidence * 100)}% confident
                </p>
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Controls */}
      <div className="flex flex-col gap-4">
        <Button
          onClick={handleToggleCamera}
          variant={isActive ? "destructive" : "hero"}
          size="lg"
          className="w-full"
        >
          {isActive ? (
            <>
              <CameraOff className="w-5 h-5" />
              Stop Camera
            </>
          ) : (
            <>
              <Camera className="w-5 h-5" />
              Start Camera
            </>
          )}
        </Button>

        {/* Emotion Analysis */}
        {detectedEmotion && (
          <Card className="p-4 bg-card border-border shadow-card">
            <h3 className="font-semibold mb-3 text-card-foreground">Emotion Analysis</h3>
            
            {/* Primary Emotion */}
            <div className="space-y-2 mb-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Primary Emotion:</span>
                <span className={`font-medium ${detectedEmotion.color}`}>
                  {detectedEmotion.emotion}
                </span>
              </div>
              <div className="w-full bg-muted rounded-full h-2">
                <div
                  className="bg-gradient-accent h-2 rounded-full transition-all duration-500"
                  style={{ width: `${detectedEmotion.confidence * 100}%` }}
                />
              </div>
            </div>
            
            {/* All Emotions */}
            {detectedEmotion.allEmotions && Object.keys(detectedEmotion.allEmotions).length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-card-foreground mb-2">All Detected Emotions:</h4>
                <div className="space-y-1">
                  {Object.entries(detectedEmotion.allEmotions)
                    .sort(([,a], [,b]) => b - a)
                    .map(([emotion, confidence]) => (
                      <div key={emotion} className="flex items-center justify-between text-xs">
                        <span className={`capitalize ${getEmotionColor(emotion)}`}>
                          {emotion}
                        </span>
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-muted rounded-full h-1">
                            <div
                              className="bg-primary h-1 rounded-full transition-all duration-300"
                              style={{ width: `${confidence * 100}%` }}
                            />
                          </div>
                          <span className="text-muted-foreground w-8 text-right">
                            {Math.round(confidence * 100)}%
                          </span>
                        </div>
                      </div>
                    ))
                  }
                </div>
              </div>
            )}
          </Card>
        )}
      </div>
    </div>
  );
};

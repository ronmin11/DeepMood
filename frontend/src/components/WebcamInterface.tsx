import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Camera, CameraOff, User } from 'lucide-react';
import { useRef, useCallback } from 'react';

interface EmotionData {
  emotion: string;
  confidence: number;
  color: string;
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
          : '/api/predict';
          
        const response = await fetch(apiUrl, {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          const emotionData: EmotionData = {
            emotion: result.emotion,
            confidence: result.confidence,
            color: getEmotionColor(result.emotion)
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
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsActive(true);
        
        // Start emotion detection every 3 seconds
        const interval = setInterval(() => {
          if (isActive) {
            captureFrame();
          }
        }, 3000);
        
        // Store interval for cleanup
        (videoRef.current as any).emotionInterval = interval;
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Unable to access camera. Please ensure you have granted camera permissions.');
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

        {/* Emotion History */}
        {detectedEmotion && (
          <Card className="p-4 bg-card border-border shadow-card">
            <h3 className="font-semibold mb-3 text-card-foreground">Current Mood</h3>
            <div className="space-y-2">
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
          </Card>
        )}
      </div>
    </div>
  );
};
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Camera, CameraOff, User } from 'lucide-react';

interface EmotionData {
  emotion: string;
  confidence: number;
  color: string;
}

export const WebcamInterface = () => {
  const [isActive, setIsActive] = useState(false);
  const [detectedEmotion, setDetectedEmotion] = useState<EmotionData | null>(null);

  // Mock emotion detection for demo (you can implement your own logic here)
  const mockEmotions = [
    { emotion: 'Happy', confidence: 0.85, color: 'text-accent' },
    { emotion: 'Calm', confidence: 0.78, color: 'text-secondary' },
    { emotion: 'Focused', confidence: 0.92, color: 'text-primary' },
    { emotion: 'Excited', confidence: 0.67, color: 'text-accent-glow' },
  ];

  const handleToggleCamera = () => {
    setIsActive(!isActive);
    if (!isActive) {
      // Mock emotion detection after 2 seconds
      setTimeout(() => {
        const randomEmotion = mockEmotions[Math.floor(Math.random() * mockEmotions.length)];
        setDetectedEmotion(randomEmotion);
      }, 2000);
    } else {
      setDetectedEmotion(null);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Webcam Feed */}
      <Card className="relative overflow-hidden bg-gradient-primary border-primary/20 shadow-glow">
        <div className="aspect-video bg-muted/50 flex items-center justify-center relative">
          {isActive ? (
            <div className="w-full h-full bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center">
              <div className="text-center">
                <User className="w-16 h-16 mx-auto mb-4 text-primary-glow animate-pulse" />
                <p className="text-sm text-muted-foreground">Camera Active - Analyzing...</p>
              </div>
            </div>
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
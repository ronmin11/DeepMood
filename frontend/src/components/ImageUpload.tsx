import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Upload, Image as ImageIcon, X } from 'lucide-react';

interface EmotionData {
  emotion: string;
  confidence: number;
  color: string;
  allEmotions?: { [key: string]: number };
}

interface ImageUploadProps {
  onEmotionDetected?: (emotion: string) => void;
}

export const ImageUpload = ({ onEmotionDetected }: ImageUploadProps) => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [detectedEmotion, setDetectedEmotion] = useState<EmotionData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      
      // Reset previous results
      setDetectedEmotion(null);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      setIsAnalyzing(true);
      
      // Direct connection to backend - bypass any proxy issues
      const apiUrl = 'https://deep-mood.vercel.app/';
      
      // Add mode: 'cors' to explicitly handle cross-origin
        
      console.log('Making upload request to:', apiUrl);
      console.log('NODE_ENV:', process.env.NODE_ENV);
      console.log('FormData contents:', formData.get('image'));
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        mode: 'cors',
        body: formData,
        // Don't set Content-Type for FormData, let browser set it automatically
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
      } else {
        console.error('Upload failed with status:', response.status);
        console.error('Response headers:', response.headers);
        
        let errorMessage = 'Unknown error';
        
        if (response.status === 404) {
          errorMessage = 'Upload endpoint not found. Please restart the backend server to pick up the new /api/upload route.';
        } else {
          try {
            const error = await response.json();
            errorMessage = error.error || 'Unknown error';
          } catch (e) {
            // If we can't parse JSON, it's likely an HTML error page
            errorMessage = `Server error (${response.status}). Please check the backend server.`;
          }
        }
        
        alert(`Error analyzing image: ${errorMessage}`);
      }
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert('Error analyzing image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setDetectedEmotion(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Image Upload Area */}
      <Card className="relative overflow-hidden bg-gradient-primary border-primary/20 shadow-glow">
        <div className="aspect-video bg-muted/50 flex items-center justify-center relative">
          {imagePreview ? (
            <>
              <img
                src={imagePreview}
                alt="Selected for analysis"
                className="w-full h-full object-contain"
                style={{ maxHeight: '400px' }}
              />
              <button
                onClick={clearImage}
                className="absolute top-2 right-2 bg-destructive text-destructive-foreground rounded-full p-1 hover:bg-destructive/80 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
              {isAnalyzing && (
                <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                  <div className="bg-card/90 backdrop-blur-sm rounded-lg p-4 border border-border">
                    <p className="text-sm text-card-foreground">Analyzing image...</p>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-center p-8">
              <ImageIcon className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
              <p className="text-sm text-muted-foreground mb-4">
                Upload an image to analyze emotions
              </p>
              <Button onClick={handleUploadClick} variant="outline">
                <Upload className="w-4 h-4 mr-2" />
                Select Image
              </Button>
            </div>
          )}
          
          {/* Emotion Overlay */}
          {detectedEmotion && (
            <div className="absolute top-4 left-4 bg-card/90 backdrop-blur-sm rounded-lg p-3 border border-border shadow-card">
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

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleImageSelect}
        className="hidden"
      />

      {/* Controls */}
      <div className="flex flex-col gap-4">
        <div className="flex gap-3">
          <Button
            onClick={handleUploadClick}
            variant="outline"
            className="flex-1"
          >
            <Upload className="w-4 h-4 mr-2" />
            {selectedImage ? 'Change Image' : 'Upload Image'}
          </Button>
          
          {selectedImage && (
            <Button
              onClick={analyzeImage}
              disabled={isAnalyzing}
              variant="hero"
              className="flex-1"
            >
              {isAnalyzing ? 'Analyzing...' : 'Analyze Emotion'}
            </Button>
          )}
        </div>

        {/* Emotion Analysis Results */}
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
import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { ChatInterface } from '@/components/ChatInterface';
import { Brain, MessageCircle } from 'lucide-react';

const Chatbot = () => {
  const [currentEmotion, setCurrentEmotion] = useState<string>('neutral');

  useEffect(() => {
    // Check if emotion was passed via URL params
    const urlParams = new URLSearchParams(window.location.search);
    const emotionParam = urlParams.get('emotion');
    if (emotionParam) {
      setCurrentEmotion(emotionParam);
    }
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <a href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity cursor-pointer">
              <Brain className="w-8 h-8 text-primary" />
              <h1 className="text-2xl font-bold text-foreground">DeepMood</h1>
            </a>
            <nav className="hidden md:flex items-center gap-6">
              <a href="/" className="text-muted-foreground hover:text-foreground transition-smooth">Home</a>
              <a href="/chatbot" className="text-primary font-semibold">Chatbot</a>
              <a href="#features" className="text-muted-foreground hover:text-foreground transition-smooth">Features</a>
              <a href="#contact" className="text-muted-foreground hover:text-foreground transition-smooth">Contact</a>
            </nav>
          </div>
        </div>
      </header>

      {/* Chatbot Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            {/* Page Header */}
            <div className="text-center mb-12">
              <div className="flex items-center justify-center gap-3 mb-4">
                <MessageCircle className="w-8 h-8 text-primary" />
                <h1 className="text-3xl md:text-4xl font-bold text-foreground">AI Therapist Chatbot</h1>
              </div>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                Have a conversation with our AI therapist. The chatbot will provide empathetic and supportive responses.
              </p>
              {currentEmotion && currentEmotion !== 'neutral' && (
                <div className="mt-4 p-3 bg-primary/10 rounded-lg inline-block">
                  <p className="text-sm text-muted-foreground">
                    Detected emotion: <span className="font-semibold text-primary capitalize">{currentEmotion}</span>
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    The AI will tailor responses based on your emotional state
                  </p>
                </div>
              )}
            </div>

            {/* Chat Interface */}
            <Card className="p-6 bg-card border-border shadow-card">
              <ChatInterface detectedEmotion={currentEmotion} />
            </Card>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Chatbot; 

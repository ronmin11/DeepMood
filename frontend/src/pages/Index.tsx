import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { WebcamInterface } from '@/components/WebcamInterface';
import { Brain, Sparkles, MessageCircle, Camera } from 'lucide-react';

const Index = () => {
  const [currentEmotion, setCurrentEmotion] = useState<string>();
  const [isDemoActive, setIsDemoActive] = useState(false);

  const handleEmotionDetected = (emotion: string) => {
    setCurrentEmotion(emotion);
  };

  const handleStartDemo = () => {
    setIsDemoActive(true);
    // Scroll to the demo section
    const demoSection = document.getElementById('demo');
    if (demoSection) {
      demoSection.scrollIntoView({ behavior: 'smooth' });
    }
  };
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Brain className="w-8 h-8 text-primary" />
              <h1 className="text-2xl font-bold text-foreground">DeepMood</h1>
            </div>
            <nav className="hidden md:flex items-center gap-6">
              <a href="/chatbot" className="text-muted-foreground hover:text-foreground transition-smooth">Chatbot</a>
              <a href="#features" className="text-muted-foreground hover:text-foreground transition-smooth">Features</a>
              <a href="#technology" className="text-muted-foreground hover:text-foreground transition-smooth">Technology</a>
              <a href="#contact" className="text-muted-foreground hover:text-foreground transition-smooth">Contact</a>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 md:py-28 relative overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-background to-accent/5"></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,hsl(var(--primary)/0.1),transparent_50%)]"></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_80%,hsl(var(--accent)/0.08),transparent_50%)]"></div>
        
        <div className="container mx-auto px-4 text-center relative z-10">
          <div className="max-w-4xl mx-auto">
            {/* Decorative Elements */}
            <div className="flex items-center justify-center gap-4 mb-8">
              <div className="w-12 h-[1px] bg-gradient-to-r from-transparent to-primary"></div>
              <Brain className="w-8 h-8 text-primary animate-pulse" />
              <div className="w-12 h-[1px] bg-gradient-to-l from-transparent to-primary"></div>
            </div>
            
            <h1 className="text-4xl md:text-6xl font-bold mb-6 leading-tight">
              <span className="text-foreground">Emotion-Aware AI</span>
              <span className="block bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent animate-pulse">
                Conversations
              </span>
            </h1>
            
            <div className="relative">
              <p className="text-xl text-muted-foreground mb-8 leading-relaxed max-w-2xl mx-auto">
                Advanced facial recognition technology that detects emotions in real-time, 
                enabling more empathetic and contextually appropriate AI interactions.
              </p>
              
              {/* Subtle glow effect */}
              <div className="absolute -inset-4 bg-gradient-to-r from-primary/10 via-transparent to-accent/10 blur-3xl opacity-30 -z-10"></div>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button 
                size="lg" 
                className="text-base group relative overflow-hidden"
                onClick={handleStartDemo}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-primary to-accent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <Camera className="w-5 h-5 relative z-10" />
                <span className="relative z-10">Start Demo</span>
              </Button>
              <Button variant="outline" size="lg" className="text-base border-primary/20 hover:border-primary/40 hover:bg-primary/5" onClick={() => window.location.href = '/chatbot'}>
                <MessageCircle className="w-5 h-5" />
                Chatbot
              </Button>
            </div>
            
            {/* Current Emotion Display */}
            {currentEmotion && (
              <div className="mt-8 p-4 bg-card/50 backdrop-blur-sm rounded-lg border border-border inline-block">
                <p className="text-sm text-muted-foreground mb-1">Current Detected Emotion:</p>
                <p className="text-lg font-semibold text-primary capitalize">{currentEmotion}</p>
              </div>
            )}
            
            {/* Floating Elements */}
            <div className="absolute top-20 left-10 w-2 h-2 bg-primary/30 rounded-full animate-bounce" style={{animationDelay: '0s'}}></div>
            <div className="absolute top-32 right-16 w-1 h-1 bg-accent/40 rounded-full animate-bounce" style={{animationDelay: '1s'}}></div>
            <div className="absolute bottom-20 left-20 w-1.5 h-1.5 bg-primary/20 rounded-full animate-bounce" style={{animationDelay: '2s'}}></div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-secondary/30">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Key Features
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Cutting-edge technology that understands human emotion and responds intelligently
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto mb-16">
            <div className="text-center">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <Brain className="w-8 h-8 text-primary" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">AI-Powered Analysis</h3>
              <p className="text-muted-foreground">
                Advanced neural networks analyze facial expressions with high accuracy
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <Sparkles className="w-8 h-8 text-primary" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">Real-time Processing</h3>
              <p className="text-muted-foreground">
                Instant emotion recognition with continuous monitoring and updates
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <MessageCircle className="w-8 h-8 text-primary" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">Adaptive Responses</h3>
              <p className="text-muted-foreground">
                Contextually appropriate conversations based on detected emotions
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Main Interface */}
      <section id="demo" className="py-20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Try the Technology
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Experience real-time emotion detection and AI conversation in action
            </p>
          </div>
          
          <div className="max-w-4xl mx-auto">
            {/* Webcam Section */}
            <Card className="p-8 bg-card border-border shadow-card">
              <div className="flex items-center gap-3 mb-6">
                <Camera className="w-6 h-6 text-primary" />
                <h3 className="text-xl font-semibold text-card-foreground">
                  Emotion Detection
                </h3>
                {isDemoActive && (
                  <div className="ml-auto">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      Demo Active
                    </span>
                  </div>
                )}
              </div>
              <p className="text-muted-foreground mb-6">
                Enable your camera to analyze facial expressions and detect emotional states in real-time.
              </p>
              <WebcamInterface onEmotionDetected={handleEmotionDetected} />
              
              {/* Chatbot Button at bottom of camera section */}
              <div className="mt-8 pt-6 border-t border-border">
                <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
                  <div className="text-center sm:text-left">
                    <h4 className="font-semibold text-card-foreground mb-1">
                      Ready to talk?
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      Chat with our AI therapist for personalized support
                    </p>
                  </div>
                  <div className="flex gap-3">
                    <Button 
                      variant="outline"
                      onClick={() => window.location.href = '/chatbot'}
                      className="group"
                    >
                      <MessageCircle className="w-4 h-4 mr-2 group-hover:animate-pulse" />
                      Start Chatting
                    </Button>
                    {currentEmotion && (
                      <Button 
                        onClick={() => window.location.href = `/chatbot?emotion=${currentEmotion}`}
                        className="group"
                      >
                        <MessageCircle className="w-4 h-4 mr-2" />
                        Chat with {currentEmotion} context
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-12 bg-secondary/20">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-4 gap-8 mb-8">
            <div>
              <div className="flex items-center gap-3 mb-4">
                <Brain className="w-6 h-6 text-primary" />
                <h3 className="text-lg font-semibold text-foreground">DeepMood</h3>
              </div>
              <p className="text-muted-foreground text-sm">
                Advanced emotion-aware AI technology for more empathetic digital interactions.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold text-foreground mb-3">Technology</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>Facial Recognition</li>
                <li>Machine Learning</li>
                <li>Natural Language Processing</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold text-foreground mb-3">Company</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>About Us</li>
                <li>Careers</li>
                <li>Privacy Policy</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold text-foreground mb-3">Contact</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>support@deepmood.com</li>
                <li>+1 (555) 123-4567</li>
                <li>San Francisco, CA</li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-border pt-8 text-center">
            <p className="text-muted-foreground text-sm">
              © 2024 DeepMood Technologies. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
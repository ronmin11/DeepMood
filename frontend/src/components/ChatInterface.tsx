import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Send, Bot, User, Loader2 } from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  emotion?: string;
  isLoading?: boolean;
}

interface ChatInterfaceProps {
  detectedEmotion?: string;
}

export const ChatInterface = ({ detectedEmotion }: ChatInterfaceProps) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'bot',
      content: 'Hello! I\'m your DeepMood assistant. I can help tailor our conversation based on your current emotional state. How are you feeling today?',
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Test backend connection
  const testBackendConnection = async () => {
    try {
      const apiUrl = process.env.NODE_ENV === 'production' 
        ? '/api/health'
        : 'http://localhost:5000/api/health';
      
      console.log('Testing backend connection to:', apiUrl);
      
      const response = await fetch(apiUrl, {
        method: 'GET',
      });

      console.log('Health check response status:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        console.log('Health check response:', data);
        return true;
      } else {
        const errorText = await response.text();
        console.error('Health check failed:', errorText);
        return false;
      }
    } catch (error) {
      console.error('Health check error:', error);
      return false;
    }
  };

  // Test connection on component mount
  useEffect(() => {
    testBackendConnection();
  }, []);

  // API call to backend chatbot
  const sendMessageToBackend = async (userMessage: string, emotion?: string) => {
    try {
      // Use localhost for development, deployed URL for production
      const apiUrl = process.env.NODE_ENV === 'production' 
        ? '/api/chatbot'
        : 'http://localhost:5000/api/chatbot';
      
      console.log('Environment:', process.env.NODE_ENV);
      console.log('Sending request to:', apiUrl);
      console.log('Request data:', { message: userMessage, emotion: emotion || 'neutral' });
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          emotion: emotion || 'neutral'
        }),
      });

      console.log('Response status:', response.status);
      console.log('Response status text:', response.statusText);
      console.log('Response headers:', Object.fromEntries(response.headers.entries()));
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        throw new Error(`Failed to get response from chatbot: ${response.status} ${response.statusText}`);
      }

      const responseText = await response.text();
      console.log('Raw response:', responseText);
      
      let data;
      try {
        data = JSON.parse(responseText);
      } catch (parseError) {
        console.error('Failed to parse JSON:', parseError);
        console.error('Response text:', responseText);
        throw new Error('Server returned invalid JSON response');
      }
      
      console.log('Parsed response:', data);
      return data.reply;
    } catch (error) {
      console.error('Error sending message to backend:', error);
      return `I'm sorry, I'm having trouble connecting right now. Error: ${error.message}`;
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    setIsTyping(true);
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputValue;
    setInputValue('');

    // Add loading message
    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: 'bot',
      content: 'Thinking...',
      timestamp: new Date(),
      isLoading: true,
    };
    setMessages(prev => [...prev, loadingMessage]);

    try {
      // Get response from backend
      const botResponse = await sendMessageToBackend(currentInput, detectedEmotion);
      
      // Replace loading message with actual response
      setMessages(prev => prev.map(msg => 
        msg.id === loadingMessage.id 
          ? { ...msg, content: botResponse, emotion: detectedEmotion, isLoading: false }
          : msg
      ));
    } catch (error) {
      // Replace loading message with error
      setMessages(prev => prev.map(msg => 
        msg.id === loadingMessage.id 
          ? { ...msg, content: "I'm sorry, I'm having trouble connecting right now. Please try again later.", isLoading: false }
          : msg
      ));
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Chat Header */}
      <Card className="p-4 mb-4 bg-gradient-primary border-primary/20 shadow-glow">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-accent/20 flex items-center justify-center">
            <Bot className="w-5 h-5 text-accent" />
          </div>
          <div>
            <h3 className="font-semibold text-primary-foreground">DeepMood Assistant</h3>
            <p className="text-sm text-primary-foreground/70">
              {detectedEmotion ? `Adapting to your ${detectedEmotion.toLowerCase()} mood` : 'Ready to chat'}
            </p>
          </div>
        </div>
      </Card>

      {/* Messages */}
      <Card className="flex-1 p-4 bg-card border-border shadow-card mb-4 overflow-hidden">
        <div className="h-full overflow-y-auto space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex items-start gap-3 ${
                message.type === 'user' ? 'flex-row-reverse' : 'flex-row'
              }`}
            >
              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                message.type === 'user' 
                  ? 'bg-primary text-primary-foreground' 
                  : 'bg-secondary text-secondary-foreground'
              }`}>
                {message.type === 'user' ? (
                  <User className="w-4 h-4" />
                ) : (
                  <Bot className="w-4 h-4" />
                )}
              </div>
              <div className={`max-w-[70%] p-3 rounded-lg ${
                message.type === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground'
              }`}>
                <div className="flex items-center gap-2">
                  {message.isLoading && (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  )}
                  <p className="text-sm">{message.content}</p>
                </div>
                <p className={`text-xs mt-1 opacity-70`}>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </Card>

      {/* Input */}
      <div className="flex gap-2">
        <Input
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Type your message..."
          className="flex-1 bg-input border-border focus:ring-primary"
        />
        <Button 
          onClick={handleSendMessage}
          variant="accent"
          size="icon"
          disabled={!inputValue.trim() || isTyping}
        >
          {isTyping ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </Button>
      </div>
    </div>
  );
};

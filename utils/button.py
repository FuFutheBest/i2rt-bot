#!/usr/bin/env python3
"""
Simple robot control button class
For controlling robot movement pause and resume

Author: AI Assistant
"""

import threading
import time


class Button:
    """Simple robot control button class"""
    
    def __init__(self, key: str = 'space'):
        """
        Initialize button
        
        Args:
            key: Key binding for the button
        """
        self.key = key
        self.pressed = False
        self.running = False
        self.listener_thread = None
        self.lock = threading.Lock()
        
    def start_listener(self):
        """Start key listener"""
        if self.running:
            return
            
        self.running = True
        self.listener_thread = threading.Thread(target=self._listen, daemon=True)
        self.listener_thread.start()
        
    def stop_listener(self):
        """Stop key listener"""
        self.running = False
        if self.listener_thread:
            self.listener_thread.join(timeout=1.0)
    
    def is_pressed(self) -> bool:
        """Check if button was pressed (one-time check, resets after checking)"""
        with self.lock:
            if self.pressed:
                self.pressed = False
                return True
            return False
    
    def _listen(self):
        """Listen for key presses"""
        try:
            # Try using pynput library, no root permission required
            from pynput import keyboard as pynput_keyboard
            print(f"Starting to listen for key '{self.key}' (using pynput)")
            
            def on_press(key):
                try:
                    # Handle normal keys
                    if hasattr(key, 'char') and key.char == self.key:
                        with self.lock:
                            self.pressed = True
                        print(f"Detected key '{self.key}' pressed")
                except AttributeError:
                    # Handle special keys
                    if self.key == 'space' and key == pynput_keyboard.Key.space:
                        with self.lock:
                            self.pressed = True
                        print(f"Detected space key pressed")
                    elif self.key == 'enter' and key == pynput_keyboard.Key.enter:
                        with self.lock:
                            self.pressed = True
                        print(f"Detected enter key pressed")
            
            # Start listener
            with pynput_keyboard.Listener(on_press=on_press) as listener:
                while self.running:
                    time.sleep(0.1)
                listener.stop()
                
        except ImportError:
            print("pynput module not installed, trying keyboard module...")
            self._try_keyboard()
        except Exception as e:
            print(f"pynput listening error: {e}")
            self._try_keyboard()
    
    def _try_keyboard(self):
        """Try using keyboard module"""
        try:
            import keyboard
            print(f"Starting to listen for key '{self.key}' (using keyboard, may need root permission)")
            
            while self.running:
                if keyboard.is_pressed(self.key):
                    with self.lock:
                        self.pressed = True
                    print(f"Detected key '{self.key}' pressed")
                    # Prevent repeated triggering
                    time.sleep(0.2)
                time.sleep(0.01)
                
        except ImportError:
            print(f"keyboard module not installed")
            print(f"Please install: pip install pynput or pip install keyboard")
            print(f"Recommend using pynput, no root permission required")
        except Exception as e:
            print(f"keyboard listening error: {e}")
            if "root" in str(e).lower():
                print(f"keyboard module requires root permission, try: sudo python your_script.py")
                print(f"or install pynput: pip install pynput")

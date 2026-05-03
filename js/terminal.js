// BikeGuard Terminal Animation
class TerminalAnimation {
  constructor() {
    this.terminalContent = document.getElementById('terminalContent');
    this.lines = [
      '> Loading BikeGuard v0.2.0-alpha...',
      '> DirectML context: initialized',
      '> Loading BARON-v1 calibration...',
      '> QA Suite: running...',
      '> QA Suite: 47/47 tests passing ✓',
      '> Telemetry logger: active',
      '> Awaiting frame input...',
      '> [SYSTEM READY]'
    ];
    
    this.currentLine = 0;
    this.isRunning = false;
    this.init();
  }
  
  init() {
    this.startAnimation();
  }
  
  startAnimation() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.clearTerminal();
    this.typeLines();
  }
  
  clearTerminal() {
    this.terminalContent.innerHTML = '';
    this.currentLine = 0;
  }
  
  typeLines() {
    if (this.currentLine >= this.lines.length) {
      // All lines typed, wait 4 seconds then restart
      setTimeout(() => {
        this.startAnimation();
      }, 4000);
      return;
    }
    
    const line = this.lines[this.currentLine];
    const lineElement = document.createElement('div');
    lineElement.className = 'terminal-line';
    lineElement.textContent = line;
    
    this.terminalContent.appendChild(lineElement);
    
    // Trigger animation
    setTimeout(() => {
      lineElement.style.opacity = '1';
    }, 50);
    
    this.currentLine++;
    
    // Type next line after 1.5 seconds
    setTimeout(() => {
      this.typeLines();
    }, 1500);
  }
  
  // Method to restart animation manually
  restart() {
    this.isRunning = false;
    setTimeout(() => {
      this.startAnimation();
    }, 100);
  }
}

// Initialize terminal animation when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('terminalContent')) {
    new TerminalAnimation();
  }
});

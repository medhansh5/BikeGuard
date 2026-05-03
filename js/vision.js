// BikeGuard Vision Simulation
class VisionSimulation {
  constructor() {
    this.boundingBox = document.getElementById('boundingBox');
    this.riderLabel = document.getElementById('riderLabel');
    this.helmetLabel = document.getElementById('helmetLabel');
    this.pillionLabel = document.getElementById('pillionLabel');
    this.pillionHelmetLabel = document.getElementById('pillionHelmetLabel');
    
    this.states = [
      {
        rider: 'RIDER: DETECTED',
        helmet: 'HELMET: COMPLIANT',
        pillion: 'PILLION: DETECTED',
        pillionHelmet: 'HELMET: COMPLIANT',
        boxColor: '#22C55E',
        helmetColor: '#22C55E'
      },
      {
        rider: 'RIDER: DETECTED',
        helmet: 'HELMET: NON-COMPLIANT',
        pillion: 'PILLION: DETECTED',
        pillionHelmet: 'HELMET: COMPLIANT',
        boxColor: '#FF3333',
        helmetColor: '#FF3333'
      },
      {
        rider: 'RIDER: DETECTED',
        helmet: 'HELMET: COMPLIANT',
        pillion: 'PILLION: DETECTED',
        pillionHelmet: 'HELMET: COMPLIANT',
        boxColor: '#22C55E',
        helmetColor: '#22C55E'
      }
    ];
    
    this.currentState = 0;
    this.init();
  }
  
  init() {
    // Start label cycling
    setInterval(() => this.cycleLabels(), 3000);
    
    // Start box animation
    this.animateBox();
  }
  
  cycleLabels() {
    const state = this.states[this.currentState];
    
    // Update labels with fade effect
    this.fadeAndUpdate(this.riderLabel, state.rider);
    this.fadeAndUpdate(this.helmetLabel, state.helmet, state.helmetColor);
    this.fadeAndUpdate(this.pillionLabel, state.pillion);
    this.fadeAndUpdate(this.pillionHelmetLabel, state.pillionHelmet);
    
    // Update box color
    this.boundingBox.style.stroke = state.boxColor;
    
    // Flash red for non-compliant helmet
    if (state.helmet === 'HELMET: NON-COMPLIANT') {
      this.flashNonCompliant();
    }
    
    this.currentState = (this.currentState + 1) % this.states.length;
  }
  
  fadeAndUpdate(element, text, color = '#22C55E') {
    element.style.opacity = '0';
    element.style.fill = color;
    
    setTimeout(() => {
      element.textContent = text;
      element.style.opacity = '1';
    }, 150);
  }
  
  flashNonCompliant() {
    let flashes = 0;
    const flashInterval = setInterval(() => {
      this.boundingBox.style.opacity = flashes % 2 === 0 ? '0.3' : '1';
      flashes++;
      
      if (flashes >= 6) { // 3 flashes = 6 state changes
        clearInterval(flashInterval);
        this.boundingBox.style.opacity = '1';
      }
    }, 200);
  }
  
  animateBox() {
    // Box animation is handled by CSS
    // This method can be extended for more complex animations
  }
}

// Initialize vision simulation when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('visionCanvas')) {
    new VisionSimulation();
  }
});
